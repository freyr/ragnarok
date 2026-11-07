"""Check notes against the knowledge-base frontmatter model, their links, their reference targets and the identity keys.

The checks:
1. the keys and values of the model (errors);
2. the required keys `class`, and `validity` for a state or a proposal (errors, only for a draft or a given path);
3. keys outside the model, the existing keys and the configured known keys (warnings);
4. each wiki-link of the body (outside code) and of the frontmatter resolves to a note (errors);
5. each `subject`, `owner`, `supersedes` and `superseded_by` target exists; `subject` and `owner` are entities;
6. the target of `superseded_by` lists the note in its `supersedes`;
7. at most one current state note for each identity key (type, primary subject, task).

A note is read from its file; the other notes, the entities and the references come from the index.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import frontmatter

from .config import Config
from .index import ENTITY_KEYS, REF_KEYS, WIKI_LINK, _link_key, _Resolver
from .metadata import NoteMeta, link_targets, normalize_alias, read_note, resolve

VALIDITY = {
    "state": ("current", "superseded"),
    "proposal": ("open", "accepted", "rejected", "superseded"),
    "record": (),
}
ALL_VALIDITY = ("current", "superseded", "open", "accepted", "rejected")
MODEL_KEYS = ("class", "validity", "as_of", "subject", "task", "supersedes", "superseded_by", "owner")
EXISTING_KEYS = ("project", "area", "type", "status", "description")
ENTITY_ONLY_KEYS = ("kind", "aliases", "facts")  # known only on an entity note
OWNER_KINDS = ("person", "team")
# a link to one of these files is an attachment, not a note
ATTACHMENTS = {
    "png", "jpg", "jpeg", "gif", "svg", "webp", "bmp", "avif", "heic", "tif", "tiff", "pdf",
    "mp3", "wav", "ogg", "m4a", "flac", "webm", "mp4", "mov", "mkv", "ogv", "3gp", "canvas", "excalidraw",
}  # fmt: skip
DRAFT = "(draft)"
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
LINK_TEXT = re.compile(r"\[\[([^\[\]]+)\]\]")
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")
INLINE_CODE = re.compile(r"(`+).+?\1")
ERROR, WARNING = "error", "warning"


@dataclass
class Issue:
    path: str
    check: int
    severity: str
    kind: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    notes: int = 0
    issues: list[Issue] = field(default_factory=list)
    counts: dict | None = None  # the whole-index scan only

    @property
    def errors(self) -> int:
        return sum(i.severity == ERROR for i in self.issues)

    @property
    def warnings(self) -> int:
        return sum(i.severity == WARNING for i in self.issues)

    def as_dict(self) -> dict:
        out = {
            "notes": self.notes,
            "error_count": self.errors,
            "warning_count": self.warnings,
            "issues": [i.as_dict() for i in self.issues],
        }
        if self.counts is not None:
            out["counts"] = self.counts
        return out


# --- values ------------------------------------------------------------------------------------------------------


def _label(value) -> str | None:
    text = str(value).strip().lower() if value is not None else ""
    return text or None


def _link(value) -> str | None:
    """The target of one wiki-link value: the string "[[x]]", or the YAML form of an unquoted [[x]] (nested lists)."""
    if isinstance(value, str):
        m = LINK_TEXT.fullmatch(value.strip())
        return m.group(1) if m else None
    depth = 0
    while isinstance(value, list) and len(value) == 1:
        value, depth = value[0], depth + 1
    if depth and isinstance(value, str) and value.strip() and "[" not in value:
        return value
    return None


def _links(value) -> list[str] | None:
    """The targets of a list of wiki-links, or None when the value is not such a list."""
    if not isinstance(value, list):
        return None
    targets = [_link(v) for v in value]
    return targets if all(t is not None for t in targets) else None


def _clean(target: str) -> str:
    """A link target with no escape of a table pipe and no outer spaces."""
    return target.strip().rstrip("\\").strip()


def body_links(body: str) -> list[str]:
    """The wiki-link targets of a body, outside fenced code blocks and inline code."""
    kept, fence = [], None
    for line in body.splitlines():
        m = FENCE.match(line)
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line[m.end() :].strip():
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        kept.append(INLINE_CODE.sub("", line))
    return [_clean(m.group(1)) for m in WIKI_LINK.finditer("\n".join(kept))]


def _value_links(value) -> list[str]:
    if isinstance(value, str):
        return [_clean(m.group(1)) for m in WIKI_LINK.finditer(value)]
    if isinstance(value, list):
        unquoted = _link(value) if len(value) == 1 and isinstance(value[0], list) else None
        return [_clean(unquoted)] if unquoted else [t for v in value for t in _value_links(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _value_links(v)]
    return []


def frontmatter_links(fm: dict) -> list[str]:
    """The wiki-link targets in the frontmatter values, except the reference keys (check 5 has them)."""
    return [t for key, value in fm.items() if str(key) not in REF_KEYS for t in _value_links(value)]


def _attachment(target: str) -> bool:
    name = target.rsplit("/", 1)[-1]
    return "." in name and name.rsplit(".", 1)[1].lower() in ATTACHMENTS


def _names(target: str, path: str) -> bool:
    """True when a link target names the note at path by the path rule of the links."""
    key, own = _link_key(target).strip("/"), _link_key(path)
    return bool(key) and own.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1] and own.endswith(key)


def _task(fm: dict) -> str:
    task = fm.get("task")
    return normalize_alias(task) if isinstance(task, str) else ""


# --- one note ----------------------------------------------------------------------------------------------------


@dataclass
class _Note:
    label: str  # the path, or DRAFT
    path: str | None
    note_id: int | None
    meta: NoteMeta
    issues: list[Issue] = field(default_factory=list)
    identity: tuple[str, int, str] | None = None
    current: bool = False
    labeled: bool = True

    def add(self, check: int, kind: str, message: str, severity: str = ERROR) -> None:
        self.issues.append(Issue(self.label, check, severity, kind, message))


class _Context:
    def __init__(self, cfg: Config, conn: sqlite3.Connection):
        self.cfg, self.conn = cfg, conn
        self.resolver = _Resolver(conn)
        self.kinds = dict(conn.execute("SELECT note_id, kind FROM entities").fetchall())
        self.ids = dict(conn.execute("SELECT path, id FROM notes").fetchall())

    def note(self, text: str, path: str | None, strict: bool) -> _Note:
        try:
            post = frontmatter.loads(text)
            fm, body, valid = dict(post.metadata), post.content, True
        except Exception:
            fm, body, valid = {}, text, False
        if path:
            meta = resolve(self.cfg, path, fm, body)
        else:
            meta = resolve(self.cfg, "draft.md", fm, body, lambda f: False, lambda f: False)
        note = _Note(path or DRAFT, path, self.ids.get(path) if path else None, meta)
        if not valid:
            note.add(1, "bad-yaml", "the frontmatter is not valid YAML")
        self._model(note, fm)
        self._required(note, fm, strict)
        self._keys(note, fm)
        self._links(note, fm, body)
        successor, primary = self._refs(note, fm)
        self._successor(note, successor)
        if primary is not None:
            note.identity = (meta.type, primary, _task(fm))
        note.current = meta.note_class == "state" and meta.validity in (None, "current") and meta.status != "archived"
        return note

    def _model(self, note: _Note, fm: dict) -> None:
        """Check 1: the values of the model keys."""
        cls, validity = _label(fm.get("class")), _label(fm.get("validity"))
        if cls is not None and cls not in VALIDITY:
            note.add(1, "bad-class", f"class {fm['class']!r} is not state, proposal or record")
        if validity is not None:
            if cls == "record":
                note.add(1, "bad-validity", f"a record has no validity, the note has {validity!r}")
            elif cls in VALIDITY and validity not in VALIDITY[cls]:
                allowed = ", ".join(VALIDITY[cls])
                note.add(1, "bad-validity", f"validity {validity!r} is not allowed for a {cls}; use {allowed}")
            elif cls not in VALIDITY and validity not in ALL_VALIDITY:
                note.add(1, "bad-validity", f"validity {validity!r} is not {', '.join(ALL_VALIDITY)}")
        as_of = fm.get("as_of")
        if as_of is not None and not _iso_date(as_of):
            note.add(1, "bad-as-of", f"as_of {str(as_of)!r} is not an ISO date (YYYY-MM-DD)")
        for key in ("subject", "supersedes"):
            if fm.get(key) is not None and _links(fm[key]) is None:
                note.add(1, f"bad-{key}", f"{key} is not a list of wiki-links")
        for key in ("owner", "superseded_by"):
            if fm.get(key) is not None and _link(fm[key]) is None:
                note.add(1, f"bad-{key.replace('_', '-')}", f"{key} is not one wiki-link")
        if fm.get("task") is not None and not isinstance(fm["task"], str):
            note.add(1, "bad-task", "task is not a short text")

    def _required(self, note: _Note, fm: dict, strict: bool) -> None:
        """Check 2: class, and validity for a state or a proposal. Outside strict mode, only the label count."""
        cls, validity = _label(fm.get("class")), _label(fm.get("validity"))
        missing = None
        if cls is None:
            missing = ("no-class", "class is missing (state, proposal or record)")
        elif cls in ("state", "proposal") and validity is None:
            missing = ("no-validity", f"validity is missing for a {cls}")
        note.labeled = missing is None
        if strict and missing:
            note.add(2, *missing)

    def _keys(self, note: _Note, fm: dict) -> None:
        """Check 3: keys that are not in the model, the existing keys or the known keys."""
        known = {*MODEL_KEYS, *EXISTING_KEYS, *self.cfg.write.known_keys}
        if note.meta.entity:
            known.update(ENTITY_ONLY_KEYS)
        for key in fm:
            if str(key) not in known:
                note.add(3, "unknown-key", f"unknown frontmatter key {str(key)!r}", WARNING)

    def _links(self, note: _Note, fm: dict, body: str) -> None:
        """Check 4: each wiki-link resolves to a note. Each target counts once for each note."""
        seen = set()
        for target in [*frontmatter_links(fm), *body_links(body)]:
            key = _link_key(target)
            if not target or key in seen:
                continue
            seen.add(key)
            if self.resolver.paths.find(target, note.note_id, None, note.path) is None and not _attachment(target):
                note.add(4, "broken-link", f"link [[{target}]] resolves to no note")

    def _refs(self, note: _Note, fm: dict) -> tuple[int | None, int | None]:
        """Check 5: each reference target exists. Returns the note of superseded_by and the primary subject."""
        found: dict[str, list[int | None]] = {}
        for key in REF_KEYS:
            for target in link_targets(fm.get(key)):
                if key in ENTITY_KEYS:
                    to = self.resolver.entity(target, note.note_id, note.path)
                    what = "entity note"
                else:
                    to = self.resolver.note(target, note.note_id, note.path)
                    what = "note"
                found.setdefault(key, []).append(to)
                if to is None:
                    note.add(5, "missing-target", f"{key} target [[{target}]] resolves to no {what}")
                elif key == "owner" and self.kinds.get(to) not in (None, *OWNER_KINDS):
                    note.add(5, "bad-owner-kind", f"owner [[{target}]] is a {self.kinds[to]}, not a person or a team")
        successor = (found.get("superseded_by") or [None])[0]
        primary = (found.get("subject") or [None])[0]
        return successor, primary

    def _successor(self, note: _Note, successor: int | None) -> None:
        """Check 6: the note that superseded_by names lists this note in its supersedes."""
        if successor is None:
            return
        if note.note_id is not None:
            ok = self.conn.execute(
                "SELECT 1 FROM refs WHERE note_id = ? AND key = 'supersedes' AND to_note = ?",
                (successor, note.note_id),
            ).fetchone()
        else:
            targets = self.conn.execute(
                "SELECT target FROM refs WHERE note_id = ? AND key = 'supersedes'", (successor,)
            ).fetchall()
            ok = note.path is not None and any(_names(t, note.path) for (t,) in targets)
        if not ok:
            path = self.conn.execute("SELECT path FROM notes WHERE id = ?", (successor,)).fetchone()[0]
            note.add(
                6, "no-matching-supersedes", f"superseded_by names {path}, which does not list this note in supersedes"
            )

    def others(self, note: _Note) -> list[tuple[str, str]]:
        """Check 7 against the index: (path, sensitivity) of the other current state notes of the identity key."""
        note_type, primary, task = note.identity
        rows = self.conn.execute(
            "SELECT n.path, n.sensitivity FROM notes n JOIN refs r ON r.note_id = n.id"
            " WHERE r.key = 'subject' AND r.to_note = ? AND n.type = ? AND n.note_class = 'state'"
            " AND (n.validity IS NULL OR n.validity = 'current') AND n.status != 'archived' AND n.id IS NOT ?"
            " AND r.rowid = (SELECT min(rowid) FROM refs WHERE note_id = n.id AND key = 'subject')"
            " ORDER BY n.path",
            (primary, note_type, note.note_id),
        ).fetchall()
        return [
            (path, sensitivity)
            for path, sensitivity in rows
            if path != note.path and _task(read_note(self.cfg.note_path(path))[0]) == task
        ]


def _iso_date(value) -> bool:
    if isinstance(value, datetime):
        return False
    if isinstance(value, date):
        return True
    if not isinstance(value, str) or not ISO_DATE.fullmatch(value.strip()):
        return False
    try:
        date.fromisoformat(value.strip())
    except ValueError:
        return False
    return True


def _double(note: _Note, paths: list[str]) -> None:
    note.add(
        7, "double-identity", f"another current state note has the same type, subject and task: {', '.join(paths)}"
    )


def _single(ctx: _Context, notes: list[_Note], show_private: bool) -> Report:
    for note in notes:
        if note.identity is None or not note.current:
            continue
        others = ctx.others(note)
        if others:
            _double(note, [p if show_private or s == "normal" else "(a private note)" for p, s in others])
    return Report(
        notes=len(notes), issues=sorted((i for n in notes for i in n.issues), key=lambda i: (i.path, i.check))
    )


def check_paths(cfg: Config, conn: sqlite3.Connection, paths: list[str], show_private: bool = True) -> Report:
    """Check the notes at these keys (`<vault>/<path>`), with the required keys."""
    ctx = _Context(cfg, conn)
    notes = []
    for rel in paths:
        file = cfg.note_path(rel)
        if not file.is_file():
            raise FileNotFoundError(f"{rel} is not a note")
        notes.append(ctx.note(file.read_text(errors="replace"), rel, strict=True))
    return _single(ctx, notes, show_private)


def check_draft(
    cfg: Config, conn: sqlite3.Connection, text: str, path: str | None = None, show_private: bool = True
) -> Report:
    """Check a draft (frontmatter and body as one Markdown text) with its intended key, with the required keys.
    A draft at the key of an existing note takes the place of that note."""
    ctx = _Context(cfg, conn)
    return _single(ctx, [ctx.note(text, path, strict=True)], show_private)


def check_index(cfg: Config, conn: sqlite3.Connection) -> Report:
    """Check all notes of the index. A missing class or validity is no error; it counts as unlabeled."""
    ctx = _Context(cfg, conn)
    notes = []
    for (path,) in conn.execute("SELECT path FROM notes ORDER BY path").fetchall():
        file = cfg.note_path(path)
        if file.is_file():
            notes.append(ctx.note(file.read_text(errors="replace"), path, strict=False))
    groups: dict[tuple, list[_Note]] = {}
    for note in notes:
        if note.identity is not None and note.current:
            groups.setdefault(note.identity, []).append(note)
    doubles = [group for group in groups.values() if len(group) > 1]
    for group in doubles:
        for note in group:
            _double(note, [n.label for n in group if n is not note])
    issues = sorted((i for n in notes for i in n.issues), key=lambda i: (i.path, i.check))
    kinds = [i.kind for i in issues]
    passed = sum(not any(i.severity == ERROR for i in n.issues) for n in notes)
    counts = {
        "m1": kinds.count("broken-link") + kinds.count("missing-target"),
        "broken_links": kinds.count("broken-link"),
        "missing_targets": kinds.count("missing-target"),
        "m3": len(doubles),
        "m4": {"share": round(passed / len(notes), 4) if notes else 1.0, "passed": passed, "notes": len(notes)},
        "unlabeled": sum(not n.labeled for n in notes),
        "warnings": sum(i.severity == WARNING for i in issues),
        "check6": kinds.count("no-matching-supersedes"),
    }
    return Report(notes=len(notes), issues=issues, counts=counts)


def format_text(report: Report) -> str:
    """The issues, one for each line, then the totals and the counts of a scan."""
    lines = [f"{i.path}: {i.severity} [{i.check}] {i.message}" for i in report.issues]
    lines.append(f"{report.notes} notes, {report.errors} errors, {report.warnings} warnings")
    c = report.counts
    if c is not None:
        m4 = c["m4"]
        lines += [
            f"M1 broken links and missing targets: {c['m1']}"
            f" ({c['broken_links']} links, {c['missing_targets']} targets)",
            f"M3 identity keys with more than one current state note: {c['m3']}",
            f"M4 notes with no error: {m4['passed']} of {m4['notes']} ({m4['share']:.1%})",
            f"unlabeled notes: {c['unlabeled']}",
            f"warnings: {c['warnings']}",
            f"superseded_by with no matching supersedes: {c['check6']}",
        ]
    return "\n".join(lines)
