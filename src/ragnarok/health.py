"""The health of the knowledge base: structural errors and label coverage, for each area and for the totals.

An area is the first folder under the vault; the notes at the top of the vault are the area `(top)`. The index
column `notes.area` is not used, because a frontmatter `area:` key can change it.

Coverage counts each key over the notes that require it, from the frontmatter as written (not `notes.note_class`,
which falls back to a default class from `type`):
- `class`: all notes;
- `validity`: the notes whose explicit class is state or proposal;
- `as_of`: the notes whose explicit class is record;
- `subject`: all notes except entity notes.
A note with no explicit class, or a class outside the three, fails `class` and needs no `validity` or `as_of`.

The `[health] accept` rules mark issues as accepted: they show with their reason and do not count. A note with an
accepted frontmatter defect (`bad-yaml`, `bad-validity`) is out of the coverage count.

The worklist gives each error that is not accepted one fix. `link_decision` is the one function that decides
between `ragnarok fix links` and the user; the fixer of the links calls the same function.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase

import frontmatter
import yaml

from .config import Config
from .index import _link_key
from .validate import ERROR, Issue, _Context, _label, _Note, mark_doubles

TOP = "(top)"
TOTAL = "(total)"
KEYS = ("class", "validity", "as_of", "subject")
FRONTMATTER_DEFECTS = ("bad-yaml", "bad-validity")
GENERIC = {"readme", "summary", "index", "overview", "dashboard", "brief", "decisionlog", "drivers", "todo"}
SAME_AREA = {"dashboard"}  # a generic name that a link fix keeps when the target is in the area of the source
FIX_LINKS, FIX_YAML, USER = "ragnarok fix links", "ragnarok fix yaml", "the user"
KB_WRITE_KINDS = ("double-identity", "bad-superseded-by", "bad-supersedes", "no-matching-supersedes")
KB_CONVERT_KINDS = ("no-class", "no-validity", "no-as-of", "no-subject")
BROKEN_LINK = re.compile(r"link \[\[(.*)\]\] resolves to no note")


def area_of(path: str) -> tuple[str, str]:
    """(vault, area) of a note key `<vault>/<path in the vault>`."""
    parts = path.split("/")
    return parts[0], parts[1] if len(parts) > 2 else TOP


def _present(value) -> bool:
    if isinstance(value, (list, dict)):
        return bool(value)
    return _label(value) is not None


def required(fm: dict, entity: bool) -> dict[str, bool]:
    """For each key, True when the note requires it."""
    cls = _label(fm.get("class"))
    return {
        "class": True,
        "validity": cls in ("state", "proposal"),
        "as_of": cls == "record",
        "subject": not entity,
    }


@dataclass
class Values:
    notes: int = 0
    errors: int = 0
    accepted: int = 0
    have: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KEYS, 0))
    need: dict[str, int] = field(default_factory=lambda: dict.fromkeys(KEYS, 0))

    def share(self, key: str) -> float | None:
        return round(self.have[key] / self.need[key], 4) if self.need[key] else None

    def as_dict(self) -> dict:
        return {
            "notes": self.notes,
            "structural_errors": {
                "errors": self.errors,
                "accepted": self.accepted,
                "not_accepted": self.errors - self.accepted,
            },
            "coverage": {k: {"have": self.have[k], "need": self.need[k], "share": self.share(k)} for k in KEYS},
        }


@dataclass
class Result:
    areas: dict[tuple[str, str], Values]
    total: Values
    issues: list[tuple[Issue, str | None]]  # the error issues, each with the reason of its accept rule or None
    gate: dict
    vault: str | None = None  # the scope: one vault, or all vaults
    area: str | None = None

    def label(self, vault: str, area: str) -> str:
        """The name of an area: with one vault in scope the area alone, else `<vault>/<area>`."""
        return area if self.vault else f"{vault}/{area}"

    def rows(self) -> list[dict]:
        """One dict for each area, then one for the totals."""
        rows = [{"vault": v, "area": self.label(v, a), **values.as_dict()} for (v, a), values in self.areas.items()]
        return [*rows, {"vault": self.vault, "area": TOTAL, **self.total.as_dict(), "gate": self.gate}]

    def as_dict(self) -> dict:
        return {
            "vault": self.vault,
            "area": self.area,
            "areas": self.rows()[:-1],
            "total": self.total.as_dict(),
            "gate": self.gate,
            "issues": [
                {**issue.as_dict(), "accepted": reason is not None, "reason": reason} for issue, reason in self.issues
            ],
        }


def _reason(cfg: Config, issue: Issue) -> str | None:
    for rule in cfg.health.accept:
        if issue.kind in rule.kinds and fnmatchcase(issue.path, rule.glob):
            return rule.reason
    return None


def _count(values: Values, note: _Note, reasons: list[str | None]) -> None:
    errors = [r for i, r in zip(note.issues, reasons, strict=True) if i.severity == ERROR]
    values.notes += 1
    values.errors += len(errors)
    values.accepted += sum(r is not None for r in errors)
    defect = any(r is not None and i.kind in FRONTMATTER_DEFECTS for i, r in zip(note.issues, reasons, strict=True))
    if defect:
        return
    for key, needed in required(note.fm, note.meta.entity).items():
        if needed:
            values.need[key] += 1
            values.have[key] += _present(note.fm.get(key))


def gate(cfg: Config, total: Values) -> dict:
    """Pass or fail of each attribute on the totals; `no target` when the config has none."""
    targets = cfg.health.targets
    out = {}

    def judge(name: str, value, target, passed) -> None:
        if target is None:
            out[name] = {"value": value, "target": None, "result": "no target"}
        else:
            out[name] = {"value": value, "target": target, "result": "pass" if passed(value, target) else "fail"}

    judge("structural_errors", total.errors - total.accepted, targets.get("structural_errors"), lambda v, t: v <= t)
    for key in KEYS:
        judge(f"coverage.{key}", total.share(key), targets.get("coverage"), lambda v, t: v is None or v >= t)
    judged = [g["result"] for g in out.values() if g["result"] != "no target"]
    out["result"] = "no target" if not judged else "fail" if "fail" in judged else "pass"
    return out


def health(cfg: Config, conn: sqlite3.Connection, vault: str | None = None, area: str | None = None) -> Result:
    """The values of the notes of the index in the scope. Check 7 runs over all notes, as in `validate`."""
    notes = list(_Context(cfg, conn).index_notes())
    mark_doubles(notes)
    areas: dict[tuple[str, str], Values] = {}
    total = Values()
    issues = []
    for note in notes:
        v, a = area_of(note.path)
        if (vault is not None and v != vault) or (area is not None and a != area):
            continue
        reasons = [_reason(cfg, i) for i in note.issues]
        _count(areas.setdefault((v, a), Values()), note, reasons)
        _count(total, note, reasons)
        issues += [(i, r) for i, r in zip(note.issues, reasons, strict=True) if i.severity == ERROR]
    issues.sort(key=lambda ir: (ir[0].path, ir[0].check))
    areas = dict(sorted(areas.items()))
    return Result(areas, total, issues, gate(cfg, total), vault, area)


# --- the worklist ------------------------------------------------------------------------------------------------


@dataclass
class LinkDecision:
    """auto: the fix applies, to the note `target`; else the user decides, and `reason` says why."""

    auto: bool
    target: str | None = None
    reason: str | None = None


def note_names(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """The note keys of the index, for each basename (lower case, no `.md`). `_archive` notes are in it."""
    names: dict[str, list[str]] = {}
    for (path,) in conn.execute("SELECT path FROM notes ORDER BY path"):
        names.setdefault(_link_key(path).rsplit("/", 1)[-1], []).append(path)
    return names


def link_decision(cfg: Config, names: dict[str, list[str]], source: str, target: str) -> LinkDecision:
    """Decide the fix of one broken link: `source` is the key of the note with the link, `target` the link text.

    A link fix applies when the basename of the target is the name of exactly one note of the vault of the source,
    the name is not generic (`Dashboard` passes in the same area), and the source is not under an accepted glob
    or in an `assets` folder. The edit itself, which keeps `!`, `|alias`, `#heading` and `^block`, is the fixer's.
    """
    vault, area = area_of(source)
    if "assets" in source.split("/")[1:-1] or any(
        "broken-link" in r.kinds and fnmatchcase(source, r.glob) for r in cfg.health.accept
    ):
        return LinkDecision(False, reason="accepted-or-assets")
    name = _link_key(target.strip()).rstrip("/").rsplit("/", 1)[-1]
    found = names.get(name, [])
    same = [p for p in found if p.split("/")[0] == vault]
    if not same:
        return LinkDecision(False, reason="cross-vault-only" if found else "no-candidate")
    if len(same) > 1:
        return LinkDecision(False, reason="several-candidates")
    if name in GENERIC and not (name in SAME_AREA and area_of(same[0]) == (vault, area)):
        return LinkDecision(False, reason="generic-name")
    return LinkDecision(True, target=same[0])


def yaml_line(text: str) -> int | None:
    """The index of the line that `ragnarok fix yaml` fixes, or None when the user must fix the frontmatter.

    The failing line must be a top-level `key: value` with a plain scalar that holds `: ` and does not start with
    a quote or a YAML indicator, and quoting the value must make the frontmatter parse.
    """
    try:
        block = frontmatter.YAMLHandler().split(text)[0]
        yaml.safe_load(block)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        if mark is None:
            return None
        lines = block.split("\n")
        n = mark.line
    except Exception:
        return None
    else:
        return None
    line = lines[n]
    key, _, value = line.partition(": ")
    if not key or key[0] in " \t#-?:\"'[{|>&*!%@`" or ": " not in value or value[0] in "\"'[{|>&*! \t":
        return None
    if n + 1 < len(lines) and lines[n + 1][:1] in (" ", "\t"):
        return None  # the scalar goes on in the next line
    quoted = value.replace("\\", "\\\\").replace('"', '\\"')
    lines[n] = f'{key}: "{quoted}"'
    try:
        return n if isinstance(yaml.safe_load("\n".join(lines)), dict) else None
    except yaml.YAMLError:
        return None


@dataclass
class Work:
    path: str
    vault: str
    area: str
    kind: str
    detail: str
    fix: str
    target: str | None = None  # a link fix: the note that the link names after the fix
    reason: str | None = None  # a link for the user: why

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("path", "vault", "area", "kind", "detail", "fix", "target", "reason")}

    def text(self) -> str:
        return f"{self.path}: {self.kind} {self.detail}  fix: {self.fix}"


def worklist(cfg: Config, conn: sqlite3.Connection, result: Result) -> list[Work]:
    """One line for each error that is not accepted, with its fix, sorted by area, then fix."""
    names = note_names(conn)
    out = []
    for issue, reason in result.issues:
        if reason is not None:
            continue
        vault, area = area_of(issue.path)
        work = Work(issue.path, vault, area, issue.kind, issue.message, USER)
        match = BROKEN_LINK.fullmatch(issue.message) if issue.kind == "broken-link" else None
        if match:
            link = match.group(1)
            decision = link_decision(cfg, names, issue.path, link)
            work.target, work.reason = decision.target, decision.reason
            work.detail = f"[[{link}]] -> {decision.target}" if decision.auto else f"[[{link}]] ({decision.reason})"
            work.fix = FIX_LINKS if decision.auto else USER
        elif issue.kind == "bad-yaml":
            file = cfg.note_path(issue.path)
            if yaml_line(file.read_text(errors="replace")) is not None:
                work.fix = FIX_YAML
        elif issue.kind in KB_WRITE_KINDS:
            work.fix = "kb-write"
        elif issue.kind in KB_CONVERT_KINDS:
            work.fix = f"kb-convert {area}"
        out.append(work)
    return sorted(out, key=lambda w: (w.vault, w.area, w.fix, w.path, w.kind))


def format_worklist(works: list[Work]) -> str:
    return "worklist:\n" + "\n".join(w.text() for w in works) if works else "worklist: empty"


def _pct(values: Values, key: str) -> str:
    share = values.share(key)
    return "-" if share is None else f"{values.have[key]}/{values.need[key]} {share:.1%}"


def format_text(result: Result) -> str:
    """The error issues (accepted ones marked), a table of the areas and the totals, then the gate."""
    lines = [
        f"{i.path}: {i.severity} [{i.check}] {i.message}" + (f" (accepted: {r})" if r is not None else "")
        for i, r in result.issues
    ]
    rows = [(result.label(v, a), values) for (v, a), values in result.areas.items()] + [(TOTAL, result.total)]
    width = max(len(name) for name, _ in rows)
    header = ["notes", "errors", "accepted", *KEYS]
    lines.append(f"{'area':{width}}  " + "  ".join(f"{h:>8}" if i < 3 else f"{h:>16}" for i, h in enumerate(header)))
    for name, values in rows:
        cells = [f"{values.notes:>8}", f"{values.errors:>8}", f"{values.accepted:>8}"]
        cells += [f"{_pct(values, key):>16}" for key in KEYS]
        lines.append(f"{name:{width}}  " + "  ".join(cells))
    lines.append("gate on the totals:")
    for name, g in result.gate.items():
        if name == "result":
            continue
        value, target = g["value"], g["target"]
        if name.startswith("coverage."):
            value = "-" if value is None else f"{value:.1%}"
            target = "-" if target is None else f"{target:.1%}"
        shown = "no target" if g["target"] is None else f"target {target}  {g['result']}"
        lines.append(f"  {name}: {value}  {shown}")
    lines.append(f"result: {result.gate['result']}")
    return "\n".join(lines)


def save(cfg: Config, result: Result) -> None:
    """Add one line for each area and one for the totals to data/health/history.jsonl."""
    time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    folder = cfg.data_dir / "health"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "history.jsonl").open("a") as f:
        for row in result.rows():
            f.write(json.dumps({"time": time, **row}, ensure_ascii=False) + "\n")
