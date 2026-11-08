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

Drift is the share of the notes that are changed since the conversion and pass the model. The baseline
`data/health/baseline.json` holds, for each note path, the SHA-256 of the text, the date and the source of the entry.
A note is changed when its hash differs from the baseline or its path is not in the baseline. Only a vault with an
entry in the baseline counts. A note passes when it lacks no required key and has no error that is not accepted.
A changed note with an accepted frontmatter defect is out of the drift count, as in coverage.
A W10 text with `{{apply-date}}` has no single hash: when the live note matches the text with each placeholder read
as any ISO date, the entry holds the hash of the live note; else it holds the hash of the template, which no live
note equals, so the note is changed.

The seed takes one more rule, with a cutoff time: a note that differs from its W10 entry, or has none, and whose
mtime is at or before the cutoff gets the hash of the live note (source `pre-cutoff-live`). mtime is used at the
seed, once, and never at the drift check.

The worklist gives each error that is not accepted one fix. `link_decision` is the one function that decides
between `ragnarok fix links` and the user; the fixer of the links calls the same function.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path

import frontmatter
import yaml

from .config import Config
from .index import _link_key
from .metadata import is_private, iter_notes, parse_note
from .replay import wilson
from .validate import ERROR, Issue, _Context, _label, _Note, mark_doubles

TOP = "(top)"
TOTAL = "(total)"
KEYS = ("class", "validity", "as_of", "subject")
FRONTMATTER_DEFECTS = ("bad-yaml", "bad-validity")
GENERIC = {"readme", "summary", "index", "overview", "dashboard", "brief", "decisionlog", "drivers", "todo"}
SAME_AREA = {"dashboard"}  # a generic name that a link fix keeps when the target is in the area of the source
FIX_LINKS, FIX_YAML, USER = "ragnarok fix links", "ragnarok fix yaml", "the user"
KB_WRITE_KINDS = ("double-identity", "bad-superseded-by", "bad-supersedes", "no-matching-supersedes")
BROKEN_LINK = re.compile(r"link \[\[(.*)\]\] resolves to no note", re.DOTALL)
APPLY_DATE = b"{{apply-date}}"
ISO_DATE = rb"\d{4}-\d{2}-\d{2}"


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
    baseline: bool = False  # a baseline file exists, so the drift fields hold values
    changed: int = 0  # the notes that differ from the baseline or are not in it
    passing: int = 0  # the changed notes that pass the model

    def share(self, key: str) -> float | None:
        return round(self.have[key] / self.need[key], 4) if self.need[key] else None

    def drift_share(self) -> float | None:
        return round(self.passing / self.changed, 4) if self.changed else None

    def as_dict(self) -> dict:
        return {
            "notes": self.notes,
            "structural_errors": {
                "errors": self.errors,
                "accepted": self.accepted,
                "not_accepted": self.errors - self.accepted,
            },
            "coverage": {k: {"have": self.have[k], "need": self.need[k], "share": self.share(k)} for k in KEYS},
            "drift": {"changed": self.changed, "passing": self.passing, "share": self.drift_share()}
            if self.baseline
            else None,
        }


@dataclass
class Result:
    areas: dict[tuple[str, str], Values]
    total: Values
    issues: list[tuple[Issue, str | None]]  # the error issues, each with the reason of its accept rule or None
    gate: dict
    vault: str | None = None  # the scope: one vault, or all vaults
    area: str | None = None
    gaps: list[tuple[str, str]] = field(default_factory=list)  # (path, key) of each missing required key

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


def _defect(note: _Note, reasons: list[str | None]) -> bool:
    """True when the note has an accepted frontmatter defect: it is out of the coverage and drift counts."""
    return any(r is not None and i.kind in FRONTMATTER_DEFECTS for i, r in zip(note.issues, reasons, strict=True))


def _count(values: Values, note: _Note, reasons: list[str | None]) -> list[str]:
    """Add the note to the values. Returns the required keys that the note lacks."""
    errors = [r for i, r in zip(note.issues, reasons, strict=True) if i.severity == ERROR]
    values.notes += 1
    values.errors += len(errors)
    values.accepted += sum(r is not None for r in errors)
    missing = []
    if _defect(note, reasons):
        return missing
    for key, needed in required(note.fm, note.meta.entity).items():
        if needed:
            values.need[key] += 1
            values.have[key] += _present(note.fm.get(key))
            if not _present(note.fm.get(key)):
                missing.append(key)
    return missing


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
    if total.baseline:
        judge("drift", total.drift_share(), targets.get("drift"), lambda v, t: v is None or v >= t)
    else:
        out["drift"] = {"value": None, "target": targets.get("drift"), "result": "no baseline"}
    record = last_audit(cfg)
    if record is None:
        out["correctness"] = {"value": None, "target": targets.get("correctness"), "result": "no audit"}
    else:
        judge("correctness", record["share"], targets.get("correctness"), lambda v, t: v >= t)
        out["correctness"].update(low=record["low"], high=record["high"], date=record["date"])
    judged = [g["result"] for g in out.values() if g["result"] in ("pass", "fail")]
    out["result"] = "no target" if not judged else "fail" if "fail" in judged else "pass"
    return out


def health(cfg: Config, conn: sqlite3.Connection, vault: str | None = None, area: str | None = None) -> Result:
    """The values of the notes of the index in the scope. Check 7 runs over all notes, as in `validate`."""
    notes = list(_Context(cfg, conn).index_notes())
    mark_doubles(notes)
    base = load_baseline(cfg)
    base_vaults = {path.split("/")[0] for path in base or {}}
    areas: dict[tuple[str, str], Values] = {}
    total = Values(baseline=base is not None)
    issues, gaps = [], []
    for note in notes:
        v, a = area_of(note.path)
        if (vault is not None and v != vault) or (area is not None and a != area):
            continue
        reasons = [_reason(cfg, i) for i in note.issues]
        values = areas.setdefault((v, a), Values(baseline=base is not None))
        missing = _count(values, note, reasons)
        _count(total, note, reasons)
        gaps += [(note.path, key) for key in missing]
        if base is not None and v in base_vaults and not _defect(note, reasons) and changed(cfg, base, note.path):
            open_errors = [i for i, r in zip(note.issues, reasons, strict=True) if i.severity == ERROR and r is None]
            ok = not missing and not open_errors
            for each in (values, total):
                each.changed += 1
                each.passing += ok
        issues += [(i, r) for i, r in zip(note.issues, reasons, strict=True) if i.severity == ERROR]
    issues.sort(key=lambda ir: (ir[0].path, ir[0].check))
    areas = dict(sorted(areas.items()))
    return Result(areas, total, issues, gate(cfg, total), vault, area, gaps)


# --- the drift baseline -----------------------------------------------------------------------------------------


def baseline_path(cfg: Config) -> Path:
    return cfg.data_dir / "health" / "baseline.json"


def file_hash(path: Path) -> str | None:
    """The SHA-256 of the bytes of the file, or None when it is not a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_baseline(cfg: Config) -> dict[str, dict] | None:
    """note key -> {sha256, date, source}, or None when there is no baseline file."""
    path = baseline_path(cfg)
    return json.loads(path.read_text()) if path.exists() else None


def write_baseline(cfg: Config, data: dict[str, dict]) -> None:
    path = baseline_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n")
    os.replace(temp, path)


def changed(cfg: Config, baseline: dict[str, dict], path: str) -> bool:
    """True when the note differs from the baseline or its path is not in the baseline."""
    entry = baseline.get(path)
    return entry is None or entry["sha256"] != file_hash(cfg.note_path(path))


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def update_baseline(cfg: Config, paths: list[str]) -> int:
    """Write the current hash of each note key in `paths` that has a file. Returns the number written.
    With no baseline file nothing is written: the baseline starts only with the seed."""
    data = load_baseline(cfg)
    if data is None:
        return 0
    written = 0
    for path in paths:
        digest = file_hash(cfg.note_path(path))
        if digest is not None:
            data[path] = {"sha256": digest, "date": _today(), "source": "update"}
            written += 1
    write_baseline(cfg, data)
    return written


@dataclass
class Seed:
    entries: dict[str, dict]
    areas: int
    proposed: int
    proposed_changed: list[str]  # the proposed texts that the live note does not equal
    before: int  # the notes that only `real-before.txt` holds
    before_changed: list[str]
    pre_cutoff: int = 0  # the notes with the live hash, as the cutoff rule says
    differ: list[str] = field(default_factory=list)  # the entries that differ from the live note, at the end


def seed_baseline(cfg: Config, w10: Path, cutoff: datetime) -> Seed:
    """The first baseline from the W10 records: `<area>/proposed/<note key>.proposed` for each converted note, then
    the `real-before.txt` hashes (`<sha256>  ./<path in the area>`) of the notes that the conversion did not change.
    The vault of an area is the first folder of its proposed keys. `<area>/skipped/` is not read.
    Then each note of these vaults that differs from its entry, or has none, and has an mtime at or before
    `cutoff` (a naive time is local time) gets the live hash."""
    areas = sorted(a for a in w10.iterdir() if (a / "proposed").is_dir())
    if not areas:
        raise ValueError(f"no area with a proposed folder in {w10}")
    date = _today()
    entries: dict[str, dict] = {}
    vaults: dict[Path, str] = {}
    proposed_changed = []
    for area in areas:
        for file in sorted((area / "proposed").rglob("*.proposed")):
            key = file.relative_to(area / "proposed").as_posix().removesuffix(".proposed")
            vaults[area] = key.split("/")[0]
            text = file.read_bytes()
            live = cfg.note_path(key)
            live_bytes = live.read_bytes() if live.is_file() else None
            if APPLY_DATE not in text:
                digest, source, equal = hashlib.sha256(text).hexdigest(), "proposed", live_bytes == text
            else:
                pattern = re.compile(ISO_DATE.join(re.escape(p) for p in text.split(APPLY_DATE)), re.DOTALL)
                equal = live_bytes is not None and pattern.fullmatch(live_bytes) is not None
                if equal:
                    digest, source = hashlib.sha256(live_bytes).hexdigest(), "proposed-live"
                else:
                    digest, source = hashlib.sha256(text).hexdigest(), "proposed-unmatched"
            entries[key] = {"sha256": digest, "date": date, "source": source}
            if not equal:
                proposed_changed.append(key)
    proposed = len(entries)
    before_changed = []
    for area, vault in vaults.items():
        lines = (area / "real-before.txt").read_text().splitlines() if (area / "real-before.txt").is_file() else []
        for line in lines:
            digest, _, rel = line.partition("  ")
            key = f"{vault}/{area.name}/{rel.removeprefix('./')}"
            if rel and key.endswith(".md") and key not in entries:
                entries[key] = {"sha256": digest, "date": date, "source": "real-before"}
                if file_hash(cfg.note_path(key)) != digest:
                    before_changed.append(key)
    before = len(entries) - proposed
    limit, pre_cutoff = cutoff.timestamp(), 0
    for key in iter_notes(cfg):
        if key.split("/")[0] not in vaults.values():
            continue
        live = cfg.note_path(key)
        digest = file_hash(live)
        if (key not in entries or entries[key]["sha256"] != digest) and live.stat().st_mtime <= limit:
            entries[key] = {"sha256": digest, "date": date, "source": "pre-cutoff-live"}
            pre_cutoff += 1
    differ = [key for key, entry in entries.items() if entry["sha256"] != file_hash(cfg.note_path(key))]
    return Seed(entries, len(areas), proposed, proposed_changed, before, before_changed, pre_cutoff, differ)


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


def excluded(cfg: Config, source: str, kind: str) -> bool:
    """True when no automatic fix of this issue kind applies to the note: it is in an `assets` folder or under an
    accept rule of the kind."""
    return "assets" in source.split("/")[1:-1] or any(
        kind in r.kinds and fnmatchcase(source, r.glob) for r in cfg.health.accept
    )


def link_decision(cfg: Config, names: dict[str, list[str]], source: str, target: str) -> LinkDecision:
    """Decide the fix of one broken link: `source` is the key of the note with the link, `target` the link text.

    A link fix applies when the basename of the target is the name of exactly one note of the vault of the source,
    the name is not generic (`Dashboard` passes in the same area), and the source is not under an accepted glob
    or in an `assets` folder. The edit itself, which keeps `!`, `|alias`, `#heading` and `^block`, is the fixer's.
    """
    vault, area = area_of(source)
    if excluded(cfg, source, "broken-link"):
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


def yaml_decision(cfg: Config, source: str, text: str) -> int | None:
    """The line to fix in the frontmatter of the note `source`, or None when the user must decide."""
    return None if excluded(cfg, source, "bad-yaml") else yaml_line(text)


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
    reason: str | None = None  # a link or YAML defect for the user: why
    label: bool = False  # a missing required key, not a validate error

    def as_dict(self) -> dict:
        keys = ("path", "vault", "area", "kind", "detail", "fix", "target", "reason", "label")
        return {k: getattr(self, k) for k in keys}

    def text(self) -> str:
        return f"{self.path}: {self.kind} {self.detail}  fix: {self.fix}".replace("\n", " ")


def worklist(cfg: Config, conn: sqlite3.Connection, result: Result) -> list[Work]:
    """One line for each error that is not accepted and one for each missing required key, with the fix for it,
    sorted by area, then fix."""
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
            if yaml_decision(cfg, issue.path, cfg.note_path(issue.path).read_text(errors="replace")) is not None:
                work.fix = FIX_YAML
            elif excluded(cfg, issue.path, "bad-yaml"):
                work.reason = "accepted-or-assets"
        elif issue.kind in KB_WRITE_KINDS:
            work.fix = "kb-write"
        out.append(work)
    for path, key in result.gaps:
        vault, area = area_of(path)
        out.append(
            Work(
                path, vault, area, f"no-{key.replace('_', '-')}", f"{key} is missing", f"kb-convert {area}", label=True
            )
        )
    return sorted(out, key=lambda w: (w.vault, w.area, w.fix, w.path, w.kind))


def format_worklist(works: list[Work]) -> str:
    return "worklist:\n" + "\n".join(w.text() for w in works) if works else "worklist: empty"


def _pct(values: Values, key: str) -> str:
    share = values.share(key)
    return "-" if share is None else f"{values.have[key]}/{values.need[key]} {share:.1%}"


def _drift(values: Values) -> str:
    if not values.baseline:
        return "no baseline"
    share = values.drift_share()
    return "-" if share is None else f"{values.passing}/{values.changed} {share:.1%}"


def format_text(result: Result) -> str:
    """A table of the areas and the totals, the accepted issues with their reason, then the gate. The issues that
    are not accepted are in the worklist and the JSON output."""
    lines = []
    rows = [(result.label(v, a), values) for (v, a), values in result.areas.items()] + [(TOTAL, result.total)]
    width = max(len(name) for name, _ in rows)
    header = ["notes", "errors", "accepted", *KEYS, "drift"]
    lines.append(f"{'area':{width}}  " + "  ".join(f"{h:>8}" if i < 3 else f"{h:>16}" for i, h in enumerate(header)))
    for name, values in rows:
        cells = [f"{values.notes:>8}", f"{values.errors:>8}", f"{values.accepted:>8}"]
        cells += [f"{_pct(values, key):>16}" for key in KEYS] + [f"{_drift(values):>16}"]
        lines.append(f"{name:{width}}  " + "  ".join(cells))
    lines += [
        f"{i.path}: {i.severity} [{i.check}] {i.message} (accepted: {r})" for i, r in result.issues if r is not None
    ]
    lines.append("gate on the totals:")
    for name, g in result.gate.items():
        if name == "result":
            continue
        value, target = g["value"], g["target"]
        if g["result"] == "no baseline":
            lines.append(f"  {name}: no baseline")
            continue
        if name == "drift":
            value = _drift(result.total)
            target = "-" if target is None else f"{target:.1%}"
        if name == "correctness":
            if g["result"] == "no audit":
                lines.append("  correctness: no audit")
                continue
            value = f"{value:.1%} ({g['low']:.1%}-{g['high']:.1%}, {g['date']})"
            target = "-" if target is None else f"{target:.1%}"
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


# --- the label audit --------------------------------------------------------------------------------------------

AUDIT_HEADER = ("path", "class", "validity", "as_of", "subject", "subject_reason", "type", "verdict")
VERDICT = re.compile(r"ok|wrong:\s*\S.*", re.IGNORECASE)
CELL_SPLIT = re.compile(r"(?<!\\)\|")


def audit_records_path(cfg: Config) -> Path:
    return cfg.data_dir / "health" / "audit-records.jsonl"


def last_audit(cfg: Config) -> dict | None:
    """The last line of the audit records, or None when there is no record."""
    path = audit_records_path(cfg)
    lines = path.read_text().splitlines() if path.is_file() else []
    return json.loads(lines[-1]) if lines else None


def labeled_notes(cfg: Config) -> dict[str, dict]:
    """Each note with an explicit `class` that is not private, as {key: frontmatter}. Read from the files."""
    notes = {}
    for key in iter_notes(cfg):
        if is_private(cfg, key):
            continue
        fm, _ = parse_note(cfg.note_path(key).read_text(errors="replace"))
        if _present(fm.get("class")):
            notes[key] = fm
    return notes


def allocate(sizes: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """The share of the sample for each area. The rule:
    - n at least the number of notes: each note;
    - else the largest remainder of n * size / total (a tie goes to the first area in key order);
    - with n at least the number of areas, an area with a share of 0 takes 1 from the area with the largest share."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    quota = {k: n * s // total for k, s in sizes.items()}
    by_remainder = sorted(sizes, key=lambda k: (-(n * sizes[k] % total), k))
    for k in by_remainder[: n - sum(quota.values())]:
        quota[k] += 1
    if n >= len(sizes):
        for k in sorted(k for k, q in quota.items() if q == 0):
            donor = max(sorted(quota), key=lambda d: quota[d])
            quota[donor] -= 1
            quota[k] = 1
    return quota


def sample_notes(cfg: Config, n: int, seed: int) -> list[tuple[str, dict]]:
    """The sample of the labeled notes, stratified by area (see `allocate`), sorted by path."""
    notes = labeled_notes(cfg)
    groups: dict[tuple[str, str], list[str]] = {}
    for key in sorted(notes):
        groups.setdefault(area_of(key), []).append(key)
    quota = allocate({k: len(v) for k, v in groups.items()}, n)
    rng = random.Random(seed)
    chosen = [key for area in sorted(groups) for key in rng.sample(groups[area], quota[area])]
    return [(key, notes[key]) for key in sorted(chosen)]


def _cell(value) -> str:
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return "" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def write_sheet(cfg: Config, sample: list[tuple[str, dict]], n: int, seed: int) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = cfg.data_dir / "health"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"audit-{stamp}.md"
    lines = [
        f"# Label audit {stamp}",
        "",
        f"Sample {len(sample)} (requested {n}), seed {seed}. Write `ok` or `wrong: <correct value>` in the last cell"
        " of each row. Then run `ragnarok health audit score <this file>`.",
        "",
        "| " + " | ".join(AUDIT_HEADER) + " |",
        "|" + "|".join("---" for _ in AUDIT_HEADER) + "|",
    ]
    for key, fm in sample:
        cells = [key, *(fm.get(k) for k in AUDIT_HEADER[1:-1])]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |  |")
    path.write_text("\n".join(lines) + "\n")
    return path


def read_sheet(path: Path) -> list[tuple[str, str]]:
    """(note path, verdict) for each row of the table of a sheet."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("\\|", "|") for c in CELL_SPLIT.split(line.strip())[1:-1]]
        if len(cells) != len(AUDIT_HEADER) or cells[0] in ("path", "---"):
            continue
        rows.append((cells[0], cells[-1]))
    return rows


def score_sheet(cfg: Config, path: Path) -> dict:
    """The record of a marked sheet, saved to the audit records. Raises ValueError, with the rows that have no
    verdict `ok` or `wrong: <value>` listed, when the sheet is not fully marked."""
    rows = read_sheet(path)
    if not rows:
        raise ValueError(f"{path} has no rows")
    unmarked = [key for key, verdict in rows if not VERDICT.fullmatch(verdict)]
    if unmarked:
        raise ValueError(
            f"{len(unmarked)} rows have no verdict (ok or wrong: <correct value>):\n"
            + "\n".join(f"  {key}" for key in unmarked)
        )
    correct = sum(verdict.lower() == "ok" for _, verdict in rows)
    low, high = wilson(correct, len(rows))
    record = {
        "date": _today(),
        "sheet": str(path),
        "n": len(rows),
        "correct": correct,
        "share": round(correct / len(rows), 4),
        "low": round(low, 4),
        "high": round(high, 4),
    }
    out = audit_records_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
