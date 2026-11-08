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
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatchcase

from .config import Config
from .validate import ERROR, Issue, _Context, _label, _Note, mark_doubles

TOP = "(top)"
TOTAL = "(total)"
KEYS = ("class", "validity", "as_of", "subject")
FRONTMATTER_DEFECTS = ("bad-yaml", "bad-validity")


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
