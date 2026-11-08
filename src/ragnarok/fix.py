"""The fixer of broken links and bad YAML: `ragnarok fix links|yaml`.

The fixer takes the errors of `health` that no accept rule covers. A link fix applies only when
`health.link_decision` says auto, on the note names of the index after a fresh scan; a YAML fix only when
`health.yaml_decision` gives a line. So the fixer and the worklist cannot disagree.

A link edit changes the target text to the basename of the target note, in each occurrence of the broken target
(same link key) outside code: in the body outside fenced blocks and inline code, and in the frontmatter values of
the keys that are not reference keys. It keeps `!`, `|alias`, `#heading`, `^block` and the `\\` of a table pipe.
A YAML edit puts the failing value in double quotes.

For each note: skip it when its mtime is less than 10 minutes old; copy it to `data/fix-backups/<stamp>/<note key>`;
read it again and check that the old text is still there; write a temporary file and rename it; add the lines
`path, old, new` to `log.tsv` of the run.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import yaml

from . import health as h
from .config import Config
from .index import REF_KEYS, _link_key
from .validate import FENCE, INLINE_CODE, _clean, check_paths

LINK = re.compile(r"!?\[\[([^\]|#^]+)")  # the WIKI_LINK of the index, with the span of the target
BOUNDARY = frontmatter.YAMLHandler.FM_BOUNDARY
TOP_KEY = re.compile(r"([^\s#\-][^:]*):")
MIN_AGE = 600  # seconds: a note changed more recently is skipped


@dataclass
class Change:
    """One edit: a broken link (old and new target text) or a YAML line (old and new line)."""

    path: str
    old: str
    new: str
    line: int | None = None  # a YAML fix: the line index in the note


@dataclass
class Run:
    changes: list[Change]
    written: list[str] = field(default_factory=list)  # the note keys written
    recent: list[str] = field(default_factory=list)  # skipped: changed in the last 10 minutes
    gone: list[str] = field(default_factory=list)  # skipped: the old text is not there any more
    folder: Path | None = None  # the backup folder of the run


def _scope(issues, vault, area, exclude_area):
    for issue, reason in issues:
        v, a = h.area_of(issue.path)
        if reason is None and (vault is None or v == vault) and (area is None or a == area) and a != exclude_area:
            yield issue


def plan_links(cfg: Config, conn: sqlite3.Connection, vault=None, area=None, exclude_area=None) -> list[Change]:
    """One change for each broken link (note, link key) that `link_decision` fixes."""
    names = h.note_names(conn)
    changes = []
    for issue in _scope(h.health(cfg, conn, vault, area).issues, vault, area, exclude_area):
        match = h.BROKEN_LINK.fullmatch(issue.message) if issue.kind == "broken-link" else None
        if match:
            decision = h.link_decision(cfg, names, issue.path, match.group(1))
            if decision.auto:
                name = decision.target.rsplit("/", 1)[-1].removesuffix(".md")
                changes.append(Change(issue.path, match.group(1), name))
    return changes


def plan_yaml(cfg: Config, conn: sqlite3.Connection, vault=None, area=None, exclude_area=None) -> list[Change]:
    """One change for each note with bad YAML that `yaml_decision` fixes."""
    changes = []
    for issue in _scope(h.health(cfg, conn, vault, area).issues, vault, area, exclude_area):
        if issue.kind != "bad-yaml":
            continue
        text = _read(cfg, issue.path)
        n = h.yaml_decision(cfg, issue.path, text) if text is not None else None
        new = fix_yaml(text, n) if n is not None else None
        if new is not None:
            changes.append(Change(issue.path, text.split("\n")[n], new.split("\n")[n], line=n))
    return changes


# --- the edits ---------------------------------------------------------------------------------------------------


def _parts(text: str) -> tuple[int, int, int]:
    """(start, end) of the frontmatter block and the start of the body. With no frontmatter that parses, the
    whole text is body, as in validate."""
    try:
        frontmatter.loads(text)
    except Exception:
        return 0, 0, 0
    marks = list(BOUNDARY.finditer(text)) if BOUNDARY.match(text) else []
    if len(marks) < 2:
        return 0, 0, 0
    return marks[0].end(), marks[1].start(), marks[1].end()


def _spans(text: str) -> list[tuple[int, int]]:
    """The spans of the text where validate reads links: the frontmatter lines of keys that are not reference
    keys, and the body outside code."""
    start, end, body = _parts(text)
    spans, pos, key = [], start, None
    for line in text[start:end].splitlines(keepends=True):
        top = TOP_KEY.match(line)
        if top:
            key = top.group(1).strip()
        if key not in REF_KEYS and not line.lstrip().startswith("#"):
            spans.append((pos, pos + len(line)))
        pos += len(line)
    pos, fence = body, None
    for line in text[body:].splitlines(keepends=True):
        m = FENCE.match(line)
        if fence is not None:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and not line[m.end() :].strip():
                fence = None
        elif m:
            fence = m.group(1)
        else:
            last = 0
            for code in INLINE_CODE.finditer(line):
                spans.append((pos + last, pos + code.start()))
                last = code.end()
            spans.append((pos + last, pos + len(line)))
        pos += len(line)
    return spans


def edit_links(text: str, fixes: dict[str, str]) -> tuple[str, Counter]:
    """Change each link outside code whose target has a link key in `fixes` to the new name. Returns the new text
    and the number of links changed for each key."""
    out, last, counts = [], 0, Counter()
    for start, end in _spans(text):
        for m in LINK.finditer(text, start, end):
            raw = m.group(1)
            target = _clean(raw)
            key = _link_key(target)
            if key not in fixes:
                continue
            at = m.start(1) + raw.index(target)
            out += [text[last:at], fixes[key]]
            last = at + len(target)
            counts[key] += 1
    return "".join(out) + text[last:], counts


def fix_yaml(text: str, n: int) -> str | None:
    """The text with the value of frontmatter line n in double quotes, or None when the fixed frontmatter does
    not parse, or a key other than the fixed one differs from the frontmatter with the line removed."""
    lines = text.split("\n")
    try:
        block = frontmatter.YAMLHandler().split(text)[0].split("\n")
    except ValueError:
        return None
    if n >= len(block) or n >= len(lines) or block[n] != lines[n]:
        return None
    key, _, value = lines[n].partition(": ")
    quoted = value.replace("\\", "\\\\").replace('"', '\\"')
    new = [*lines[:n], f'{key}: "{quoted}"', *lines[n + 1 :]]
    try:
        fixed = dict(frontmatter.loads("\n".join(new)).metadata)
        rest = yaml.safe_load("\n".join(block[:n] + block[n + 1 :])) or {}
    except Exception:
        return None
    if fixed.get(key) != value or {k: v for k, v in fixed.items() if k != key} != rest:
        return None
    return "\n".join(new)


# --- the apply ---------------------------------------------------------------------------------------------------


def _read(cfg: Config, path: str) -> str | None:
    try:
        return cfg.note_path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _edit(text: str, changes: list[Change]) -> str | None:
    """The new text, or None when one change no longer applies (the old text is not there)."""
    if changes[0].line is not None:
        n = changes[0].line
        lines = text.split("\n")
        if n >= len(lines) or lines[n] != changes[0].old:
            return None
        return fix_yaml(text, n)
    fixes = {_link_key(c.old): c.new for c in changes}
    new, counts = edit_links(text, fixes)
    return new if all(counts[key] for key in fixes) else None


def apply(cfg: Config, changes: list[Change], dry_run: bool = False) -> Run:
    """Apply the changes note by note, with the steps of the module docstring. A dry run writes nothing."""
    now = time.time()
    run = Run(changes)
    by_note: dict[str, list[Change]] = {}
    for change in changes:
        by_note.setdefault(change.path, []).append(change)
    if not dry_run and by_note:
        run.folder = cfg.data_dir / "fix-backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run.folder.mkdir(parents=True)
        with (run.folder / "log.tsv").open("w", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(["path", "old", "new"])
    for path, items in by_note.items():
        file = cfg.note_path(path)
        if now - file.stat().st_mtime < MIN_AGE:
            run.recent.append(path)
            continue
        if dry_run:
            continue
        backup = run.folder / path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, backup)
        text = _read(cfg, path)
        new = _edit(text, items) if text is not None else None
        if new is None:
            run.gone.append(path)
            continue
        temp = file.with_name(file.name + ".tmp")
        temp.write_bytes(new.encode("utf-8"))
        os.replace(temp, file)
        with (run.folder / "log.tsv").open("a", newline="") as f:
            csv.writer(f, delimiter="\t").writerows([c.path, c.old, c.new] for c in items)
        run.written.append(path)
    return run


def logged_paths(folder: Path) -> list[str]:
    """The note keys of the log of a run, each once."""
    with (folder / "log.tsv").open(newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    return list(dict.fromkeys(row[0] for row in rows))


def unresolved(cfg: Config, conn: sqlite3.Connection, run: Run) -> int:
    """Run validate on the written notes. Returns the number of fixes that did not take: a fixed link that still
    resolves to no note, or a note that still has bad YAML."""
    if not run.written:
        return 0
    fixed = {(c.path, _link_key(c.new)) for c in run.changes if c.line is None and c.path in run.written}
    count = 0
    for issue in check_paths(cfg, conn, run.written).issues:
        if issue.kind == "bad-yaml":
            count += 1
        match = h.BROKEN_LINK.fullmatch(issue.message) if issue.kind == "broken-link" else None
        count += bool(match) and (issue.path, _link_key(match.group(1))) in fixed
    return count


def format_changes(kind: str, run: Run, dry_run: bool) -> list[str]:
    lines = []
    for c in run.changes:
        if kind == "links":
            lines.append(f"{c.path}: [[{c.old}]] -> [[{c.new}]]")
        else:
            lines.append(f"{c.path}: line {c.line + 1}: {c.old} -> {c.new}")
    lines += [f"skipped: {p} (changed in the last {MIN_AGE // 60} minutes)" for p in run.recent]
    lines += [f"skipped: {p} (the old text is not there any more)" for p in run.gone]
    notes = len({c.path for c in run.changes})
    tail = " (dry run, nothing written)" if dry_run else f", {len(run.written)} notes written"
    lines.append(f"{kind}: {len(run.changes)} in {notes} notes{tail}")
    return lines
