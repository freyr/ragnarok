"""Survey of one vault: the facts that a person needs to write its [vault_rules.<vault>] table.

The survey prints folder names, file names, frontmatter keys, a few frontmatter values and counts.
It never prints note text. The values of notes that match private_patterns are not listed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

import frontmatter

from .config import Config
from .metadata import JIRA_RE, is_private, iter_notes

VALUE_KEYS = ("type", "status", "project", "area")
# Strings that look like ticket keys but are not: standards, encodings, hashes, vulnerability ids.
NON_TICKET_PREFIXES = {
    "UTF", "ISO", "SHA", "RFC", "TLS", "SSL", "HTTP", "PSR", "CVE", "CWE", "MD", "AES", "RSA", "IEEE",
    "ASCII", "GPT", "USB", "PDF", "SQL", "PHP", "TCP", "UDP", "DNS", "RGB", "NF", "EN", "DIN", "ANSI",
}  # fmt: skip
ARCHIVE_RE = re.compile(r"^_?archive[sd]?$", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
PREFIX_RE = re.compile(r"^([^-_\s]+[-_])")
MARKER_MIN_FOLDERS = 3


def _ranked(counter: Counter, top: int) -> tuple[dict, int]:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(items[:top]), max(len(items) - top, 0)


def _folders(counter: Counter, top: int) -> dict:
    folders, more = _ranked(counter, top)
    return {"folders": folders, "more": more}


def survey(cfg: Config, vault: str, top: int = 30) -> dict:
    """The facts of the folder `vault` under cfg.root. The folder need not be in cfg.vaults."""
    notes = 0
    with_frontmatter = 0
    malformed: list[str] = []
    by_depth: Counter = Counter()
    level1: Counter = Counter()
    level2: Counter = Counter()
    keys: Counter = Counter()
    values: dict[str, Counter] = {key: Counter() for key in VALUE_KEYS}
    folders_of_name: dict[str, set[str]] = defaultdict(set)
    prefixes: Counter = Counter()
    date_prefixed = 0
    archive: Counter = Counter()
    ticket_keys: dict[str, set[str]] = defaultdict(set)
    ticket_notes: Counter = Counter()

    for rel in iter_notes(replace(cfg, vaults=[vault])):
        notes += 1
        parts = rel.split("/")[1:]
        folders, name = parts[:-1], parts[-1]
        by_depth[len(folders)] += 1
        if folders:
            level1[folders[0]] += 1
        if len(folders) > 1:
            level2["/".join(folders[:2])] += 1
        folders_of_name[name].add("/".join(folders))
        for folder in {f for f in folders if ARCHIVE_RE.match(f)}:
            archive[folder] += 1

        stem = name.removesuffix(".md")
        if DATE_RE.match(stem):
            date_prefixed += 1
        elif (m := PREFIX_RE.match(stem)) and not m.group(1)[:-1].isdigit():
            prefixes[m.group(1)] += 1

        text = (cfg.root / rel).read_text(errors="replace")
        try:
            metadata = dict(frontmatter.loads(text).metadata)
        except Exception:
            metadata = {}
            malformed.append(rel)
        if metadata:
            with_frontmatter += 1
            keys.update(str(key) for key in metadata)
            if not is_private(cfg, rel):
                for key in VALUE_KEYS:
                    value = metadata.get(key)
                    if isinstance(value, (str, int, float)) and str(value):
                        values[key][str(value)] += 1

        found = {(m.group(1), m.group(0)) for m in JIRA_RE.finditer(text) if m.group(1) not in NON_TICKET_PREFIXES}
        for prefix, key in found:
            ticket_keys[prefix].add(key)
        ticket_notes.update({prefix for prefix, _ in found})

    tickets = {p: {"keys": len(ticket_keys[p]), "notes": ticket_notes[p]} for p in ticket_keys}
    ranked_tickets = sorted(tickets.items(), key=lambda kv: (-kv[1]["notes"], kv[0]))
    return {
        "vault": vault,
        "notes": notes,
        "with_frontmatter": with_frontmatter,
        "malformed_frontmatter": malformed,
        "by_depth": {str(d): n for d, n in sorted(by_depth.items())},
        "level1": _folders(level1, top),
        "level2": _folders(level2, top),
        "keys": dict(sorted(keys.items(), key=lambda kv: (-kv[1], kv[0]))),
        "values": {key: _ranked(counter, top)[0] for key, counter in values.items()},
        "marker_candidates": dict(
            sorted(
                ((n, len(f)) for n, f in folders_of_name.items() if len(f) >= MARKER_MIN_FOLDERS),
                key=lambda kv: (-kv[1], kv[0]),
            )
        ),
        "project_markers": {m: len(folders_of_name.get(m, ())) for m in cfg.project_markers},
        "prefixes": _ranked(prefixes, top)[0],
        "date_prefixed": date_prefixed,
        "archive_folders": dict(sorted(archive.items(), key=lambda kv: (-kv[1], kv[0]))),
        "tickets": dict(ranked_tickets[:top]),
    }


def format_text(facts: dict) -> str:
    """The facts as plain text, one section for each rule input."""
    out = [f"vault {facts['vault']}: {facts['notes']} notes, {facts['with_frontmatter']} with frontmatter"]

    def section(title: str, rows: dict, fmt=str):
        out.append(f"\n{title}")
        out.extend([f"  {key}: {fmt(value)}" for key, value in rows.items()] or ["  (none)"])

    out.append(f"frontmatter that cannot be parsed: {len(facts['malformed_frontmatter'])}")
    out.extend(f"  {rel}" for rel in facts["malformed_frontmatter"])
    section("notes by folder depth (0 = top of the vault)", facts["by_depth"])
    for level in ("level1", "level2"):
        section(f"folders at level {level[-1]} (notes)", facts[level]["folders"])
        if facts[level]["more"]:
            out.append(f"  and {facts[level]['more']} more")
    section("frontmatter keys (notes)", facts["keys"])
    for key, rows in facts["values"].items():
        section(f"values of {key} (notes, private notes left out)", rows)
    section("file names in 3 or more folders (folders)", facts["marker_candidates"])
    section("configured project_markers (folders)", facts["project_markers"])
    section("file name prefixes (notes)", facts["prefixes"])
    out.append(f"names that start with a date YYYY-MM-DD: {facts['date_prefixed']}")
    section("archive folder names (notes)", facts["archive_folders"])
    section("ticket key prefixes", facts["tickets"], lambda v: f"{v['keys']} keys in {v['notes']} notes")
    return "\n".join(out)
