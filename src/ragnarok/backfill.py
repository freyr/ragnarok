"""Propose the schema frontmatter for existing notes, and apply an approved proposal.

The proposal is a YAML file outside the vaults. `apply` archives the vault first, then inserts only the
missing keys as new lines before the closing `---`, so the existing frontmatter and the body keep every byte.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config
from .metadata import TYPES, iter_notes, parse_note, resolve
from .redact import redact

KEYS = ("project", "area", "type", "status", "description")
# instruction files for agents, not notes: their frontmatter has other meanings
SKIP_NAMES = {"CLAUDE.md", "AGENTS.md"}
WORD_RE = re.compile(r"[a-z0-9]+")
EXCERPT_LINES = 40
EXCERPT_CHARS = 2000
MAX_WORDS = 25
H1_RE = re.compile(r"^#\s+(.+)$", re.M)
INSTRUCTIONS = (
    "Write one description for each row. Answer with one JSON object: "
    '{path: {"description": str, "type": str (optional)}}. '
    f"Write in English, {MAX_WORDS} words or fewer. A type is one of: {', '.join(sorted(TYPES))}."
)
STOP = {"md", "the", "and", "of", "a", "to", "for", "in", "on", "plan", "feat", "fix", "docs", "notes"}


def proposal_path(cfg: Config, vault: str) -> Path:
    return cfg.data_dir / "proposals" / f"backfill-{vault}.yaml"


def backup_dir(cfg: Config) -> Path:
    return cfg.data_dir / "backups"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOP and not w.isdigit()}


def _match_project(name: str, slugs: set[str]) -> str | None:
    """The project slug with the most words (two or more) in common with the file name, or None."""
    words = _words(Path(name).stem)
    best, best_n = None, 1  # one common word is too weak: "review", "development"
    for slug in sorted(slugs):
        n = len(words & _words(slug))
        if n > best_n:
            best, best_n = slug, n
    return best


def propose(cfg: Config, vault: str) -> dict:
    """Return the proposal for one vault: one row for each note with a missing schema key."""
    notes, malformed = [], []
    for rel in iter_notes(cfg):
        if rel.split("/")[0] != vault or rel.rsplit("/", 1)[-1] in SKIP_NAMES:
            continue
        try:
            text = (cfg.root / rel).read_bytes().decode()
        except UnicodeDecodeError:
            malformed.append(rel)  # apply writes only notes that decode as UTF-8
            continue
        fm, body = parse_note(text)
        if _checked_insert(text, fm, body, {"project": "trial"}) is None:
            malformed.append(rel)  # a frontmatter that apply cannot extend safely: the user corrects it by hand
            continue
        notes.append((rel, text, fm, resolve(cfg, rel, fm, body)))

    slugs = {m.project for _, _, _, m in notes if m.project != "none"}
    rows = []
    for rel, text, fm, meta in notes:
        # a status from the SUMMARY.md stage follows the stage; written to the note, it would outlive a reopen
        derived = {"status"} if meta.sources.get("status") == "summary" else set()
        missing = [k for k in KEYS if k not in fm and k not in derived]
        if not missing:
            continue
        values = {"project": meta.project, "area": meta.area, "type": meta.type, "status": meta.status}
        sources = {k: meta.sources.get(k, "path") for k in values}
        uncertain = False
        if "project" in missing and meta.project == "none" and (match := _match_project(rel, slugs)):
            values["project"], sources["project"], uncertain = match, "match", True
        private = meta.sensitivity == "private"
        row = {
            "path": rel,
            "approve": True,
            "uncertain": uncertain,
            "private": private,
            "hash": _hash(text),
            "set": {k: values.get(k, "") for k in missing},
            "source": {k: sources[k] for k in missing if k in sources},
        }
        rows.append(row)
    return {"vault": vault, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "malformed": malformed, "rows": rows}


def coverage(cfg: Config) -> dict:
    """The share of notes in each vault whose frontmatter has each schema key."""
    counts: dict[str, dict[str, int]] = {}
    for rel in iter_notes(cfg):
        vault = rel.split("/")[0]
        fm, _ = parse_note((cfg.root / rel).read_text(errors="replace"))
        c = counts.setdefault(vault, {"notes": 0, **dict.fromkeys(KEYS, 0)})
        c["notes"] += 1
        for k in KEYS:
            c[k] += k in fm
    return {
        vault: {"notes": c["notes"], **{k: round(c[k] / c["notes"], 2) for k in KEYS}} for vault, c in counts.items()
    }


def write_proposal(cfg: Config, vault: str, proposal: dict) -> Path:
    path = proposal_path(cfg, vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(proposal, allow_unicode=True, sort_keys=False, width=1000))
    return path


def read_proposal(cfg: Config, vault: str) -> dict:
    return yaml.safe_load(proposal_path(cfg, vault).read_text())


def describe_path(cfg: Config, vault: str) -> Path:
    return cfg.data_dir / "proposals" / f"describe-{vault}.json"


def export_rows(cfg: Config, proposal: dict) -> list[dict]:
    """One item for each public row that needs a description: path, title, type and the redacted first lines."""
    items = []
    for row in proposal["rows"]:
        if "description" not in row["set"] or row["private"]:
            continue
        fm, body = parse_note((cfg.root / row["path"]).read_text(errors="replace"))
        h1 = H1_RE.search(body)
        title = fm.get("title") or (h1.group(1).strip() if h1 else Path(row["path"]).stem)
        # redact the whole body first: a cut can split a multi-line credential so that no pattern matches
        excerpt = "\n".join(redact(body)[0].splitlines()[:EXCERPT_LINES])[:EXCERPT_CHARS].strip()
        items.append(
            {
                "path": row["path"],
                "title": str(title),
                "type": row["set"].get("type", ""),
                "excerpt": excerpt,
            }
        )
    return items


def write_export(cfg: Config, vault: str, proposal: dict) -> tuple[Path, int]:
    items = export_rows(cfg, proposal)
    path = describe_path(cfg, vault)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"instructions": INSTRUCTIONS, "types": sorted(TYPES), "rows": items}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path, len(items)


@dataclass
class MergeResult:
    rows: int = 0
    merged: int = 0
    types: int = 0
    redacted: int = 0
    missing: list[str] = field(default_factory=list)
    too_long: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def merge_descriptions(proposal: dict, descriptions: dict) -> MergeResult:
    """Put the descriptions and the types from Claude into the proposal rows. approve and hash do not change."""
    result = MergeResult(rows=len(proposal["rows"]))
    paths = {row["path"] for row in proposal["rows"]}
    result.unknown = sorted(set(descriptions) - paths)
    for row in proposal["rows"]:
        needs = "description" in row["set"]
        if row["private"]:
            if needs:
                row["set"]["description"] = ""
            continue
        item = descriptions.get(row["path"])
        if not item or not item.get("description"):
            if needs:
                result.missing.append(row["path"])
            continue
        if needs:
            text, count = redact(" ".join(item["description"].split()))
            result.redacted += bool(count)
            if len(text.split()) > MAX_WORDS:
                result.too_long.append(row["path"])
            row["set"]["description"] = text
            result.merged += 1
        new_type = item.get("type")
        if (
            new_type in TYPES
            and new_type != row["set"].get("type")
            and "type" in row["set"]
            and row["source"].get("type") == "default"
        ):
            row["set"]["type"], row["source"]["type"] = new_type, "claude"
            result.types += 1
    return result


def archive(cfg: Config, vault: str) -> Path:
    """Copy the vault to a tar.gz in the backup folder and check that every note is in it."""
    out = backup_dir(cfg) / f"{vault}-{time.strftime('%Y%m%d-%H%M%S')}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = cfg.root / vault

    def keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        return None if any(p in cfg.skip_dirs for p in Path(info.name).parts) else info

    with tarfile.open(out, "w:gz") as tar:
        tar.add(base, arcname=vault, filter=keep)
    with tarfile.open(out, "r:gz") as tar:
        names = set(tar.getnames())
    absent = [rel for rel in iter_notes(cfg) if rel.split("/")[0] == vault and rel not in names]
    if absent:
        raise RuntimeError(f"the archive {out} does not hold {len(absent)} notes, for example {absent[0]}")
    return out


def _split_frontmatter(text: str) -> tuple[int, str] | None:
    """Return (offset of the closing --- line, newline) for a note that starts with frontmatter, else None."""
    nl = "\r\n" if text.startswith("---\r\n") else "\n"
    if not text.startswith("---" + nl):
        return None
    pos = len("---" + nl)
    if text.startswith("---", pos) and text[pos + 3 : pos + 3 + len(nl)] in (nl, ""):
        return pos, nl  # empty frontmatter
    m = re.compile(r"^---[ \t]*\r?$", re.M).search(text, pos)
    if m is None:
        return None
    return m.start(), nl


def insert_keys(text: str, values: dict) -> str:
    """Add the keys to the frontmatter of text (or a new frontmatter), with no other change."""
    block = yaml.safe_dump(values, allow_unicode=True, sort_keys=False, width=1000)
    split = _split_frontmatter(text)
    if split is None:
        nl = "\r\n" if "\r\n" in text else "\n"
        return "---" + nl + block.replace("\n", nl) + "---" + nl + text
    at, nl = split
    if at > 0 and not text[:at].endswith(nl):
        block = nl + block  # the last frontmatter line has no newline before the closing ---
    return text[:at] + block.replace("\n", nl) + text[at:]


def _checked_insert(text: str, fm: dict, body: str, values: dict) -> str | None:
    """insert_keys, but None when the result does not parse to the same body and frontmatter plus the new keys."""
    new = insert_keys(text, values)
    new_fm, new_body = parse_note(new)
    if new_body != body or new_fm != {**fm, **values}:
        return None
    return new


@dataclass
class ApplyResult:
    archive: str = ""
    written: int = 0
    keys: int = 0
    skipped: list[str] = field(default_factory=list)


def apply(cfg: Config, vault: str, proposal: dict) -> ApplyResult:
    if proposal.get("vault") != vault:
        raise ValueError(f"the proposal is for the vault {proposal.get('vault')}, not {vault}")
    result = ApplyResult(archive=str(archive(cfg, vault)))
    for row in proposal["rows"]:
        rel = row["path"]
        if not row.get("approve"):
            continue
        path = cfg.root / rel
        if not path.is_file():
            result.skipped.append(f"{rel}: not found")
            continue
        text = path.read_bytes().decode(errors="replace")
        if _hash(text) != row["hash"]:
            result.skipped.append(f"{rel}: changed after the proposal")
            continue
        fm, body = parse_note(text)
        values = {}
        for k, v in row["set"].items():
            if k not in KEYS or k in fm or v is None or (isinstance(v, str) and not v.strip()):
                continue
            values[k] = v.strip() if isinstance(v, str) else v
        if not values:
            continue
        new = _checked_insert(text, fm, body, values)
        if new is None:
            result.skipped.append(f"{rel}: the check after the insert failed")
            continue
        path.write_text(new)
        result.written += 1
        result.keys += len(values)
    return result
