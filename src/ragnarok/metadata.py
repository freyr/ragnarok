"""Resolve the metadata of a note: frontmatter first, then the path rules of its vault.

Sensitivity comes only from the configured path patterns, never from frontmatter.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import frontmatter

from .config import Config, VaultRules

TYPES = {
    "drivers",
    "summary",
    "research",
    "spec",
    "plan",
    "brainstorm",
    "review",
    "decision",
    "guide",
    "reference",
    "meeting",
    "log",
    "template",
    "note",
}

# docs/<kind>/ and other folders that carry the type in the older layout
FOLDER_TYPES = {
    "plans": "plan",
    "brainstorms": "brainstorm",
    "research": "research",
    "specs": "spec",
    "adrs": "decision",
    "decisions": "decision",
    "solutions": "reference",
    "reviews": "review",
    "todos": "log",
    "99-Templates": "template",
}

# existing raw `type:` values -> schema types
TYPE_MAP = {
    "research": "research",
    "investigation": "research",
    "analysis": "research",
    "content-analysis": "research",
    "spec": "spec",
    "specification": "spec",
    "design": "spec",
    "brief": "spec",
    "plan": "plan",
    "feat": "plan",
    "fix": "plan",
    "refactor": "plan",
    "brainstorm": "brainstorm",
    "proposal": "brainstorm",
    "one-pager": "brainstorm",
    "pitch-input": "brainstorm",
    "review": "review",
    "incident": "review",
    "incident-rca": "review",
    "sla-review": "review",
    "adr": "decision",
    "decision": "decision",
    "guide": "guide",
    "how-to": "guide",
    "howto": "guide",
    "tutorial": "guide",
    "runbook": "guide",
    "facilitator-runbook": "guide",
    "knowledge-base": "reference",
    "reference": "reference",
    "meeting": "meeting",
    "log": "log",
    "journal": "log",
    "digest": "log",
    "template": "template",
    "drivers": "drivers",
    "summary": "summary",
}

JIRA_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})-(\d+)\b")


@dataclass
class NoteMeta:
    vault: str
    path: str  # relative to the vault root, starts with the vault name
    title: str
    project: str
    area: str
    type: str
    status: str
    sensitivity: str
    description: str | None = None
    jira: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def iter_notes(cfg: Config) -> Iterator[str]:
    """Yield the paths (relative to cfg.root) of all Markdown notes of the configured vaults."""
    for vault in cfg.vaults:
        base = cfg.root / vault
        if not base.is_dir():
            continue
        stack = [base]
        while stack:
            folder = stack.pop()
            for entry in sorted(folder.iterdir()):
                if entry.is_dir():
                    if not cfg.skip_dir(entry.name):
                        stack.append(entry)
                elif entry.suffix == ".md":
                    yield entry.relative_to(cfg.root).as_posix()


def is_private(cfg: Config, rel: str) -> bool:
    return any(fnmatchcase(rel, pattern) for pattern in cfg.private_patterns)


def parse_note(text: str) -> tuple[dict, str]:
    """Return (frontmatter, body). Malformed frontmatter counts as none."""
    try:
        post = frontmatter.loads(text)
        return dict(post.metadata), post.content
    except Exception:
        return {}, text


def read_note(path: Path) -> tuple[dict, str]:
    return parse_note(path.read_text(errors="replace"))


def find_jira(cfg: Config, vault: str, text: str) -> list[str]:
    allowed = set(cfg.jira_prefixes) | set(cfg.rules(vault).jira_prefixes)
    keys = {f"{m.group(1)}-{m.group(2)}" for m in JIRA_RE.finditer(text) if m.group(1) in allowed}
    return sorted(keys)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v) for v in value]


def normalize_project(value: str) -> str:
    """A project given as a wiki-link to its SUMMARY/DRIVERS/README note becomes the folder name."""
    value = value.strip()
    if not (value.startswith("[[") and value.endswith("]]")):
        return value
    target = value[2:-2].split("|")[0].split("#")[0].strip()
    segments = [s for s in target.split("/") if s]
    if not segments:
        return value
    if segments[-1].removesuffix(".md") in ("SUMMARY", "DRIVERS", "README") and len(segments) > 1:
        return segments[-2]
    return segments[-1].removesuffix(".md")


def _closest_project_dir(parts: list[str], has_marker: Callable[[str], bool]) -> int | None:
    """Index into parts of the deepest folder (below the vault) that holds a project marker."""
    for depth in range(len(parts) - 1, 1, -1):
        if has_marker("/".join(parts[:depth])):
            return depth - 1
    return None


def _project_from_folders(folders: list[str], rules: VaultRules, archive: str) -> str:
    for parent in rules.project_parents:
        prefix = parent.split("/")
        if folders[: len(prefix)] == prefix and len(folders) > len(prefix):
            return folders[len(prefix)]
    folders = [f for f in folders if f != archive]
    if len(folders) >= rules.project_depth:
        return folders[rules.project_depth - 1]
    if rules.project_fallback and folders:
        return folders[0]
    return "none"


def _path_rules(cfg: Config, parts: list[str], has_marker: Callable[[str], bool]) -> tuple[str, str, bool, str | None]:
    """Return (area, project, archived, project folder) from the path. parts = [vault, folder..., file]."""
    vault, folders = parts[0], parts[1:-1]
    rules = cfg.rules(vault)
    archived = cfg.archive_folder in folders
    marker = _closest_project_dir(parts, has_marker)
    ignored = bool(folders) and folders[0] in rules.ignore_folders

    if rules.area != "folder":
        area = rules.area
    elif folders and not ignored:
        area = folders[0].lower()
    else:
        area = rules.area_default or vault.lower()

    if marker is not None:
        project = parts[marker]
    elif ignored:
        project = "none"
    else:
        project = _project_from_folders(folders, rules, cfg.archive_folder)
    folder = "/".join(parts[: marker + 1]) if marker is not None else None
    return area, project, archived, folder


def _summary_done(cfg: Config, folder: str) -> bool:
    """A project is finished when the stage in its SUMMARY.md frontmatter is DONE."""
    path = cfg.root / folder / "SUMMARY.md"
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return False
    return _stage_done(path, mtime)


@lru_cache(maxsize=4096)
def _stage_done(path: Path, mtime_ns: int) -> bool:
    fm, _ = read_note(path)
    return str(fm.get("stage", "")).strip().upper() == "DONE"


def _type_from_path(parts: list[str]) -> str | None:
    name = parts[-1]
    if name == "DRIVERS.md":
        return "drivers"
    if name == "SUMMARY.md":
        return "summary"
    for prefix in ("research", "spec", "plan"):
        if name.startswith(prefix + "-"):
            return prefix
    for folder in reversed(parts[1:-1]):
        if folder in FOLDER_TYPES:
            return FOLDER_TYPES[folder]
    return None


def resolve(
    cfg: Config,
    rel: str,
    fm: dict | None = None,
    body: str = "",
    has_marker: Callable[[str], bool] | None = None,
    project_done: Callable[[str], bool] | None = None,
) -> NoteMeta:
    """Resolve the metadata of the note at rel (relative to cfg.root)."""
    fm = fm or {}
    if has_marker is None:

        def has_marker(folder: str) -> bool:
            return any((cfg.root / folder / m).is_file() for m in cfg.project_markers)

    if project_done is None:

        def project_done(folder: str) -> bool:
            return _summary_done(cfg, folder)

    parts = rel.split("/")
    vault = parts[0]
    area, project, archived, folder = _path_rules(cfg, parts, has_marker)
    sources: dict[str, str] = {}

    def pick(key: str, path_value: str) -> str:
        value = fm.get(key)
        if isinstance(value, str) and value.strip():
            sources[key] = "frontmatter"
            return value.strip()
        sources[key] = "path"
        return path_value

    project = normalize_project(pick("project", project))
    area = pick("area", area).lower()

    raw_type = str(fm.get("type", "")).strip().lower()
    if raw_type in TYPES:
        note_type, sources["type"] = raw_type, "frontmatter"
    elif (path_type := _type_from_path(parts)) is not None:
        note_type, sources["type"] = path_type, "path"
    elif raw_type in TYPE_MAP:
        note_type, sources["type"] = TYPE_MAP[raw_type], "frontmatter-map"
    else:
        note_type, sources["type"] = "note", "default"

    if str(fm.get("status", "")).strip().lower() == "archived":
        status, sources["status"] = "archived", "frontmatter"
    elif folder is not None and project_done(folder):
        status, sources["status"] = "archived", "summary"
    else:
        status, sources["status"] = ("archived" if archived else "active"), "path"

    description = fm.get("description")
    description = description.strip() if isinstance(description, str) and description.strip() else None
    if description:
        sources["description"] = "frontmatter"

    jira = sorted(set(_as_list(fm.get("jira"))) | set(find_jira(cfg, vault, body)))

    return NoteMeta(
        vault=vault,
        path=rel,
        title=Path(parts[-1]).stem,
        project=project,
        area=area,
        type=note_type,
        status=status,
        sensitivity="private" if is_private(cfg, rel) else "normal",
        description=description,
        jira=jira,
        sources=sources,
    )


def resolve_file(cfg: Config, rel: str) -> NoteMeta:
    fm, body = read_note(cfg.root / rel)
    return resolve(cfg, rel, fm, body)
