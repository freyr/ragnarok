"""Resolve the metadata of a note: frontmatter first, then the path rules of its vault.

Sensitivity comes only from the configured path patterns, never from frontmatter.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import frontmatter

from .config import Config, KbConfig, VaultRules

# the closed document types: the names of the default `kb.doc_types`; `resolve` reads the configured ones
TYPES = frozenset(KbConfig().doc_types)

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
    "howto": "guide",
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
    class_: str = "none"  # the class of the type: state, record or none
    mode: str = "files"
    collection: str = ""
    # a reference stays as text; owner, primary_subject and superseded_by stay empty until the references resolve
    owner: str | None = None
    owner_text: str | None = None
    primary_subject: str | None = None
    subjects: list[str] = field(default_factory=list)  # the target texts of the subject links, at most three
    task: str | None = None
    identity_key: str | None = None
    canonical: bool = False
    superseded_by: str | None = None
    superseded_by_text: str | None = None
    as_of: str | None = None
    review_by: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def iter_notes(cfg: Config) -> Iterator[str]:
    """Yield the keys (`<vault>/<path in the vault>`) of all Markdown notes of the configured vaults."""
    for vault in cfg.vaults:
        base = cfg.vault_dir(vault)
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
                    yield f"{vault}/{entry.relative_to(base).as_posix()}"


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


def link_target(value) -> str | None:
    """The target text of a wiki-link, or the text of an ID. Empty gives None."""
    text = str(value).strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].split("|")[0].split("#")[0].strip()
    return text or None


def _first_target(value) -> str | None:
    return next(iter(_targets(value)), None)


def _targets(value) -> list[str]:
    """The target texts of one link or a list of links, in order. A comma list in one string is a list."""
    if isinstance(value, str) and not value.strip().startswith("[["):
        value = _as_list(value)
    elif isinstance(value, str):
        value = [value]
    return [t for item in value or [] if (t := link_target(item))]


def _iso_date(value) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        return None


def _add_months(day: str, months: int) -> str:
    start = date.fromisoformat(day)
    index = start.year * 12 + start.month - 1 + months
    year, month = divmod(index, 12)
    month += 1
    return date(year, month, min(start.day, calendar.monthrange(year, month)[1])).isoformat()


def _identity_key(one_per: str, parts: dict[str, str | None], vault: str, note_type: str) -> str | None:
    """`collection|type|<key of one_per>`. Null when a part is absent or one_per is none."""
    wanted = {
        "subject+task": ["subject", "task"],
        "subject": ["subject"],
        "entity": ["subject"],
        "project": ["project"],
    }.get(one_per)
    if wanted is None or any(not parts[name] for name in wanted):
        return None
    return "|".join([vault, note_type, *(parts[name] for name in wanted)])


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


def _summary_archived(cfg: Config, folder: str) -> bool:
    """A project is archived when its SUMMARY.md has state superseded or abandoned. State done is current."""
    path = cfg.note_path(folder) / "SUMMARY.md"
    try:
        mtime = path.stat().st_mtime_ns
    except FileNotFoundError:
        return False
    return _summary_state_archived(path, mtime)


@lru_cache(maxsize=4096)
def _summary_state_archived(path: Path, mtime_ns: int) -> bool:
    fm, _ = read_note(path)
    return str(fm.get("state", "")).strip().lower() in ("superseded", "abandoned")


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
    project_archived: Callable[[str], bool] | None = None,
) -> NoteMeta:
    """Resolve the metadata of the note with the key rel."""
    fm = fm or {}
    if has_marker is None:

        def has_marker(folder: str) -> bool:
            return any((cfg.note_path(folder) / m).is_file() for m in cfg.project_markers)

    if project_archived is None:

        def project_archived(folder: str) -> bool:
            return _summary_archived(cfg, folder)

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

    doc_types = cfg.kb.doc_types
    raw_type = str(fm.get("type", "")).strip().lower()
    if raw_type in doc_types:
        note_type, sources["type"] = raw_type, "frontmatter"
    elif (path_type := _type_from_path(parts)) is not None:
        note_type, sources["type"] = path_type, "path"
    elif raw_type in TYPE_MAP:
        note_type, sources["type"] = TYPE_MAP[raw_type], "frontmatter-map"
    else:
        note_type, sources["type"] = "note", "default"

    raw_status = str(fm.get("status", "")).strip().lower()
    if raw_status in ("archived", "superseded", "redirect"):
        status, sources["status"] = raw_status, "frontmatter"
    elif folder is not None and project_archived(folder):
        status, sources["status"] = "archived", "summary"
    else:
        status, sources["status"] = ("archived" if archived else "active"), "path"

    description = fm.get("description")
    description = description.strip() if isinstance(description, str) and description.strip() else None
    if description:
        sources["description"] = "frontmatter"

    jira = sorted(set(_as_list(fm.get("jira"))) | set(find_jira(cfg, vault, body)))

    doc_type = doc_types[note_type]
    subjects = _targets(fm.get("subject"))[:3]
    task = fm.get("task")
    task = task.strip() if isinstance(task, str) and task.strip() else None
    as_of = _iso_date(fm.get("as_of")) if fm.get("as_of") is not None else None
    review_by = _iso_date(fm.get("review_by")) if fm.get("review_by") is not None else None
    if review_by is None and as_of is not None and doc_type.review_months:
        review_by = _add_months(as_of, doc_type.review_months)
    key_parts = {
        "subject": subjects[0].lower() if subjects else None,
        "task": " ".join(task.lower().split()) if task else None,
        "project": project if project != "none" else None,
    }
    owner_text = _first_target(fm.get("owner"))
    superseded_by_text = _first_target(fm.get("superseded_by"))

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
        class_=doc_type.class_,
        collection=vault,
        owner_text=owner_text,
        subjects=subjects,
        task=task,
        identity_key=_identity_key(doc_type.one_per, key_parts, vault, note_type),
        canonical=fm.get("canonical") is True,
        superseded_by_text=superseded_by_text,
        as_of=as_of,
        review_by=review_by,
    )


def resolve_file(cfg: Config, rel: str) -> NoteMeta:
    fm, body = read_note(cfg.note_path(rel))
    return resolve(cfg, rel, fm, body)
