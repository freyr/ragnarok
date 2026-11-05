"""Configuration: vault locations, exclusions, sensitive paths, path rules, model and splitter settings.

The home folder holds config.toml, data/ (index, backups, proposals) and logs/. It is the repo checkout,
or the folder in RAGNAROK_HOME. A relative path in config.toml is relative to the folder of the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PATH_KEYS = ("root", "data_dir", "index_path")


def home_dir() -> Path:
    """RAGNAROK_HOME, else the repo checkout that holds this source file."""
    if env := os.environ.get("RAGNAROK_HOME"):
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists():
        return checkout
    raise RuntimeError("ragnarok is not run from its repo checkout. Set RAGNAROK_HOME to the home folder.")


@dataclass
class VaultRules:
    """How the path of a note without frontmatter gives its area and project. One table for each vault."""

    # "folder": the first folder below the vault, in lower case; any other value is the area of all notes
    area: str = "folder"
    # the area of a note at the top of the vault or in an ignored folder; empty: the vault name in lower case
    area_default: str = ""
    # without a project marker, the folder at this level (1 = the first folder) names the project
    project_depth: int = 2
    # a note with fewer folders than project_depth gets its first folder as the project, else "none"
    project_fallback: bool = True
    # folder paths whose child folders are projects, for example "Clients/Projects"
    project_parents: list[str] = field(default_factory=list)
    # first-level folders that give no area and no project, for example "docs"
    ignore_folders: list[str] = field(default_factory=list)
    # ticket prefixes found only in this vault, in addition to Config.jira_prefixes
    jira_prefixes: list[str] = field(default_factory=list)


def _kinds() -> dict[str, str]:
    """The kind and its standard: the Backstage kinds, then the local kinds."""
    backstage = ["domain", "system", "component", "resource", "api", "group", "person"]
    local = ["client", "environment", "vendor", "feature", "process"]
    return {**dict.fromkeys(backstage, "backstage"), **dict.fromkeys(local, "local")}


@dataclass
class DocType:
    """A closed document type. class_ is `state`, `record` or `none`; one_per is the identity key."""

    class_: str
    one_per: str
    required_sections: list[str] = field(default_factory=list)
    review_months: int | None = None
    standard: str = "local"


@dataclass
class RelationType:
    """A closed relation type. An empty list of kinds allows every kind."""

    src_kinds: list[str] = field(default_factory=list)
    dst_kinds: list[str] = field(default_factory=list)
    inverse: str = ""
    standard: str = "local"


@dataclass
class AttributeDef:
    """An attribute of an entity kind. A live attribute holds no value: it names its system of record and skill."""

    value_type: str = "string"
    live: bool = False
    system: str = ""
    skill: str = ""
    scoped: bool = False


def _doc_types() -> dict[str, DocType]:
    types = {}
    for names, class_, one_per, standard in (
        (["how-to", "tutorial", "reference", "explanation"], "state", "subject+task", "Diataxis"),
        (["entity"], "state", "entity", "Backstage entity descriptor"),
        (["drivers", "summary", "spec", "plan"], "state", "project", "local"),
        (["decision"], "record", "none", "MADR"),
        (["incident"], "record", "none", "KCS"),
        (["research", "measurement", "meeting", "review", "brainstorm", "log"], "record", "none", "local"),
        (["guide", "note", "template"], "none", "none", "legacy"),
    ):
        types.update({name: DocType(class_, one_per, standard=standard) for name in names})
    return types


def _relation_types() -> dict[str, RelationType]:
    backstage = "backstage"
    return {
        "partOf": RelationType([], [], "hasPart", backstage),
        "dependsOn": RelationType([], [], "dependencyOf", backstage),
        "providesApi": RelationType([], [], "apiProvidedBy", backstage),
        "consumesApi": RelationType([], [], "apiConsumedBy", backstage),
        "requests": RelationType(["doc", "client"], ["feature"], "requestedBy"),
        "memberOf": RelationType(["person"], ["group"], "hasMember"),
    }


@dataclass
class AuthorityConfig:
    """The factors that multiply the RRF score of a document in the rank stage."""

    canonical: float = 1.5  # the canonical document of a subject in the query
    review_overdue: float = 0.8  # a state document with review_by in the past
    archived: float = 0.5
    superseded: float = 0.5  # a record with superseded_by
    duplicate_identity: float = 0.5  # the second document of one identity key


@dataclass
class KbConfig:
    """The closed vocabularies of the knowledge base and the read-path limit."""

    # at or below this count of eligible units the search scans exactly
    exact_limit: int = 25000
    kinds: dict[str, str] = field(default_factory=_kinds)  # kind -> standard
    doc_types: dict[str, DocType] = field(default_factory=_doc_types)
    relation_types: dict[str, RelationType] = field(default_factory=_relation_types)
    attributes: dict[str, dict[str, AttributeDef]] = field(default_factory=dict)  # kind -> name -> definition
    authority: AuthorityConfig = field(default_factory=AuthorityConfig)


@dataclass
class TraceConfig:
    retention_days: int = 90


@dataclass
class W0Config:
    """The admission evaluator."""

    summary_chars: int = 2000
    private_paths: list[str] = field(default_factory=list)
    # homes whose sessions `w0 backfill` skips (the agent evaluation writes transcripts there); the current
    # home is skipped too when RAGNAROK_HOME is set. A relative path is relative to the home.
    skip_homes: list[str] = field(default_factory=list)
    # a false reject costs this many false admits; the threshold is 1 / (1 + cost_ratio)
    cost_ratio: float = 10
    # the uncertain band of the score; the judge sees a step inside it
    band: list[float] = field(default_factory=lambda: [0.03, 0.30])


@dataclass
class Config:
    root: Path = field(default_factory=lambda: Path("~/Documents/Vaults").expanduser())
    vaults: list[str] = field(default_factory=list)
    # the folder of a vault outside root; a vault with no entry is the folder root/<vault>
    vault_paths: dict[str, Path] = field(default_factory=dict)
    skip_dirs: set[str] = field(default_factory=lambda: {"node_modules"})
    private_patterns: list[str] = field(default_factory=list)
    allow_private: bool = False
    # a folder that holds one of these files is a project folder
    project_markers: list[str] = field(default_factory=lambda: ["DRIVERS.md", "SUMMARY.md"])
    # notes in a folder with this name get status archived
    archive_folder: str = "_archive"
    # ticket prefixes (the ABC of ABC-123) found in all vaults; no prefix means no ticket keys
    jira_prefixes: list[str] = field(default_factory=list)
    vault_rules: dict[str, VaultRules] = field(default_factory=dict)
    # notes the evaluation ignores, for example the notes that quote the evaluation questions
    eval_exclude: list[str] = field(default_factory=list)
    kb: KbConfig = field(default_factory=KbConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    w0: W0Config = field(default_factory=W0Config)
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "embeddinggemma-2:270m"
    query_prefix: str = "task: search result | query: "
    doc_prefix: str = "title: {title} | text: "
    dimension: int = 768
    min_words: int = 80
    max_words: int = 300
    # data_dir holds index.db, backups/ and proposals/; without index_path the index is data_dir/index.db
    data_dir: Path | None = None
    index_path: Path | None = None

    def __post_init__(self):
        if self.index_path is None:
            self.data_dir = self.data_dir or home_dir() / "data"
            self.index_path = self.data_dir / "index.db"
        elif self.data_dir is None:
            self.data_dir = self.index_path.parent

    @property
    def core_path(self) -> Path:
        """The file that holds the data that survives a rebuild: next to index_path, else data_dir/core.db."""
        return self.index_path.parent / "core.db"

    def vault_dir(self, vault: str) -> Path:
        return self.vault_paths.get(vault) or self.root / vault

    def note_path(self, key: str) -> Path:
        """The file of a note key `<vault>/<path in the vault>`."""
        vault, _, rel = key.partition("/")
        return self.vault_dir(vault) / rel

    def rules(self, vault: str) -> VaultRules:
        return self.vault_rules.get(vault) or VaultRules()

    def skip_dir(self, name: str) -> bool:
        """Folders the walker never enters: dot folders (.obsidian, .claude, .git, .trash) and skip_dirs."""
        return name.startswith(".") or name in self.skip_dirs


# the values that give the metadata, the sensitivity and the set of indexed notes; a change needs no new vectors
RULE_KEYS = ("root", "vaults", "skip_dirs", "private_patterns", "project_markers", "archive_folder", "jira_prefixes")


def _vocabularies(kb: KbConfig) -> dict:
    return {
        "kinds": kb.kinds,
        "doc_types": {name: asdict(t) for name, t in kb.doc_types.items()},
        "relation_types": {name: asdict(t) for name, t in kb.relation_types.items()},
        "attributes": {k: {n: asdict(a) for n, a in attrs.items()} for k, attrs in kb.attributes.items()},
    }


def rules_hash(cfg: Config) -> str:
    """A stable hash of the configuration values that give the metadata of the notes."""
    values = {key: getattr(cfg, key) for key in RULE_KEYS}
    values["root"] = str(cfg.root)
    values["skip_dirs"] = sorted(cfg.skip_dirs)
    values["vault_rules"] = {vault: asdict(rules) for vault, rules in sorted(cfg.vault_rules.items())}
    if cfg.vault_paths:  # only when set, so that an index made before vault_paths keeps its hash
        values["vault_paths"] = {vault: str(path) for vault, path in sorted(cfg.vault_paths.items())}
    vocabularies = _vocabularies(cfg.kb)
    if vocabularies != _vocabularies(KbConfig()):  # only when changed, so that the default keeps its hash
        values["kb"] = vocabularies
    return hashlib.sha1(json.dumps(values, sort_keys=True).encode()).hexdigest()


def config_path() -> Path:
    return home_dir() / "config.toml"


def _load_kb(table: dict) -> KbConfig:
    """The [kb] table. A list table (kinds, doc_types, relation_types, attributes) replaces its default."""
    values = {}
    for key, value in table.items():
        if key == "doc_types":
            value = {
                name: DocType(**{("class_" if k == "class" else k): v for k, v in t.items()})
                for name, t in value.items()
            }
        elif key == "relation_types":
            value = {name: RelationType(**t) for name, t in value.items()}
        elif key == "attributes":
            value = {kind: {name: AttributeDef(**a) for name, a in attrs.items()} for kind, attrs in value.items()}
        elif key == "authority":
            value = AuthorityConfig(**value)
        values[key] = value
    return KbConfig(**values)


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    values = {}
    for key, value in tomllib.loads(path.read_text()).items():
        if key not in Config.__dataclass_fields__:
            raise ValueError(f"unknown config key in {path}: {key}")
        if key in PATH_KEYS:
            value = path.parent / Path(value).expanduser()
        elif key == "skip_dirs":
            value = set(value)
        elif key == "vault_paths":
            value = {vault: path.parent / Path(folder).expanduser() for vault, folder in value.items()}
        elif key == "vault_rules":
            value = {vault: VaultRules(**rules) for vault, rules in value.items()}
        elif key == "kb":
            value = _load_kb(value)
        elif key == "trace":
            value = TraceConfig(**value)
        elif key == "w0":
            value = W0Config(**value)
        values[key] = value
    return Config(**values)


def dump_config(cfg: Config, **paths: Path | str) -> str:
    """The TOML text of cfg, with the path keys in paths in place of the values of cfg."""
    lines = [f"{key} = {json.dumps(str(value))}" for key, value in paths.items()]
    for f in fields(Config):
        value = getattr(cfg, f.name)
        if f.name in PATH_KEYS or f.name in ("vault_paths", "vault_rules", "kb", "trace", "w0") or f.name in paths:
            continue
        lines.append(f"{f.name} = {json.dumps(sorted(value) if isinstance(value, set) else value)}")
    if cfg.vault_paths:
        lines.append("\n[vault_paths]")
        lines += [f"{json.dumps(vault)} = {json.dumps(str(path))}" for vault, path in cfg.vault_paths.items()]
    for vault, rules in cfg.vault_rules.items():
        lines.append(f"\n[vault_rules.{json.dumps(vault)}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in asdict(rules).items()]
    lines += _dump_kb(cfg.kb)
    for name, table in (("trace", cfg.trace), ("w0", cfg.w0)):
        lines.append(f"\n[{name}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in asdict(table).items()]
    return "\n".join(lines) + "\n"


def _dump_kb(kb: KbConfig) -> list[str]:
    lines = ["\n[kb]", f"exact_limit = {kb.exact_limit}", "\n[kb.kinds]"]
    lines += [f"{json.dumps(kind)} = {json.dumps(standard)}" for kind, standard in kb.kinds.items()]
    for name, doc_type in kb.doc_types.items():
        lines.append(f"\n[kb.doc_types.{json.dumps(name)}]")
        lines += [
            f"{'class' if k == 'class_' else k} = {json.dumps(v)}" for k, v in asdict(doc_type).items() if v is not None
        ]
    for name, relation in kb.relation_types.items():
        lines.append(f"\n[kb.relation_types.{json.dumps(name)}]")
        lines += [f"{k} = {json.dumps(v)}" for k, v in asdict(relation).items()]
    for kind, attributes in kb.attributes.items():
        for name, attribute in attributes.items():
            lines.append(f"\n[kb.attributes.{json.dumps(kind)}.{json.dumps(name)}]")
            lines += [f"{k} = {json.dumps(v)}" for k, v in asdict(attribute).items()]
    lines.append("\n[kb.authority]")
    lines += [f"{k} = {json.dumps(v)}" for k, v in asdict(kb.authority).items()]
    return lines
