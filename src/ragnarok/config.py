"""Configuration: vault locations, exclusions, sensitive paths, path rules, model and splitter settings.

The home folder holds config.toml, data/ (index, backups, proposals) and logs/. It is the folder in
RAGNAROK_HOME, else the repo checkout, else ~/.ragnarok (an installed package, for example `uvx ragnarok-mcp`).
A relative path in config.toml is relative to the folder of the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

PATH_KEYS = ("root", "data_dir", "index_path")

log = logging.getLogger("ragnarok.config")


def default_home() -> Path:
    """The home folder of an installed package with no RAGNAROK_HOME."""
    return Path("~/.ragnarok").expanduser()


def is_checkout(folder: Path) -> bool:
    """True when the folder is a repo checkout of ragnarok."""
    pyproject = folder / "pyproject.toml"
    return pyproject.is_file() and re.search(r'^name = "ragnarok(-mcp)?"$', pyproject.read_text(), re.M) is not None


def home_dir() -> Path:
    """RAGNAROK_HOME, else the repo checkout that holds this source file, else ~/.ragnarok."""
    if env := os.environ.get("RAGNAROK_HOME"):
        return Path(env).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if is_checkout(checkout):
        return checkout
    return default_home()


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


@dataclass
class Corpus:
    """The vaults that a variant indexes: the folders root/<vault>."""

    root: Path
    vaults: list[str]


CHUNK_PREFIXES = ("none", "description")
NOTE_RANKS = ("maxp", "topk_sum", "note_vector")
# the keys of the default configuration that a variant can override
VARIANT_KEYS = (
    "model",
    "dimension",
    "query_prefix",
    "doc_template",
    "min_words",
    "max_words",
    "chunk_prefix",
    "note_rank",
)


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
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "embeddinggemma-2:270m"
    # the query prefix and the document template ("{title}" is the note title and the heading path); unset gives the
    # value of the model profile in embed.PROFILES
    query_prefix: str | None = None
    doc_template: str | None = None
    dimension: int = 768
    # how long Ollama keeps the model loaded after a call: seconds (-1 = until Ollama stops) or a duration like "30m"
    keep_alive: int | str = -1
    min_words: int = 80
    max_words: int = 300
    # "description": the description of the note goes before the text of each body chunk in the text to the model
    chunk_prefix: str = "none"
    # how the chunks of a note give its rank: "maxp" (the best chunk), "topk_sum" (the sum of the scores of the best
    # TOPK chunks in each list) or "note_vector" (also the mean chunk vector of each note as a third list)
    note_rank: str = "maxp"
    # bench: settings that override the keys in VARIANT_KEYS, and sets of vaults to index with a variant
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)
    corpora: dict[str, Corpus] = field(default_factory=dict)
    # data_dir holds index.db, backups/ and proposals/; without index_path the index is data_dir/index.db
    # Both may be None in the constructor; __post_init__ always sets them, so the type is Path.
    data_dir: Path = None  # type: ignore[assignment]
    index_path: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Ollama takes seconds as an integer or a duration with a unit ("30m"); it refuses "-1", a bool or a float.
        ok_int = isinstance(self.keep_alive, int) and not isinstance(self.keep_alive, bool)
        ok_str = isinstance(self.keep_alive, str) and not self.keep_alive.lstrip("-").isdigit()
        if not (ok_int or ok_str):
            raise ValueError(
                f"keep_alive must be an integer of seconds or a duration such as '30m', not {self.keep_alive!r}"
            )
        if self.chunk_prefix not in CHUNK_PREFIXES:
            raise ValueError(f"chunk_prefix must be one of {', '.join(CHUNK_PREFIXES)}, not {self.chunk_prefix!r}")
        if self.note_rank not in NOTE_RANKS:
            raise ValueError(f"note_rank must be one of {', '.join(NOTE_RANKS)}, not {self.note_rank!r}")
        for name, values in self.variants.items():
            if unknown := set(values) - set(VARIANT_KEYS):
                raise ValueError(f"variant {name}: unknown keys {', '.join(sorted(unknown))}")
            replace(self, **values, variants={})  # the checks above for the values of the variant
        if self.index_path is None:
            self.data_dir = self.data_dir or home_dir() / "data"
            self.index_path = self.data_dir / "index.db"
        elif self.data_dir is None:
            self.data_dir = self.index_path.parent

    def variant(self, name: str, corpus: str) -> Config:
        """The configuration of a variant on a corpus: its own index file data_dir/variants/<name>/<corpus>/index.db,
        and the same data_dir, so the same embedding cache."""
        if name not in self.variants:
            raise KeyError(f"no variant {name!r} in the configuration; the variants are {', '.join(self.variants)}")
        if corpus not in self.corpora:
            raise KeyError(f"no corpus {corpus!r} in the configuration; the corpora are {', '.join(self.corpora)}")
        c = self.corpora[corpus]
        return replace(
            self,
            **self.variants[name],
            root=c.root,
            vaults=list(c.vaults),
            vault_paths={},
            index_path=self.data_dir / "variants" / name / corpus / "index.db",
        )

    def note_key(self, raw: str) -> str:
        """The key `<vault>/<path in the vault>` of a path that is relative to the vault root or absolute."""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return path.as_posix()
        path = path.resolve()
        for vault in self.vaults:
            if path.is_relative_to(base := self.vault_dir(vault).resolve()):
                return f"{vault}/{path.relative_to(base).as_posix()}"
        return path.relative_to(self.root.resolve()).as_posix()

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


def rules_hash(cfg: Config) -> str:
    """A stable hash of the configuration values that give the metadata of the notes."""
    values = {key: getattr(cfg, key) for key in RULE_KEYS}
    values["root"] = str(cfg.root)
    values["skip_dirs"] = sorted(cfg.skip_dirs)
    values["vault_rules"] = {vault: asdict(rules) for vault, rules in sorted(cfg.vault_rules.items())}
    if cfg.vault_paths:  # only when set, so that an index made before vault_paths keeps its hash
        values["vault_paths"] = {vault: str(path) for vault, path in sorted(cfg.vault_paths.items())}
    return hashlib.sha1(json.dumps(values, sort_keys=True).encode(), usedforsecurity=False).hexdigest()


def config_path() -> Path:
    return home_dir() / "config.toml"


def load_config(path: Path | None = None) -> Config:
    path = path or config_path()
    if not path.exists():
        return Config()
    values = {}
    for key, value in tomllib.loads(path.read_text()).items():
        if key == "write":
            log.warning("%s: ragnarok does not use the [write] table any more; remove it", path)
            continue
        if key == "doc_prefix":
            log.warning("%s: the key doc_prefix is now doc_template; rename it", path)
            key = "doc_template"
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
        elif key == "variants":
            value = {name: dict(values) for name, values in value.items()}
        elif key == "corpora":
            value = {name: _corpus(path, name, values) for name, values in value.items()}
        values[key] = value
    return Config(**values)


def _corpus(path: Path, name: str, values: dict[str, Any]) -> Corpus:
    if set(values) != {"root", "vaults"}:
        raise ValueError(f"corpus {name} in {path} needs the keys root and vaults, and no other key")
    return Corpus(path.parent / Path(values["root"]).expanduser(), list(values["vaults"]))


def variant_hash(cfg: Config) -> str:
    """A stable hash of the settings that a variant can override, with the template of the model profile."""
    from .embed import profile

    p = profile(cfg)
    values = {key: getattr(cfg, key) for key in VARIANT_KEYS}
    values.update(query_prefix=p.query_prefix, doc_template=p.doc_template)
    return hashlib.sha1(json.dumps(values, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:12]


def dump_config(cfg: Config, **paths: Path | str) -> str:
    """The TOML text of cfg, with the path keys in paths in place of the values of cfg."""
    lines = [f"{key} = {json.dumps(str(value))}" for key, value in paths.items()]
    for f in fields(Config):
        value = getattr(cfg, f.name)
        skip = ("vault_paths", "vault_rules", "variants", "corpora")
        if f.name in PATH_KEYS or f.name in skip or f.name in paths or value is None:
            continue
        lines.append(f"{f.name} = {json.dumps(sorted(value) if isinstance(value, set) else value)}")
    if cfg.vault_paths:
        lines.append("\n[vault_paths]")
        lines += [f"{json.dumps(vault)} = {json.dumps(str(path))}" for vault, path in cfg.vault_paths.items()]
    for vault, rules in cfg.vault_rules.items():
        lines.append(f"\n[vault_rules.{json.dumps(vault)}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in asdict(rules).items()]
    return "\n".join(lines) + "\n"
