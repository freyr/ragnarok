"""Configuration: vault locations, exclusions, sensitive paths, path rules, model and splitter settings.

The home folder holds config.toml, data/ (index, backups, proposals) and logs/. It is the repo checkout,
or the folder in RAGNAROK_HOME. A relative path in config.toml is relative to the folder of the file.
"""

from __future__ import annotations

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


@dataclass
class Config:
    root: Path = field(default_factory=lambda: Path("~/Documents/Vaults").expanduser())
    vaults: list[str] = field(default_factory=list)
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

    def rules(self, vault: str) -> VaultRules:
        return self.vault_rules.get(vault) or VaultRules()

    def skip_dir(self, name: str) -> bool:
        """Folders the walker never enters: dot folders (.obsidian, .claude, .git, .trash) and skip_dirs."""
        return name.startswith(".") or name in self.skip_dirs


def config_path() -> Path:
    return home_dir() / "config.toml"


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
        elif key == "vault_rules":
            value = {vault: VaultRules(**rules) for vault, rules in value.items()}
        values[key] = value
    return Config(**values)


def dump_config(cfg: Config, **paths: Path | str) -> str:
    """The TOML text of cfg, with the path keys in paths in place of the values of cfg."""
    lines = [f"{key} = {json.dumps(str(value))}" for key, value in paths.items()]
    for f in fields(Config):
        value = getattr(cfg, f.name)
        if f.name in PATH_KEYS or f.name == "vault_rules" or f.name in paths:
            continue
        lines.append(f"{f.name} = {json.dumps(sorted(value) if isinstance(value, set) else value)}")
    for vault, rules in cfg.vault_rules.items():
        lines.append(f"\n[vault_rules.{json.dumps(vault)}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in asdict(rules).items()]
    return "\n".join(lines) + "\n"
