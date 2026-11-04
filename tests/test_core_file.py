import sqlite3
import tomllib
from dataclasses import replace

import pytest
import sqlite_vec
from test_index import FakeEmbedder

from ragnarok import agent_eval, cli, embed
from ragnarok.config import Config, dump_config
from ragnarok.index import connect, import_cache, update


def test_core_path_follows_data_dir_or_index_path(tmp_path):
    assert Config(data_dir=tmp_path / "d").core_path == tmp_path / "d" / "core.db"
    assert Config(index_path=tmp_path / "x" / "other.db").core_path == tmp_path / "x" / "core.db"


def test_rebuild_keeps_the_embedding_cache(vault, tmp_path, monkeypatch):
    vault.dimension = 8
    config_file = tmp_path / "config.toml"
    config_file.write_text(dump_config(vault, root=vault.root, index_path=vault.index_path))
    embedders = []

    def make(cfg):
        embedders.append(FakeEmbedder())
        return embedders[-1]

    monkeypatch.setattr(embed, "OllamaEmbedder", make)
    argv = ["--config", str(config_file), "index"]
    assert cli.main(argv) == 0
    assert embedders[0].calls > 0
    assert vault.core_path.exists()
    assert cli.main([*argv, "--rebuild"]) == 0
    assert embedders[1].calls == 0
    assert vault.core_path.exists() and vault.index_path.exists()


def v1_file(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE embed_cache (text_hash TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)")
    conn.executemany("INSERT INTO embed_cache VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def blob(x):
    return sqlite_vec.serialize_float32([x] * 8)


def test_cache_import_copies_the_v1_rows(vault, tmp_path):
    vault.dimension = 8
    v1_file(tmp_path / "v1.db", [("a", "m", blob(1)), ("b", "m", blob(2))])
    conn = connect(vault)
    conn.execute("INSERT INTO core.embed_cache VALUES ('b', 'm', ?)", (blob(9),))
    conn.commit()
    conn.close()
    assert import_cache(vault, tmp_path / "v1.db") == 1  # 'b' exists and stays
    conn = connect(vault)
    rows = dict(conn.execute("SELECT text_hash, vector FROM core.embed_cache"))
    assert rows == {"a": blob(1), "b": blob(9)}
    conn.close()


def test_cache_import_command_prints_the_count(vault, tmp_path, capsys):
    v1_file(tmp_path / "v1.db", [("a", "m", blob(1))])
    vault.dimension = 8
    config_file = tmp_path / "config.toml"
    config_file.write_text(dump_config(vault, root=vault.root, index_path=vault.index_path))
    assert cli.main(["--config", str(config_file), "cache", "import", "--from", str(tmp_path / "v1.db")]) == 0
    assert "imported 1" in capsys.readouterr().out


def test_prepare_reuse_mirror_keeps_an_existing_mirror(vault, monkeypatch):
    vault.dimension = 8
    monkeypatch.setattr(embed, "OllamaEmbedder", lambda cfg: FakeEmbedder())
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    conn.close()
    ecfg = agent_eval.prepare(vault)
    marker = ecfg.root / "marker.md"
    marker.write_text("# kept\n")
    agent_eval.prepare(vault, reuse_mirror=True)
    assert marker.exists()
    agent_eval.prepare(vault)
    assert not marker.exists()


def test_prepare_copies_the_cache_into_the_evaluation_home(vault, monkeypatch):
    vault.dimension = 8
    monkeypatch.setattr(embed, "OllamaEmbedder", lambda cfg: FakeEmbedder())
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    cached = conn.execute("SELECT count(*) FROM core.embed_cache").fetchone()[0]
    conn.close()
    ecfg = agent_eval.prepare(vault)
    conn = connect(ecfg)
    assert conn.execute("SELECT count(*) FROM core.embed_cache").fetchone()[0] >= cached > 0
    conn.close()


def test_mirror_hash_changes_when_a_file_changes(tmp_path):
    mirror = tmp_path / "vaults"
    (mirror / "A").mkdir(parents=True)
    (mirror / "A" / "x.md").write_text("one")
    first = agent_eval.mirror_hash(mirror)
    assert agent_eval.mirror_hash(mirror) == first
    (mirror / "A" / "x.md").write_text("two")
    changed = agent_eval.mirror_hash(mirror)
    assert changed != first
    (mirror / "A" / "x.md").rename(mirror / "A" / "y.md")
    assert agent_eval.mirror_hash(mirror) not in (first, changed)


def test_baseline_config_has_only_v1_keys(vault, tmp_path):
    vault.eval_exclude = ["Home/Old/"]
    mirror = tmp_path / "bet" / "vaults"
    home = tmp_path / "baseline"
    vault = replace(vault, eval_exclude=["Area/project/"])
    agent_eval.write_baseline_home(vault, home, mirror)
    values = tomllib.loads((home / "config.toml").read_text())
    v1 = {"root", "vaults", "vault_paths", "skip_dirs", "private_patterns", "allow_private", "project_markers"}
    v1 |= {"archive_folder", "jira_prefixes", "vault_rules", "eval_exclude", "ollama_url", "model", "query_prefix"}
    v1 |= {"doc_prefix", "dimension", "min_words", "max_words", "data_dir", "index_path"}
    assert set(values) <= v1
    assert values["root"] == str(mirror) and mirror.is_absolute()
    assert values["eval_exclude"] == ["Area/project/"]
    assert values["vaults"] == vault.vaults
    assert set(values["vault_rules"]) == set(vault.vault_rules)
    assert values["data_dir"] == "data"


def test_mirror_hash_fails_without_a_mirror(tmp_path):
    with pytest.raises(FileNotFoundError):
        agent_eval.mirror_hash(tmp_path / "missing")
