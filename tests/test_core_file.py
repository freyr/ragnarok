import sqlite3
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import sqlite_vec
from test_index import FakeEmbedder

from ragnarok import agent_eval, cli, embed, index
from ragnarok.config import Config, dump_config
from ragnarok.embed import input_hash
from ragnarok.index import connect, import_cache, update
from ragnarok.metadata import iter_notes
from ragnarok.splitter import chunk_file


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
    conn.execute("INSERT INTO embed_cache VALUES ('b', 'm', ?)", (blob(9),))
    conn.commit()
    conn.close()
    assert import_cache(vault, tmp_path / "v1.db") == 1  # 'b' exists and stays
    conn = connect(vault)
    rows = dict(conn.execute("SELECT text_hash, vector FROM embed_cache"))
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


def test_prepare_copies_both_files_into_the_evaluation_home(vault, monkeypatch):
    vault.dimension = 8
    embedders = fake_embedders(monkeypatch)
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    cached = conn.execute("SELECT count(*) FROM embed_cache").fetchone()[0]
    conn.execute(W0_ROW, ("s",))
    conn.commit()
    conn.close()
    ecfg = agent_eval.prepare(vault)
    assert embedders and all(e.calls == 0 for e in embedders)
    conn = connect(ecfg)
    assert conn.execute("SELECT count(*) FROM embed_cache").fetchone()[0] >= cached > 0
    assert conn.execute("SELECT count(*) FROM w0_step").fetchone()[0] == 1
    paths = {p for (p,) in conn.execute("SELECT path FROM doc")}
    assert paths == set(agent_eval.mirror_notes(vault))
    units = conn.execute("SELECT count(*) FROM unit").fetchone()[0]
    assert units == conn.execute("SELECT count(*) FROM unit_emb").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM unit WHERE doc_id NOT IN (SELECT id FROM doc)").fetchone()[0] == 0
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


def write_config(vault, tmp_path):
    vault.dimension = 8
    config_file = tmp_path / "config.toml"
    config_file.write_text(dump_config(vault, root=vault.root, index_path=vault.index_path))
    return config_file


def fake_embedders(monkeypatch):
    embedders = []

    def make(cfg):
        embedders.append(FakeEmbedder())
        return embedders[-1]

    monkeypatch.setattr(embed, "OllamaEmbedder", make)
    return embedders


def test_connect_sets_foreign_keys(vault):
    vault.dimension = 8
    conn = connect(vault)
    assert conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
    files = {name: Path(file).resolve() for _, name, file in conn.execute("PRAGMA database_list")}
    assert files == {"main": vault.core_path.resolve(), "idx": vault.index_path.resolve()}
    conn.close()


W0_ROW = (
    "INSERT INTO w0_step (created, session_id, step_kind, summary_hash) VALUES ('2026-11-05T10:00:00Z', ?, 'x', 'h')"
)


def test_rebuild_keeps_core_and_w0_step(vault, tmp_path, monkeypatch):
    config_file = write_config(vault, tmp_path)
    embedders = fake_embedders(monkeypatch)
    argv = ["--config", str(config_file), "index"]
    assert cli.main(argv) == 0
    conn = connect(vault)
    conn.execute(W0_ROW, ("s-1",))
    conn.commit()
    conn.close()
    inode = vault.core_path.stat().st_ino
    assert cli.main([*argv, "--rebuild"]) == 0
    assert embedders[1].calls == 0
    assert vault.core_path.stat().st_ino == inode
    conn = connect(vault)
    assert conn.execute("SELECT session_id FROM w0_step").fetchall() == [("s-1",)]
    assert conn.execute("SELECT count(*) FROM doc").fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM unit").fetchone()[0] > 0
    conn.close()


def test_rebuild_accepts_the_layout_of_the_earlier_tasks(vault, tmp_path, monkeypatch, capsys):
    # tasks 2a to 4: index.db held the derived tables, core.db only meta and embed_cache
    config_file = write_config(vault, tmp_path)
    embedders = fake_embedders(monkeypatch)
    argv = ["--config", str(config_file), "index"]
    assert cli.main(argv) == 0
    core = sqlite3.connect(vault.core_path)
    for (name,) in core.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall():
        if name != "embed_cache":
            core.execute(f"DROP TABLE {name}")
    core.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    core.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    core.commit()
    core.close()
    for suffix in ("", "-wal", "-shm"):
        vault.index_path.with_name(vault.index_path.name + suffix).unlink(missing_ok=True)
    old = sqlite3.connect(vault.index_path)
    old.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    rows = [("model", vault.model), ("dimension", "8"), ("schema_version", "3")]
    old.executemany("INSERT INTO meta VALUES (?, ?)", rows)
    old.execute("CREATE TABLE doc (id TEXT PRIMARY KEY)")
    old.commit()
    old.close()
    assert cli.main(argv) == 1
    assert "Run `ragnarok index --rebuild`" in capsys.readouterr().err
    assert cli.main([*argv, "--rebuild"]) == 0
    assert embedders[-1].calls == 0
    conn = connect(vault)
    assert conn.execute("SELECT count(*) FROM unit_emb").fetchone()[0] > 0
    conn.close()


# the DDL of schema version 1 (the main branch before the two files), with dimension 8
V1_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    vault TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    project TEXT NOT NULL,
    area TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    description TEXT,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_project ON notes (vault, project);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL REFERENCES notes (id),
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_note ON chunks (note_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text, tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[8] distance_metric=cosine,
    vault TEXT,
    sensitivity TEXT,
    status TEXT
);
CREATE TABLE IF NOT EXISTS embed_cache (text_hash TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS links (from_note INTEGER NOT NULL, target TEXT NOT NULL, to_note INTEGER);
CREATE INDEX IF NOT EXISTS links_from ON links (from_note);
CREATE INDEX IF NOT EXISTS links_to ON links (to_note);
CREATE TABLE IF NOT EXISTS jira (key TEXT NOT NULL, note_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS jira_key ON jira (key);
"""


def v1_index(cfg) -> tuple[int, int]:
    """A version 1 index file of the fixture vault, with a vector of each chunk in its embed_cache."""
    conn = sqlite3.connect(cfg.index_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.executescript(V1_DDL)
    conn.executemany(
        "INSERT INTO meta VALUES (?, ?)", [("model", cfg.model), ("dimension", "8"), ("schema_version", "1")]
    )
    emb = FakeEmbedder()
    notes = chunks = 0
    for rel in iter_notes(cfg):
        meta, parts, _ = chunk_file(cfg, rel)
        st = cfg.note_path(rel).stat()
        values = (meta.vault, rel, meta.title, meta.project, meta.area, meta.type, meta.status, meta.sensitivity)
        note_id = conn.execute(
            "INSERT INTO notes (vault, path, title, project, area, type, status, sensitivity, mtime_ns, size, hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'h')",
            (*values, st.st_mtime_ns, st.st_size),
        ).lastrowid
        notes += 1
        for c in parts:
            th = input_hash(cfg, c.heading, c.text)
            conn.execute(
                "INSERT INTO chunks (note_id, ord, heading, text, text_hash) VALUES (?, ?, ?, ?, ?)",
                (note_id, c.ord, c.heading, c.text, th),
            )
            vector = sqlite_vec.serialize_float32(emb.vector(c.heading + c.text))
            conn.execute("INSERT OR IGNORE INTO embed_cache VALUES (?, ?, ?)", (th, cfg.model, vector))
            chunks += 1
    conn.commit()
    conn.close()
    return notes, chunks


def test_migrates_a_v1_index_without_embedding(vault, tmp_path, monkeypatch):
    config_file = write_config(vault, tmp_path)
    embedders = fake_embedders(monkeypatch)
    notes, chunks = v1_index(vault)
    assert cli.main(["--config", str(config_file), "migrate"]) == 0
    backups = list((vault.data_dir / "backups").glob("index-v1-*.db"))
    assert len(backups) == 1
    copy = sqlite3.connect(backups[0])
    assert copy.execute("SELECT count(*) FROM notes").fetchone()[0] == notes
    copy.close()
    assert cli.main(["--config", str(config_file), "index"]) == 0
    assert embedders and all(e.calls == 0 for e in embedders)
    conn = connect(vault)
    assert conn.execute("SELECT count(*) FROM doc").fetchone()[0] == notes
    assert conn.execute("SELECT count(*) FROM unit").fetchone()[0] == chunks
    assert conn.execute("SELECT count(*) FROM unit_emb").fetchone()[0] == chunks
    assert conn.execute("SELECT count(*) FROM idx.sqlite_master WHERE name = 'notes'").fetchone()[0] == 0
    conn.close()


def test_rebuild_refuses_an_index_file_with_a_cache(vault, tmp_path, monkeypatch, capsys):
    config_file = write_config(vault, tmp_path)
    fake_embedders(monkeypatch)
    v1_index(vault)
    before = vault.index_path.read_bytes()
    assert cli.main(["--config", str(config_file), "index", "--rebuild"]) == 1
    assert "Run `ragnarok migrate`" in capsys.readouterr().err
    assert vault.index_path.read_bytes() == before


def test_backup_copies_core_while_a_connection_is_open(vault, tmp_path):
    config_file = write_config(vault, tmp_path)
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    conn.execute(W0_ROW, ("s",))
    conn.commit()
    cached = conn.execute("SELECT count(*) FROM embed_cache").fetchone()[0]
    target = tmp_path / "copy.db"
    assert cli.main(["--config", str(config_file), "backup", "--to", str(target)]) == 0
    assert cli.main(["--config", str(config_file), "backup"]) == 0
    conn.close()
    default = list((vault.data_dir / "backups").glob("core-*.db"))
    assert len(default) == 1
    for path in (target, default[0]):
        copy = sqlite3.connect(path)
        assert copy.execute("SELECT count(*) FROM embed_cache").fetchone()[0] == cached > 0
        assert copy.execute("SELECT count(*) FROM w0_step").fetchone()[0] == 1
        copy.close()


def files_bytes(path):
    return b"".join(p.read_bytes() for p in path.parent.glob(path.name + "*"))


def test_prepare_drops_the_text_of_excluded_and_private_notes_from_the_copies(vault, monkeypatch):
    vault.dimension = 8
    vault.eval_exclude = ["Home/Quiz/"]
    (vault.root / "Home/Quiz").mkdir(parents=True)
    (vault.root / "Home/Quiz/answers.md").write_text("# Answers\nThe answer is qzexcludedword.\n")
    (vault.root / "Home/Password/bank.md").write_text("# secret\nThe pin is qzprivateword.\n")
    fake_embedders(monkeypatch)
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    source = {p: ids for p, ids in conn.execute(UNIT_IDS)}
    conn.close()
    ecfg = agent_eval.prepare(vault)
    conn = connect(ecfg)
    copy = {p: ids for p, ids in conn.execute(UNIT_IDS)}
    conn.close()
    assert copy and copy == {p: ids for p, ids in source.items() if p in copy}
    assert set(copy) == set(agent_eval.mirror_notes(vault))
    for path in (ecfg.core_path, ecfg.index_path):
        data = files_bytes(path)
        assert b"qzexcludedword" not in data and b"qzprivateword" not in data


UNIT_IDS = "SELECT d.path, group_concat(u.id) FROM doc d JOIN unit u ON u.doc_id = d.id GROUP BY d.path"


def test_rebuild_refuses_an_index_file_that_it_cannot_read(vault, tmp_path, monkeypatch, capsys):
    config_file = write_config(vault, tmp_path)
    fake_embedders(monkeypatch)
    v1_index(vault)
    before = vault.index_path.read_bytes()
    holder = sqlite3.connect(vault.index_path)
    holder.execute("PRAGMA locking_mode = EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        assert cli.main(["--config", str(config_file), "index", "--rebuild"]) == 1
        err = capsys.readouterr().err
        assert "does not delete it" in err and "Move it to" in err
        assert vault.index_path.read_bytes() == before
    finally:
        holder.rollback()
        holder.close()


def test_migrate_keeps_the_old_file_when_the_cache_copy_is_short(vault, tmp_path, monkeypatch, capsys):
    config_file = write_config(vault, tmp_path)
    fake_embedders(monkeypatch)
    v1_index(vault)
    before = vault.index_path.read_bytes()
    monkeypatch.setattr(index, "import_cache", lambda cfg, source: 0)
    assert cli.main(["--config", str(config_file), "migrate"]) == 1
    assert "keeps" in capsys.readouterr().err
    assert vault.index_path.read_bytes() == before
