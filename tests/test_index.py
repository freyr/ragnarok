import dataclasses
import hashlib
import json
import os
import sqlite3

import pytest
import sqlite_vec

from ragnarok import index
from ragnarok.config import Config, VaultRules, rules_hash
from ragnarok.index import IndexMismatch, connect, status, update


class FakeEmbedder:
    """Deterministic vectors from the text hash; counts the texts it embeds."""

    def __init__(self, dimension=8):
        self.dimension = dimension
        self.calls = 0

    def embed_docs(self, items):
        self.calls += len(items)
        return [self.vector(h + t) for h, t in items]

    def vector(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255 + 0.01 for b in digest[: self.dimension]]


@pytest.fixture
def cfg(vault):
    vault.dimension = 8
    return vault


def build(cfg):
    embedder = FakeEmbedder()
    conn = connect(cfg)
    return conn, embedder, update(cfg, conn, embedder)


def count(conn, sql):
    return conn.execute(sql).fetchone()[0]


# the schema of release 1.1.0, as its index.py made it (schema version 1)
SCHEMA_1_1_0 = """
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

# what release 1.2.0 added to an index of schema version 1 (_add_kb_columns and _add_kb_tables), with no version change
ADDED_1_2_0 = """
ALTER TABLE notes ADD COLUMN note_class TEXT;
ALTER TABLE notes ADD COLUMN validity TEXT;
ALTER TABLE notes ADD COLUMN as_of TEXT;
ALTER TABLE notes ADD COLUMN superseded_by TEXT;
ALTER TABLE notes ADD COLUMN subject TEXT;
CREATE TABLE IF NOT EXISTS entities (note_id INTEGER PRIMARY KEY, kind TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases (note_id INTEGER NOT NULL, alias TEXT NOT NULL, norm TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS entity_aliases_note ON entity_aliases (note_id);
CREATE TABLE IF NOT EXISTS refs (note_id INTEGER NOT NULL, key TEXT NOT NULL, target TEXT NOT NULL, to_note INTEGER);
CREATE INDEX IF NOT EXISTS refs_note ON refs (note_id);
CREATE INDEX IF NOT EXISTS refs_to ON refs (to_note, key);
UPDATE notes SET note_class = 'state', validity = 'current', subject = 'pump' WHERE path = 'Home/Heating/SUMMARY.md';
UPDATE notes SET note_class = 'record' WHERE path LIKE '%/research-%';
INSERT INTO entities (note_id, kind) SELECT id, 'device' FROM notes WHERE path = 'Home/Heating/notes/boiler.md';
INSERT INTO entity_aliases (note_id, alias, norm) SELECT id, 'Pump', 'pump' FROM notes
    WHERE path = 'Home/Heating/notes/boiler.md';
INSERT INTO refs (note_id, key, target, to_note) SELECT s.id, 'subject', 'pump', b.id FROM notes s, notes b
    WHERE s.path = 'Home/Heating/SUMMARY.md' AND b.path = 'Home/Heating/notes/boiler.md';
"""


def tables(conn) -> dict[str, list]:
    """The columns of each table and the names of the indexes."""
    names = [
        n
        for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        if n != "sqlite_sequence"
    ]
    columns = {n: [r[1:] for r in conn.execute(f"PRAGMA table_info('{n}')")] for n in names}
    columns["indexes"] = [n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' ORDER BY 1")]
    return columns


def raw(path=":memory:"):
    """A plain connection that can read the vector table."""
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    return conn


def schema_1_1_0():
    conn = raw()
    conn.executescript(SCHEMA_1_1_0)
    return tables(conn)


def cache_file(cfg):
    """The shared embedding cache of the home: data_dir/embed-cache.db."""
    return cfg.data_dir / "embed-cache.db"


def cache_rows(cfg) -> list[tuple]:
    conn = sqlite3.connect(cache_file(cfg))
    try:
        return conn.execute("SELECT model, dimension, template, text_hash FROM embed_cache").fetchall()
    finally:
        conn.close()


def old_index(cfg, release):
    """An index with the notes of the vault and the embedding cache in the index file, as release 1.1.0, 1.2.0
    (schema version 1) or 2.0.0 (schema version 2) left it."""
    conn, emb, built = build(cfg)
    conn.close()
    conn = raw(cfg.index_path)
    if cache_file(cfg).exists():
        # the cache key of the old releases: the hash of the model and of the text that goes to the model
        conn.execute("ATTACH DATABASE ? AS cache", (str(cache_file(cfg)),))
        conn.execute("CREATE TABLE embed_cache (text_hash TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)")
        conn.execute("INSERT INTO main.embed_cache SELECT text_hash, model, vector FROM cache.embed_cache")
        conn.commit()
        conn.execute("DETACH DATABASE cache")
        for suffix in ("", "-wal", "-shm"):
            cache_file(cfg).with_name(cache_file(cfg).name + suffix).unlink(missing_ok=True)
    if release == "1.2.0":
        conn.executescript(ADDED_1_2_0)
    version = "2" if release == "2.0.0" else "1"
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (version,))
    conn.commit()
    conn.close()
    return emb, built


def run_index(cfg, monkeypatch, capsys, *args) -> dict:
    """`ragnarok index` with the fake embedder; the JSON line that it prints."""
    from ragnarok import cli, embed

    monkeypatch.setattr(embed, "OllamaEmbedder", lambda cfg: FakeEmbedder())
    assert cli.cmd_index(cfg, cli.build_parser().parse_args(["index", *args])) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("release", ["1.1.0", "1.2.0", "2.0.0"])
def test_connect_migrates_an_older_index_and_its_cache_with_no_embedding(cfg, release, monkeypatch, capsys):
    _, built = old_index(cfg, release)
    conn = connect(cfg)
    assert conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone() == ("3",)
    without_cache = schema_1_1_0()
    del without_cache["embed_cache"]
    without_cache["indexes"].remove("sqlite_autoindex_embed_cache_1")
    assert tables(conn) == without_cache
    # a 1.x index derives the metadata of all notes again on the next update; a 2.0.0 index keeps its rules hash
    assert status(cfg, conn)["rules_hash"] == (rules_hash(cfg) if release == "2.0.0" else None)
    conn.close()
    # each old entry gets the key of the current model, dimension and template
    rows = cache_rows(cfg)
    assert len(rows) == built.embedded
    assert {r[:3] for r in rows} == {(cfg.model, cfg.dimension, "title: {title} | text: ")}
    result = run_index(cfg, monkeypatch, capsys)
    assert result["embedded"] == 0
    assert (result["notes"], result["added"], result["changed"], result["removed"]) == (built.notes, 0, 0, 0)
    conn = connect(cfg)
    st = status(cfg, conn)
    assert st["rules_hash"] == rules_hash(cfg)
    assert st["notes"] == built.notes and st["chunks"] == st["vectors"] == built.chunks
    assert run_index(cfg, monkeypatch, capsys)["embedded"] == 0
    # --rebuild deletes the index file, not the cache, so all chunks come from the converted entries
    rebuilt = run_index(cfg, monkeypatch, capsys, "--rebuild")
    assert (rebuilt["added"], rebuilt["chunks"], rebuilt["embedded"]) == (built.notes, built.chunks, 0)
    assert len(cache_rows(cfg)) == built.embedded


def test_two_index_files_share_one_cache(cfg):
    _, _, first = build(cfg)
    assert first.embedded > 0
    other = dataclasses.replace(cfg, index_path=cfg.data_dir / "variants" / "v" / "c" / "index.db")
    emb = FakeEmbedder()
    result = update(other, connect(other), emb)
    assert (result.added, result.chunks) == (first.added, first.chunks)
    assert result.embedded == emb.calls == 0
    for path in (cfg.index_path, other.index_path):
        assert "embed_cache" not in tables(raw(path))
    assert len(cache_rows(cfg)) == first.embedded


@pytest.mark.parametrize("change", [{"doc_prefix": "passage: {title}\n"}, {"dimension": 16}, {"model": "other-model"}])
def test_an_index_with_other_embedding_settings_does_not_use_the_cache_entries(cfg, change):
    _, _, first = build(cfg)
    other = dataclasses.replace(cfg, index_path=cfg.data_dir / "other.db", **change)
    emb = FakeEmbedder(dimension=other.dimension)
    result = update(other, connect(other), emb)
    assert result.embedded == emb.calls == first.embedded
    assert len(cache_rows(cfg)) == 2 * first.embedded


def test_a_failed_migration_leaves_the_old_index_usable(cfg):
    old_index(cfg, "1.2.0")
    conn = raw(cfg.index_path)
    conn.execute("CREATE VIEW labels AS SELECT path, note_class FROM notes")  # DROP COLUMN fails on a used column
    conn.commit()
    before = tables(conn)
    conn.close()
    with pytest.raises(sqlite3.OperationalError):
        connect(cfg)
    conn = raw(cfg.index_path)
    assert tables(conn) == before
    meta = dict(conn.execute("SELECT key, value FROM meta"))
    assert meta["schema_version"] == "1" and meta["rules_hash"] == rules_hash(cfg)
    assert conn.execute("SELECT count(*) FROM refs").fetchone() == (1,)


@pytest.mark.parametrize("version", ["0", "4"])
def test_an_index_of_another_schema_version_is_refused(cfg, version):
    conn, _, _ = build(cfg)
    conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (version,))
    conn.commit()
    conn.close()
    with pytest.raises(IndexMismatch) as e:
        connect(cfg)
    assert "--rebuild" not in str(e.value)


def test_full_index_then_no_new_embeddings(cfg):
    conn, emb, first = build(cfg)
    assert first.added == first.notes > 0 and emb.calls == first.embedded > 0
    second = update(cfg, conn, emb)
    assert (second.added, second.changed, second.removed, second.embedded, second.rederived) == (0, 0, 0, 0, 0)
    st = status(cfg, conn)
    assert st["notes"] == first.notes and st["chunks"] == st["vectors"] == first.chunks


def test_changed_note_is_rewritten_and_only_new_chunks_are_embedded(cfg):
    conn, emb, _ = build(cfg)
    (cfg.root / "Home/Heating/notes/boiler.md").write_text("# Boiler\nThe flow temperature is 55 degrees.\n")
    result = update(cfg, conn, emb)
    assert (result.changed, result.embedded) == (1, 1)
    hits = conn.execute(
        "SELECT n.path FROM chunks_fts f JOIN chunks c ON c.id = f.rowid JOIN notes n ON n.id = c.note_id"
        " WHERE chunks_fts MATCH 'flow temperature'"
    ).fetchall()
    assert hits == [("Home/Heating/notes/boiler.md",)]


def test_touched_note_is_not_rechunked(cfg):
    conn, emb, _ = build(cfg)
    os.utime(cfg.root / "Home/Heating/notes/boiler.md", ns=(1, 1))
    result = update(cfg, conn, emb)
    assert (result.touched, result.changed, result.embedded) == (1, 0, 0)


def test_deleted_note_is_removed_everywhere(cfg):
    conn, emb, _ = build(cfg)
    (cfg.root / "Home/Astro/Theory/Field Stop.md").unlink()
    result = update(cfg, conn, emb)
    assert result.removed == 1
    assert count(conn, "SELECT count(*) FROM notes WHERE path LIKE '%Field Stop%'") == 0
    assert count(conn, "SELECT count(*) FROM chunks_vec") == count(conn, "SELECT count(*) FROM chunks")
    assert count(conn, "SELECT count(*) FROM chunks_fts") == count(conn, "SELECT count(*) FROM chunks")


def test_moved_note_needs_no_embedding(cfg):
    conn, emb, _ = build(cfg)
    dst = cfg.root / "Home/Astro/Guides/Field Stop.md"
    dst.parent.mkdir(parents=True)
    (cfg.root / "Home/Astro/Theory/Field Stop.md").rename(dst)
    result = update(cfg, conn, emb)
    assert (result.moved, result.added, result.removed, result.embedded) == (1, 0, 0, 0)


def test_sensitivity_column_for_filters(cfg):
    conn, _, _ = build(cfg)
    private = conn.execute(
        "SELECT DISTINCT n.path FROM chunks_vec v JOIN chunks c ON c.id = v.chunk_id JOIN notes n ON n.id = c.note_id"
        " WHERE v.sensitivity = 'private'"
    ).fetchall()
    assert {p for (p,) in private} == {
        "Work/Personal/People/Someone.md",
        "Work/SLA/Incidents/2026-01-01.md",
        "Home/Password/bank.md",
        "Home/Budget/2026.md",
    }


def test_vector_query_with_filter(cfg):
    conn, emb, _ = build(cfg)
    q = sqlite_vec.serialize_float32(emb.vector("anything"))
    rows = conn.execute(
        "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? AND k = 5 AND sensitivity = 'normal'",
        (q,),
    ).fetchall()
    assert len(rows) == 5


def test_links_and_jira(cfg):
    note = cfg.root / "Home/Heating/notes/boiler.md"
    note.write_text("# Boiler\nSee [[SUMMARY|the summary]] and [[Field Stop]] and [[Missing]].\n")
    conn, _, _ = build(cfg)
    rows = conn.execute(
        "SELECT l.target, n.path FROM links l JOIN notes f ON f.id = l.from_note LEFT JOIN notes n ON n.id = l.to_note"
        " WHERE f.path = 'Home/Heating/notes/boiler.md' ORDER BY l.target"
    ).fetchall()
    assert rows == [
        ("Field Stop", "Home/Astro/Theory/Field Stop.md"),
        ("Missing", None),
        ("SUMMARY", "Home/Heating/SUMMARY.md"),
    ]
    keys = conn.execute("SELECT DISTINCT key FROM jira ORDER BY key").fetchall()
    assert keys == [("CORE-123",), ("OPS-77",), ("TS-9",)]


def test_a_relative_link_resolves_from_the_folder_of_the_note(cfg):
    (cfg.root / "Home/Heating/notes/boiler.md").write_text(
        "# Boiler\nSee [[../SUMMARY]], [[./boiler|this note]], [[../../Astro/Theory/Field Stop]] and [[../Nothing]].\n"
    )
    conn, _, _ = build(cfg)
    rows = conn.execute(
        "SELECT l.target, n.path FROM links l JOIN notes f ON f.id = l.from_note LEFT JOIN notes n ON n.id = l.to_note"
        " WHERE f.path = 'Home/Heating/notes/boiler.md' ORDER BY l.target"
    ).fetchall()
    assert rows == [
        ("../../Astro/Theory/Field Stop", FIELD_STOP),
        ("../Nothing", None),
        ("../SUMMARY", "Home/Heating/SUMMARY.md"),
        ("./boiler", "Home/Heating/notes/boiler.md"),
    ]


def test_model_mismatch_needs_rebuild(cfg):
    conn, _, _ = build(cfg)
    conn.close()
    other = Config(root=cfg.root, index_path=cfg.index_path, dimension=8, model="other-model")
    with pytest.raises(IndexMismatch) as e:
        connect(other)
    assert "--rebuild" not in str(e.value)


FIELD_STOP = "Home/Astro/Theory/Field Stop.md"


def note_row(conn, path):
    return conn.execute("SELECT project, area, sensitivity FROM notes WHERE path = ?", (path,)).fetchone()


def test_new_private_pattern_makes_an_indexed_note_private_without_embedding(cfg):
    conn, emb, _ = build(cfg)
    assert note_row(conn, FIELD_STOP)[2] == "normal"
    calls = emb.calls
    cfg.private_patterns = [*cfg.private_patterns, "Home/Astro/**"]
    result = update(cfg, conn, emb)
    assert (result.rederived, result.changed, result.embedded) == (1, 0, 0)
    assert emb.calls == calls
    assert note_row(conn, FIELD_STOP)[2] == "private"
    vec = conn.execute(
        "SELECT DISTINCT v.sensitivity FROM chunks_vec v JOIN chunks c ON c.id = v.chunk_id"
        " JOIN notes n ON n.id = c.note_id WHERE n.path = ?",
        (FIELD_STOP,),
    ).fetchall()
    assert vec == [("private",)]


def test_changed_vault_rule_gives_new_project_and_area_without_embedding(cfg):
    conn, emb, _ = build(cfg)
    # no rules for Home: area = level 1 folder in lower case, project = level 2 folder
    assert note_row(conn, FIELD_STOP)[:2] == ("Theory", "astro")
    calls = emb.calls
    cfg.vault_rules = {**cfg.vault_rules, "Home": VaultRules(area="house", project_depth=1)}
    result = update(cfg, conn, emb)
    assert note_row(conn, FIELD_STOP)[:2] == ("Astro", "house")
    assert result.embedded == 0 and emb.calls == calls
    assert result.changed == 0 and result.rederived > 0


def test_changed_area_embeds_only_the_description_chunk(cfg):
    conn, emb, _ = build(cfg)
    chunks = count(conn, "SELECT count(*) FROM chunks")
    cfg.vault_rules = {**cfg.vault_rules, "Work": dataclasses.replace(cfg.vault_rules["Work"], area="ops")}
    result = update(cfg, conn, emb)
    # only Work/Devops/db-resize/SUMMARY.md has a description; its chunk text holds the area
    assert result.embedded == 1
    assert note_row(conn, "Work/Devops/db-resize/SUMMARY.md") == ("db-resize", "ops", "normal")
    assert count(conn, "SELECT count(*) FROM chunks") == chunks
    assert count(conn, "SELECT count(*) FROM chunks_vec") == chunks


def test_new_skipped_folder_removes_its_notes(cfg):
    conn, emb, _ = build(cfg)
    cfg.skip_dirs = {*cfg.skip_dirs, "Theory"}
    result = update(cfg, conn, emb)
    assert result.removed == 1
    assert count(conn, "SELECT count(*) FROM notes WHERE path LIKE 'Home/Astro/Theory/%'") == 0
    assert count(conn, "SELECT count(*) FROM chunks_vec") == count(conn, "SELECT count(*) FROM chunks")


def test_rules_are_derived_again_only_when_they_change(cfg, monkeypatch):
    conn, emb, first = build(cfg)
    calls = []
    real = index.resolve
    monkeypatch.setattr(index, "resolve", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    ids = conn.execute("SELECT id, path FROM notes ORDER BY id").fetchall()
    update(cfg, conn, emb)
    assert calls == []
    cfg.archive_folder = "_old"
    update(cfg, conn, emb)
    assert len(calls) == first.notes
    calls.clear()
    update(cfg, conn, emb)
    assert calls == []
    assert conn.execute("SELECT id, path FROM notes ORDER BY id").fetchall() != ids  # archived notes changed


def test_rule_change_ignores_the_scan_limit(cfg):
    conn, emb, _ = build(cfg)
    cfg.private_patterns = [*cfg.private_patterns, "Home/**"]
    result = update(cfg, conn, emb, limit=1)
    assert result.warnings == []
    assert count(conn, "SELECT count(*) FROM notes WHERE path LIKE 'Home/%' AND sensitivity = 'normal'") == 0


def test_added_marker_gives_the_notes_of_its_folder_a_new_project_without_embedding(cfg):
    conn, emb, _ = build(cfg)
    assert note_row(conn, FIELD_STOP)[0] == "Theory"
    calls = emb.calls
    marker = cfg.root / "Home/Astro/SUMMARY.md"
    marker.write_text("# Astro\nThe summary of the astro folder.\n")
    result = update(cfg, conn, emb)
    assert note_row(conn, FIELD_STOP)[0] == "Astro"
    assert result.rederived == 1 and result.added == 1
    assert emb.calls - calls == result.embedded  # only the new marker note is embedded
    marker.unlink()
    result = update(cfg, conn, emb)
    assert note_row(conn, FIELD_STOP)[0] == "Theory"
    assert (result.removed, result.rederived, result.embedded) == (1, 1, 0)


def test_marker_change_leaves_notes_outside_its_folder_alone(cfg, monkeypatch):
    conn, emb, _ = build(cfg)
    (cfg.root / "Home/Astro/SUMMARY.md").write_text("# Astro\nThe summary of the astro folder.\n")
    calls = []
    real = index.resolve
    monkeypatch.setattr(index, "resolve", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    update(cfg, conn, emb)
    assert calls and all(rel.startswith("Home/Astro/") for rel in calls)


def test_link_goes_to_the_note_in_the_nearest_folder(cfg):
    (cfg.root / "Home/Heating/DRIVERS.md").write_text("# Drivers\nThe heating drivers.\n")
    (cfg.root / "Home/Astro/DRIVERS.md").write_text("# Drivers\nThe astro drivers.\n")
    (cfg.root / "Home/Heating/notes/boiler.md").write_text("# Boiler\nSee [[DRIVERS]].\n")
    (cfg.root / FIELD_STOP).write_text("# Field Stop\nSee [[DRIVERS]].\n")
    conn, _, _ = build(cfg)
    rows = conn.execute(
        "SELECT f.path, n.path FROM links l JOIN notes f ON f.id = l.from_note JOIN notes n ON n.id = l.to_note"
        " WHERE l.target = 'DRIVERS' ORDER BY f.path"
    ).fetchall()
    assert rows == [
        (FIELD_STOP, "Home/Astro/DRIVERS.md"),
        ("Home/Heating/notes/boiler.md", "Home/Heating/DRIVERS.md"),
    ]


def test_marker_rederivation_does_not_use_the_scan_limit(cfg):
    conn, emb, _ = build(cfg)
    edited = cfg.root / "Work/Sales/Answers/Security.md"
    edited.write_text("# Security\nThe new answer.\n")
    os.utime(edited, ns=(1, 1))  # the oldest note, so the scan reaches it after the re-derived note
    (cfg.root / "Home/Astro/SUMMARY.md").write_text("# Astro\nThe summary of the astro folder.\n")
    result = update(cfg, conn, emb, limit=2)
    assert (result.added, result.changed, result.rederived, result.deferred) == (1, 1, 1, 0)


def test_edited_summary_state_changes_the_status_of_its_project_without_embedding(cfg):
    conn, emb, _ = build(cfg)

    def status():
        return conn.execute("SELECT status FROM notes WHERE path = ?", ("Home/Heating/notes/boiler.md",)).fetchone()[0]

    assert status() == "active"
    summary = cfg.root / "Home/Heating/SUMMARY.md"
    summary.write_text("---\nstate: superseded\n---\n# Heating\n")
    result = update(cfg, conn, emb)
    assert status() == "archived"
    assert result.embedded <= 1  # only the edited SUMMARY.md may need a new embedding


def test_a_note_that_another_writer_added_during_the_update_is_replaced(cfg, monkeypatch):
    # the MCP server scans before each call, also while `ragnarok index` runs in a shell
    conn, other = connect(cfg), connect(cfg)
    emb = FakeEmbedder()
    write = index._write

    def other_writer_first(*args):
        monkeypatch.setattr(index, "_write", write)
        assert update(cfg, other, FakeEmbedder(), limit=1).added == 1
        write(*args)

    monkeypatch.setattr(index, "_write", other_writer_first)
    result = update(cfg, conn, emb)
    assert result.added == result.notes
    assert count(conn, "SELECT COUNT(*) FROM notes") == result.notes
    assert count(conn, "SELECT COUNT(*) FROM chunks") == count(conn, "SELECT COUNT(*) FROM chunks_vec") == result.chunks
