import dataclasses
import hashlib
import os

import pytest
import sqlite_vec

from ragnarok import index
from ragnarok.config import Config, VaultRules
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


def test_model_mismatch_needs_rebuild(cfg):
    conn, _, _ = build(cfg)
    conn.close()
    other = Config(root=cfg.root, index_path=cfg.index_path, dimension=8, model="other-model")
    with pytest.raises(IndexMismatch):
        connect(other)


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
