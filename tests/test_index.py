import hashlib
import os

import pytest
import sqlite_vec

from ragnarok.config import Config
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
    assert (second.added, second.changed, second.removed, second.embedded) == (0, 0, 0, 0)
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
