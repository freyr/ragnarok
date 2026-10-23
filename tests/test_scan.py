"""The change scan before each server call: changes, moves, deletions and the limit of changed notes."""

import os

import pytest
from test_index import FakeEmbedder

from ragnarok.index import NOT_CURRENT, SCAN_LIMIT, connect, update
from ragnarok.search import search


@pytest.fixture
def built(vault):
    vault.dimension = 8
    conn = connect(vault)
    emb = FakeEmbedder()
    update(vault, conn, emb)
    return vault, conn, emb


def paths(cfg, conn, emb, query):
    return [h.path for h in search(cfg, conn, emb, query, mode="fts")]


def test_changed_note_is_in_the_next_search(built):
    cfg, conn, emb = built
    (cfg.root / "Home/Heating/notes/boiler.md").write_text("# Boiler\nThe expansion vessel lost pressure.\n")
    update(cfg, conn, emb, limit=SCAN_LIMIT)
    assert paths(cfg, conn, emb, "expansion vessel") == ["Home/Heating/notes/boiler.md"]


def test_moved_folder_needs_no_embedding(built):
    cfg, conn, emb = built
    src, dst = cfg.root / "Home/Batch", cfg.root / "Home/Moved/Batch"
    src.mkdir()
    for i in range(10):
        (src / f"note-{i}.md").write_text(f"# Note {i}\nUnique word quokka{i} in a batch note.\n")
    update(cfg, conn, emb)
    dst.parent.mkdir()
    src.rename(dst)
    calls = emb.calls
    result = update(cfg, conn, emb, limit=SCAN_LIMIT)
    assert (result.moved, result.added, result.removed, result.embedded) == (10, 0, 0, 0)
    assert emb.calls == calls
    assert paths(cfg, conn, emb, "quokka3") == ["Home/Moved/Batch/note-3.md"]


def test_move_into_a_sensitive_folder_makes_the_note_private(built):
    cfg, conn, emb = built
    dst = cfg.root / "Home/Budget/Field Stop.md"
    (cfg.root / "Home/Astro/Theory/Field Stop.md").rename(dst)
    result = update(cfg, conn, emb)
    assert result.moved == 1
    row = conn.execute("SELECT sensitivity FROM notes WHERE path = 'Home/Budget/Field Stop.md'").fetchone()
    assert row == ("private",)
    vec = conn.execute(
        "SELECT DISTINCT v.sensitivity FROM chunks_vec v JOIN chunks c ON c.id = v.chunk_id"
        " JOIN notes n ON n.id = c.note_id WHERE n.path = 'Home/Budget/Field Stop.md'"
    ).fetchall()
    assert vec == [("private",)]


def test_deleted_note_is_not_in_the_results(built):
    cfg, conn, emb = built
    assert "Home/Astro/Theory/Field Stop.md" in paths(cfg, conn, emb, "field stop")
    (cfg.root / "Home/Astro/Theory/Field Stop.md").unlink()
    result = update(cfg, conn, emb, limit=SCAN_LIMIT)
    assert result.removed == 1
    assert "Home/Astro/Theory/Field Stop.md" not in paths(cfg, conn, emb, "field stop")


def test_more_than_the_limit_gives_the_warning_and_the_next_scan_continues(built):
    cfg, conn, emb = built
    folder = cfg.root / "Home/Many"
    folder.mkdir()
    for i in range(30):
        note = folder / f"note-{i:02}.md"
        note.write_text(f"# Note {i}\nText of note {i}.\n")
        os.utime(note, ns=(i * 10**9, i * 10**9))
    first = update(cfg, conn, emb, limit=SCAN_LIMIT)
    assert (first.added, first.deferred, first.warnings) == (20, 10, [NOT_CURRENT])
    newest = conn.execute(
        "SELECT count(*) FROM notes WHERE path BETWEEN 'Home/Many/note-10.md' AND 'Home/Many/note-29.md'"
    ).fetchone()
    assert newest == (20,)
    second = update(cfg, conn, emb, limit=SCAN_LIMIT)
    assert (second.added, second.deferred, second.warnings) == (10, 0, [])


def test_no_limit_for_the_full_index(built):
    cfg, conn, emb = built
    folder = cfg.root / "Home/Many"
    folder.mkdir()
    for i in range(30):
        (folder / f"note-{i:02}.md").write_text(f"# Note {i}\nText of note {i}.\n")
    result = update(cfg, conn, emb)
    assert (result.added, result.deferred, result.warnings) == (30, 0, [])
