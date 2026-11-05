import itertools
import sqlite3
from dataclasses import replace

import numpy as np
import pytest
import sqlite_vec

from ragnarok.index import connect
from ragnarok.search import CANDIDATES, Filters, _dense, _fts, eligibility, search

DIM = 8
NORMAL = ("V", "P", "A", "T", "active", "normal")


class FixedEmbedder:
    def __init__(self, vector):
        self.v = vector

    def embed_query(self, query):
        return list(self.v)


def unit_vectors(count, seed=1):
    v = np.random.default_rng(seed).normal(size=(count, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def seed_index(cfg, specs, vectors, text="needle"):
    """Insert one doc with one unit per spec (vault, project, area, type, status, sensitivity)."""
    cfg = replace(cfg, dimension=DIM)
    conn = connect(cfg)
    for i, (vault, project, area, type_, status, sensitivity) in enumerate(specs, start=1):
        doc = f"d{i}"
        conn.execute(
            "INSERT INTO node (id, node_type, collection, mode, provisional) VALUES (?, 'doc', 'c', 'm', 0)", (doc,)
        )
        conn.execute(
            "INSERT INTO doc (id, vault, path, title, project, area, type, status, sensitivity, mtime_ns, size, hash)"
            " VALUES (?, ?, ?, 't', ?, ?, ?, ?, ?, 0, 0, 'h')",
            (doc, vault, f"{doc}.md", project, area, type_, status, sensitivity),
        )
        conn.execute(
            "INSERT INTO unit (id, doc_id, doc_hash, ord, heading_path, text, text_hash, words)"
            " VALUES (?, ?, 'h', 0, 'h', ?, ?, 1)",
            (i, doc, text, f"t{i}"),
        )
        conn.execute("INSERT INTO unit_fts (rowid, heading_path, text) VALUES (?, 'h', ?)", (i, text))
        conn.execute(
            "INSERT INTO unit_emb (unit_id, doc_id, embedding) VALUES (?, ?, ?)", (i, doc, vectors[i - 1].tobytes())
        )
    conn.commit()
    return cfg, conn


def test_exact_scan_equals_vec0_without_filters(vault):
    vectors = unit_vectors(500)
    cfg, conn = seed_index(vault, [NORMAL] * 500, vectors)
    query = unit_vectors(1, seed=99)[0]

    ref = sqlite3.connect(":memory:")
    ref.enable_load_extension(True)
    sqlite_vec.load(ref)
    ref.execute(
        f"CREATE VIRTUAL TABLE v USING vec0(unit_id INTEGER PRIMARY KEY, embedding float[{DIM}] distance_metric=cosine)"
    )
    ref.executemany(
        "INSERT INTO v (unit_id, embedding) VALUES (?, ?)",
        [(i + 1, sqlite_vec.serialize_float32(x)) for i, x in enumerate(vectors)],
    )
    expected = [
        r[0]
        for r in ref.execute(
            "SELECT unit_id FROM v WHERE embedding MATCH ? AND k = 200 ORDER BY distance",
            [sqlite_vec.serialize_float32(query)],
        )
    ]

    assert len(expected) == CANDIDATES
    assert _dense(conn, query, Filters())[0] == expected


def test_filtered_dense_finds_a_note_outside_the_global_top_k(vault):
    count = 1200
    query = unit_vectors(1, seed=5)[0]
    vectors = unit_vectors(count, seed=6)
    vectors[-1] = -query  # the farthest unit from the query: rank 1200 of 1200
    specs = [NORMAL] * (count - 1) + [("V", "Target", "A", "T", "active", "normal")]
    cfg, conn = seed_index(vault, specs, vectors)
    # the precondition: the target is outside the global top 1,000 that the old k for a filter kept
    order = sorted(range(count), key=lambda i: -float(vectors[i] @ query))
    assert order.index(count - 1) >= 1000
    hits = search(cfg, conn, FixedEmbedder(query), "q", project="Target", mode="dense")
    assert [h.path for h in hits] == [f"d{count}.md"]


@pytest.mark.parametrize("include_archived", [True, False])
@pytest.mark.parametrize("include_private", [True, False])
def test_fts_and_dense_use_the_same_predicate(vault, include_archived, include_private):
    names = [
        ("V", "P", "A", "T"),
        ("W", "P", "A", "T"),
        ("V", "Q", "A", "T"),
        ("V", "P", "B", "T"),
        ("V", "P", "A", "U"),
    ]
    specs = [(*n, status, sens) for n in names for status in ("active", "archived") for sens in ("normal", "private")]
    cfg, conn = seed_index(vault, specs, unit_vectors(len(specs)))
    query = unit_vectors(1, seed=3)[0]
    options = [None, "V", "W"], [None, "P", "Q"], [None, "A", "B"], [None, "T", "U"]
    for vault_, project, area, type_ in itertools.product(*options):
        f = Filters(vault_, project, area, type_, include_archived, include_private)
        sql, params = eligibility(f)
        want = {
            r[0] for r in conn.execute(f"SELECT u.id FROM unit u JOIN doc d ON d.id = u.doc_id WHERE {sql}", params)
        }
        assert set(_dense(conn, query, f)[0]) == want
        assert set(_fts(conn, "needle", 1000, f)) == want
    # the predicate is not trivial: the filters remove units
    assert len(want) < len(specs)


def test_a_scan_over_the_exact_limit_still_runs_and_warns(vault):
    vectors = unit_vectors(30)
    cfg, conn = seed_index(vault, [NORMAL] * 30, vectors)
    low = replace(cfg, kb=replace(cfg.kb, exact_limit=10))
    warnings: list[str] = []
    hits = search(low, conn, FixedEmbedder(vectors[0]), "q", mode="dense", warnings=warnings)
    assert hits and len(warnings) == 1
    quiet: list[str] = []
    search(cfg, conn, FixedEmbedder(vectors[0]), "q", mode="dense", warnings=quiet)
    assert quiet == []
