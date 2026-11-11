"""Variants, corpora, `bench index`, chunk_prefix and note_rank."""

import dataclasses
import json
import math
import sqlite3

import pytest
import sqlite_vec

from ragnarok.cli import build_parser
from ragnarok.config import Config, load_config
from ragnarok.index import _delete_note, connect, sync_note_rank, update
from ragnarok.search import search

# --- configuration -------------------------------------------------------------------------------------------------

CONFIG = """
root = "vaults"
vaults = ["Home", "Work"]
data_dir = "data"
model = "embeddinggemma-2:270m"

[variants.baseline]

[variants.small]
model = "qwen3-embedding:0.6b"
dimension = 512
min_words = 40
max_words = 200
chunk_prefix = "description"
note_rank = "note_vector"
query_prefix = "q: "
doc_template = "{title}\\n"

[corpora.real]
root = "vaults"
vaults = ["Home", "Work"]

[corpora.synthetic]
root = "synthetic/pilot"
vaults = ["Alpha", "Beta"]
"""


def test_a_variant_on_a_corpus_has_its_own_index_and_shares_the_data_folder(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    cfg = load_config(path)
    assert (cfg.chunk_prefix, cfg.note_rank) == ("none", "note_vector")
    assert cfg.index_path == tmp_path / "data/index.db"

    v = cfg.variant("small", "synthetic")
    assert v.index_path == tmp_path / "data/variants/small/synthetic/index.db"
    assert v.data_dir == tmp_path / "data"
    assert (v.root, v.vaults) == (tmp_path / "synthetic/pilot", ["Alpha", "Beta"])
    assert (v.model, v.dimension, v.min_words, v.max_words) == ("qwen3-embedding:0.6b", 512, 40, 200)
    assert (v.chunk_prefix, v.note_rank, v.query_prefix, v.doc_template) == (
        "description",
        "note_vector",
        "q: ",
        "{title}\n",
    )
    assert v.note_path("Alpha/x.md") == tmp_path / "synthetic/pilot/Alpha/x.md"

    base = cfg.variant("baseline", "real")
    assert base.index_path == tmp_path / "data/variants/baseline/real/index.db"
    same = ("model", "dimension", "query_prefix", "doc_template", "min_words", "max_words", "chunk_prefix", "note_rank")
    assert all(getattr(base, k) == getattr(cfg, k) for k in same)
    assert (base.root, base.vaults) == (cfg.root, cfg.vaults)


@pytest.mark.parametrize(
    "text",
    [
        '[variants.x]\nollama_url = "http://other"\n',  # not a variant key
        '[variants.x]\nchunk_prefix = "summary"\n',
        '[variants.x]\nnote_rank = "sum"\n',
        'note_rank = "sum"\n',
        '[corpora.c]\nroot = "v"\n',  # no vaults
        '[corpora.c]\nroot = "v"\nvaults = ["A"]\nextra = 1\n',
    ],
)
def test_wrong_variant_and_corpus_tables_are_refused(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        load_config(path)


def test_an_unknown_variant_or_corpus_is_refused(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(CONFIG)
    cfg = load_config(path)
    with pytest.raises(KeyError):
        cfg.variant("nope", "real")
    with pytest.raises(KeyError):
        cfg.variant("baseline", "nope")


def test_parser_has_bench_index():
    args = build_parser().parse_args(["bench", "index", "--variant", "baseline", "--corpus", "real"])
    assert (args.command, args.bench, args.variant, args.corpus) == ("bench", "index", "baseline", "real")


# --- a small fixture vault -----------------------------------------------------------------------------------------


class TableEmbedder:
    """The vector of a text is the vector of the first key word in it; the query vector is fixed."""

    def __init__(self, table, query=(1.0, 0.0, 0.0)):
        self.table = table
        self.query = list(query)
        self.items = []

    def embed_docs(self, items):
        self.items.extend(items)
        return [self.vector(t) for _, t in items]

    def embed_query(self, query):
        return self.query

    def vector(self, text):
        for word, v in self.table.items():
            if word in text:
                return list(v)
        return [0.0, 0.0, 1.0]


def make(tmp_path, notes, **settings):
    for rel, text in notes.items():
        path = tmp_path / "vaults" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Config(
        root=tmp_path / "vaults",
        vaults=["V"],
        index_path=tmp_path / "data/index.db",
        dimension=3,
        min_words=1,
        **settings,
    )


def sections(*words):
    return "".join(f"# h{i}\n{w}\n" for i, w in enumerate(words))


S09 = (0.9, math.sqrt(1 - 0.81), 0.0)  # cosine 0.9 with the query
S08 = (0.8, 0.6, 0.0)  # cosine 0.8
S06 = (0.6, 0.8, 0.0)  # cosine 0.6

# note A: three chunks of 0.8 (sum 2.4); note B: one of 0.9 and one of 0 (sum 0.9); note C: one of 0.6
DENSE = {"V/A.md": sections("wa1", "wa2", "wa3"), "V/B.md": sections("wb1", "zero"), "V/C.md": sections("wc1")}
DENSE_TABLE = {"wa": S08, "wb": S09, "wc": S06, "zero": (0.0, 0.0, 1.0)}


def ranked(cfg, emb, query="q", **kw):
    conn = connect(cfg)
    update(cfg, conn, emb)
    return [h.path for h in search(cfg, conn, emb, query, **kw)]


def test_maxp_ranks_a_note_by_its_best_chunk(tmp_path):
    cfg = make(tmp_path, DENSE, note_rank="maxp")
    assert ranked(cfg, TableEmbedder(DENSE_TABLE), mode="dense") == ["V/B.md", "V/A.md", "V/C.md"]


def test_topk_sum_ranks_a_note_by_the_sum_of_its_best_three_dense_chunks(tmp_path):
    cfg = make(tmp_path, DENSE, note_rank="topk_sum")
    assert ranked(cfg, TableEmbedder(DENSE_TABLE), mode="dense") == ["V/A.md", "V/B.md", "V/C.md"]


def test_topk_sum_takes_only_the_best_three_chunks(tmp_path):
    # D: four chunks of 0.6 (best three 1.8, all four 2.4); E: three chunks of 0.7 (2.1)
    s07 = (0.7, math.sqrt(1 - 0.49), 0.0)
    notes = {"V/D.md": sections("wd1", "wd2", "wd3", "wd4"), "V/E.md": sections("we1", "we2", "we3")}
    cfg = make(tmp_path, notes, note_rank="topk_sum")
    assert ranked(cfg, TableEmbedder({"wd": S06, "we": s07}), mode="dense") == ["V/E.md", "V/D.md"]


# A: three chunks with "zebra" in four words; B: one chunk with "zebra" in two words (a higher BM25 score than one
# chunk of A, but not three times higher); F: ten chunks with no "zebra", so that the word is rare
FTS = {
    "V/A.md": sections(*["zebra one two three"] * 3),
    "V/B.md": sections("zebra four"),
    "V/F.md": sections(*[f"other words here {i}" for i in range(10)]),
}


def test_maxp_ranks_the_note_with_the_best_bm25_chunk_first(tmp_path):
    cfg = make(tmp_path, FTS, note_rank="maxp")
    assert ranked(cfg, TableEmbedder({}), "zebra", mode="fts") == ["V/B.md", "V/A.md"]


def test_topk_sum_ranks_the_note_with_the_best_bm25_sum_first(tmp_path):
    cfg = make(tmp_path, FTS, note_rank="topk_sum")
    assert ranked(cfg, TableEmbedder({}), "zebra", mode="fts") == ["V/A.md", "V/B.md"]


# P: chunks (1, 0, 0) and (0, 0, 1): best chunk 1.0, mean vector cosine 0.707
# Q: two chunks of 0.9: best chunk 0.9, mean vector cosine 0.9
# R: one chunk of 0.8: best chunk 0.8, mean vector cosine 0.8
# chunk list P 1, Q 2, R 3; note list Q 1, R 2, P 3; RRF: Q 1/62 + 1/61 > P 1/61 + 1/63 > R 1/63 + 1/62
VECTOR = {"V/P.md": sections("wp1", "zero"), "V/Q.md": sections("wq1", "wq2"), "V/R.md": sections("wr1")}
VECTOR_TABLE = {"wp": (1.0, 0.0, 0.0), "wq": S09, "wr": S08, "zero": (0.0, 0.0, 1.0)}


def test_maxp_without_a_note_vector(tmp_path):
    cfg = make(tmp_path, VECTOR, note_rank="maxp")
    assert ranked(cfg, TableEmbedder(VECTOR_TABLE), mode="dense") == ["V/P.md", "V/Q.md", "V/R.md"]


def test_note_vector_adds_the_mean_chunk_vector_of_each_note_as_a_third_list(tmp_path):
    cfg = make(tmp_path, VECTOR, note_rank="note_vector")
    assert ranked(cfg, TableEmbedder(VECTOR_TABLE), mode="dense") == ["V/Q.md", "V/P.md", "V/R.md"]
    conn = sqlite3.connect(cfg.index_path)
    assert conn.execute("SELECT count(*) FROM notes_vec_rowids").fetchone() == (3,)


def test_note_vector_rows_follow_the_notes(tmp_path):
    cfg = make(tmp_path, VECTOR, note_rank="note_vector")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(cfg)
    update(cfg, conn, emb)
    (cfg.root / "V/P.md").unlink()
    (cfg.root / "V/R.md").write_text(sections("wr1", "wr2"))
    update(cfg, conn, emb)
    ids = {nid for (nid,) in conn.execute("SELECT note_id FROM notes_vec")}
    assert ids == {nid for (nid,) in conn.execute("SELECT id FROM notes")}
    assert len(ids) == 2
    hits = search(cfg, conn, emb, "q", mode="dense")
    assert [h.path for h in hits] == ["V/Q.md", "V/R.md"]


def test_an_index_with_maxp_has_no_note_vector_table(tmp_path):
    cfg = make(tmp_path, VECTOR, note_rank="maxp")
    connect(cfg).close()
    names = {n for (n,) in sqlite3.connect(cfg.index_path).execute("SELECT name FROM sqlite_master")}
    assert not any(n.startswith("notes_vec") for n in names)


# --- the default and the switch to note_vector ------------------------------------------------------------------------


def note_vectors(cfg):
    conn = sqlite3.connect(cfg.index_path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    rows = conn.execute(
        "SELECT n.path, v.embedding, v.vault, v.sensitivity, v.status FROM notes_vec v JOIN notes n ON n.id = v.note_id"
    ).fetchall()
    conn.close()
    return {path: rest for path, *rest in rows}


def test_note_vector_is_the_default_note_rank():
    assert Config().note_rank == "note_vector"


def test_an_index_of_maxp_gets_the_note_vectors_of_a_fresh_build_with_no_changed_note(tmp_path):
    maxp = make(tmp_path / "a", VECTOR, note_rank="maxp")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(maxp)
    update(maxp, conn, emb)
    conn.close()
    assert not note_vectors_table(maxp)
    nv = dataclasses.replace(maxp, note_rank="note_vector")
    conn = connect(nv)
    result = update(nv, conn, emb)
    assert result.added == result.changed == 0
    conn.close()
    fresh = make(tmp_path / "b", VECTOR, note_rank="note_vector")
    fconn = connect(fresh)
    update(fresh, fconn, TableEmbedder(VECTOR_TABLE))
    fconn.close()
    got, want = note_vectors(nv), note_vectors(fresh)
    assert len(got) == 3
    assert got == want
    assert stored_rank(nv) == "note_vector"


def note_vectors_table(cfg):
    names = {n for (n,) in sqlite3.connect(cfg.index_path).execute("SELECT name FROM sqlite_master")}
    return any(n.startswith("notes_vec") for n in names)


def stored_rank(cfg):
    row = sqlite3.connect(cfg.index_path).execute("SELECT value FROM meta WHERE key = 'note_rank'").fetchone()
    return row and row[0]


def test_note_vector_then_maxp_then_note_vector_leaves_no_stale_or_orphan_row(tmp_path):
    nv = make(tmp_path / "a", VECTOR, note_rank="note_vector")
    maxp = dataclasses.replace(nv, note_rank="maxp")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(nv)
    update(nv, conn, emb)
    conn.close()
    conn = connect(maxp)
    (maxp.root / "V/R.md").write_text(sections("wr1", "zero"))
    (maxp.root / "V/P.md").unlink()
    update(maxp, conn, emb)
    assert stored_rank(maxp) == "maxp"
    conn.close()
    conn = connect(nv)
    update(nv, conn, emb)
    conn.close()
    got = note_vectors(nv)
    assert sorted(got) == ["V/Q.md", "V/R.md"]
    fresh = make(
        tmp_path / "b", {"V/Q.md": VECTOR["V/Q.md"], "V/R.md": sections("wr1", "zero")}, note_rank="note_vector"
    )
    fconn = connect(fresh)
    update(fresh, fconn, TableEmbedder(VECTOR_TABLE))
    fconn.close()
    assert got == note_vectors(fresh)
    check = sqlite3.connect(nv.index_path)
    assert check.execute("SELECT count(*) FROM notes_vec_rowids").fetchone() == (2,)


def test_the_note_vectors_are_built_with_a_limit_and_with_no_embedding_call(tmp_path):
    maxp = make(tmp_path, VECTOR, note_rank="maxp")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(maxp)
    update(maxp, conn, emb)
    conn.close()
    nv = dataclasses.replace(maxp, note_rank="note_vector")
    conn = connect(nv)
    calls = len(emb.items)
    update(nv, conn, emb, limit=1)
    assert len(emb.items) == calls
    assert len(note_vectors(nv)) == 3


def test_a_change_to_another_rank_only_stores_it_and_leaves_the_table(tmp_path):
    nv = make(tmp_path, VECTOR, note_rank="note_vector")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(nv)
    update(nv, conn, emb)
    conn.close()
    topk = dataclasses.replace(nv, note_rank="topk_sum")
    conn = connect(topk)
    update(topk, conn, emb)
    assert stored_rank(topk) == "topk_sum"
    assert len(note_vectors(topk)) == 3


def test_status_reports_the_note_rank_and_the_note_vector_rows(tmp_path):
    from ragnarok.index import status

    nv = make(tmp_path / "a", VECTOR, note_rank="note_vector")
    conn = connect(nv)
    update(nv, conn, TableEmbedder(VECTOR_TABLE))
    st = status(nv, conn)
    assert (st["note_rank"], st["note_vectors"]) == ("note_vector", 3)
    maxp = make(tmp_path / "b", VECTOR, note_rank="maxp")
    conn = connect(maxp)
    update(maxp, conn, TableEmbedder(VECTOR_TABLE))
    st = status(maxp, conn)
    assert (st["note_rank"], st["note_vectors"]) == ("maxp", 0)


# --- drift of notes_vec and orphan rows ------------------------------------------------------------


def built_note_vector(tmp_path):
    cfg = make(tmp_path, VECTOR, note_rank="note_vector", allow_private=True)
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(cfg)
    update(cfg, conn, emb)
    return cfg, conn, emb


def test_an_orphan_note_vector_row_never_reaches_a_search_with_no_other_filter(tmp_path):
    cfg, conn, emb = built_note_vector(tmp_path)
    nid = conn.execute("SELECT id FROM notes WHERE path = 'V/Q.md'").fetchone()[0]
    _delete_note(conn, nid)
    conn.execute("DELETE FROM notes WHERE id = ?", (nid,))
    (cfg.root / "V/Q.md").unlink()  # an older server: it does not delete the row of notes_vec
    conn.commit()
    hits = search(cfg, conn, emb, "q", mode="dense", include_private=True)
    assert sorted(h.path for h in hits) == ["V/P.md", "V/R.md"]


def test_a_missing_note_vector_row_is_repaired_by_the_next_update(tmp_path):
    cfg, conn, emb = built_note_vector(tmp_path)
    want = note_vectors(cfg)
    conn.execute("DELETE FROM notes_vec WHERE note_id = (SELECT id FROM notes WHERE path = 'V/R.md')")
    conn.commit()
    update(cfg, conn, emb)
    assert note_vectors(cfg) == want


def test_an_orphan_note_vector_row_is_removed_by_the_next_update(tmp_path):
    cfg, conn, emb = built_note_vector(tmp_path)
    nid = conn.execute("SELECT id FROM notes WHERE path = 'V/Q.md'").fetchone()[0]
    _delete_note(conn, nid)
    conn.execute("DELETE FROM notes WHERE id = ?", (nid,))
    (cfg.root / "V/Q.md").unlink()
    conn.commit()
    assert conn.execute("SELECT count(*) FROM notes_vec_rowids").fetchone()[0] == 3
    update(cfg, conn, emb)
    assert sorted(note_vectors(cfg)) == ["V/P.md", "V/R.md"]
    assert conn.execute("SELECT count(*) FROM notes_vec_rowids").fetchone()[0] == 2


def test_no_drift_means_no_write_in_the_sync(tmp_path):
    cfg, conn, _emb = built_note_vector(tmp_path)
    before = conn.total_changes
    sync_note_rank(cfg, conn)
    assert conn.total_changes == before
    assert not conn.in_transaction


def test_status_says_whether_the_note_vectors_are_current(tmp_path):
    from ragnarok.index import status

    cfg, conn, _emb = built_note_vector(tmp_path)
    assert status(cfg, conn)["note_vectors_current"] is True
    conn.execute("DELETE FROM notes_vec WHERE note_id = (SELECT id FROM notes WHERE path = 'V/R.md')")
    conn.commit()
    assert status(cfg, conn)["note_vectors_current"] is False
    maxp = dataclasses.replace(cfg, note_rank="maxp")
    assert status(maxp, conn)["note_vectors_current"] is True


def test_the_default_rank_builds_the_note_vectors_of_an_index_with_no_note_rank_key(tmp_path):
    cfg, conn, emb = built_note_vector(tmp_path / "a")
    want = note_vectors(cfg)
    conn.execute("DELETE FROM meta WHERE key = 'note_rank'")
    conn.execute("DELETE FROM notes_vec")
    conn.commit()
    conn.close()
    conn = connect(cfg)  # the default rank, as after the upgrade of an index made by an older release
    update(cfg, conn, emb)
    assert note_vectors(cfg) == want
    assert stored_rank(cfg) == "note_vector"


def test_an_error_in_the_rebuild_leaves_the_table_and_the_meta_value(tmp_path, monkeypatch):
    maxp = make(tmp_path, VECTOR, note_rank="maxp")
    emb = TableEmbedder(VECTOR_TABLE)
    conn = connect(maxp)
    update(maxp, conn, emb)
    conn.close()
    nv = dataclasses.replace(maxp, note_rank="note_vector")
    conn = connect(nv)
    conn.execute(
        "INSERT INTO notes_vec (note_id, embedding, vault, sensitivity, status) VALUES (99, ?, 'V', 'x', 'y')",
        (sqlite_vec.serialize_float32([0.0, 0.0, 1.0]),),
    )
    conn.commit()
    monkeypatch.setattr(sqlite_vec, "serialize_float32", lambda v: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        sync_note_rank(nv, conn)
    monkeypatch.undo()
    assert not conn.in_transaction
    assert stored_rank(nv) == "maxp"
    assert [r[0] for r in conn.execute("SELECT note_id FROM notes_vec")] == [99]


# --- chunk_prefix --------------------------------------------------------------------------------------------------

PREFIX = {
    "V/D.md": "---\ndescription: Boiler service plan.\n---\n# Flow\nwd1 flow temperature\n# Pump\nwd2 pump speed\n",
    "V/N.md": "# Only\nwn1 no description\n",
}


def test_chunk_prefix_none_sends_the_chunk_text_alone(tmp_path):
    cfg = make(tmp_path, PREFIX)
    emb = TableEmbedder({})
    update(cfg, connect(cfg), emb)
    assert sorted(emb.items) == [
        ("D > Flow", "wd1 flow temperature"),
        ("D > Pump", "wd2 pump speed"),
        ("D > description", "project: none. area: v. Boiler service plan."),
        ("N > Only", "wn1 no description"),
    ]


def test_chunk_prefix_description_puts_the_description_before_each_body_chunk(tmp_path):
    cfg = make(tmp_path, PREFIX, chunk_prefix="description")
    emb = TableEmbedder({})
    conn = connect(cfg)
    update(cfg, conn, emb)
    assert sorted(emb.items) == [
        ("D > Flow", "Boiler service plan.\nwd1 flow temperature"),
        ("D > Pump", "Boiler service plan.\nwd2 pump speed"),
        ("D > description", "project: none. area: v. Boiler service plan."),
        ("N > Only", "wn1 no description"),
    ]
    # the stored chunk text and the BM25 text stay the chunk text
    assert conn.execute("SELECT text FROM chunks WHERE heading = 'D > Flow'").fetchone() == ("wd1 flow temperature",)
    assert conn.execute("SELECT text FROM chunks_fts WHERE heading = 'D > Flow'").fetchone() == (
        "wd1 flow temperature",
    )


def test_chunk_prefix_shares_the_cache_only_for_notes_without_a_description(tmp_path):
    base = make(tmp_path, PREFIX)
    update(base, connect(base), TableEmbedder({}))
    prefixed = dataclasses.replace(
        base, chunk_prefix="description", index_path=base.data_dir / "variants/p/real/index.db"
    )
    emb = TableEmbedder({})
    conn = connect(prefixed)
    result = update(prefixed, conn, emb)
    assert sorted(emb.items) == [
        ("D > Flow", "Boiler service plan.\nwd1 flow temperature"),
        ("D > Pump", "Boiler service plan.\nwd2 pump speed"),
    ]
    assert result.embedded == 2
    # a second update finds the prefixed chunks the same and embeds nothing
    assert update(prefixed, conn, emb).embedded == 0


# --- bench index ---------------------------------------------------------------------------------------------------


def bench_home(tmp_path, monkeypatch, notes, extra=""):
    home = tmp_path / "home"
    for rel, text in notes.items():
        path = home / "vaults" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (home / "config.toml").write_text(
        f'root = "vaults"\nvaults = ["V"]\ndata_dir = "data"\ndimension = 3\nmin_words = 1\n{extra}'
    )
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda cfg: TableEmbedder({}))
    return home


def run(capsys, *argv):
    from ragnarok.cli import main

    code = main(list(argv))
    out = capsys.readouterr()
    return code, out


def hashes(path):
    conn = sqlite3.connect(path)
    return sorted(
        conn.execute(
            "SELECT n.path, c.ord, c.heading, c.text, c.text_hash FROM chunks c JOIN notes n ON n.id = c.note_id"
        )
    )


def test_bench_index_of_the_baseline_on_real_gives_the_default_chunks_with_no_embedding(tmp_path, monkeypatch, capsys):
    home = bench_home(
        tmp_path, monkeypatch, PREFIX, '\n[variants.baseline]\n\n[corpora.real]\nroot = "vaults"\nvaults = ["V"]\n'
    )
    code, _ = run(capsys, "index")
    assert code == 0
    code, out = run(capsys, "bench", "index", "--variant", "baseline", "--corpus", "real")
    assert code == 0, out.err
    result = json.loads(out.out)
    assert (result["embedded"], result["added"]) == (0, 2)
    variant = home / "data/variants/baseline/real/index.db"
    assert hashes(variant) == hashes(home / "data/index.db")


@pytest.mark.parametrize(
    "changed",
    [
        "[variants.v]\nmin_words = 50",  # a setting that only the variant hash holds
        "[variants.v]\nmin_words = 1\ndimension = 4",  # a setting that the index check of connect() holds
    ],
)
def test_bench_index_refuses_an_index_of_other_variant_settings(tmp_path, monkeypatch, capsys, changed):
    home = bench_home(
        tmp_path,
        monkeypatch,
        PREFIX,
        '\n[variants.v]\nmin_words = 1\n\n[corpora.real]\nroot = "vaults"\nvaults = ["V"]\n',
    )
    assert run(capsys, "bench", "index", "--variant", "v", "--corpus", "real")[0] == 0
    text = (home / "config.toml").read_text().replace("[variants.v]\nmin_words = 1", changed)
    (home / "config.toml").write_text(text)
    code, out = run(capsys, "bench", "index", "--variant", "v", "--corpus", "real")
    assert code == 1
    # the way out of a variant index is to delete its folder, not the index_path advice of the default index
    assert f"Delete the folder {home / 'data/variants/v/real'}" in out.err
    assert "ragnarok bench index" in out.err
    assert "index_path" not in out.err


def test_bench_index_of_an_unknown_variant_fails(tmp_path, monkeypatch, capsys):
    bench_home(tmp_path, monkeypatch, PREFIX)
    code, out = run(capsys, "bench", "index", "--variant", "nope", "--corpus", "real")
    assert code == 1
    assert "nope" in out.err
