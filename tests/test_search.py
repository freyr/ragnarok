import hashlib
from dataclasses import replace

import pytest

from ragnarok.index import connect, update
from ragnarok.search import PrivateNotAllowed, fts_query, search


class TopicEmbedder:
    """Vectors from word overlap with a few topics, so that dense search is meaningful in tests."""

    TOPICS = ["boiler", "eyepiece", "password", "database", "queue", "budget", "person", "incident"]

    def embed_docs(self, items):
        return [self.vector(h + " " + t) for h, t in items]

    def embed_query(self, query):
        return self.vector(query)

    def vector(self, text):
        text = text.lower()
        v = [1.0 if t in text else 0.0 for t in self.TOPICS]
        return v if any(v) else [hashlib.sha256(text.encode()).digest()[0] / 255 + 0.01] * len(v)


EXTRA = {
    "Home/Heating/notes/boiler.md": "# Boiler\nThe boiler flow temperature follows the weather curve.\n",
    "Home/Astro/Theory/Field Stop.md": "# Field stop\nThe eyepiece field stop sets the true field.\n",
    "Home/Budget/2026.md": "# Budget\nThe boiler replacement is in the budget for 2026.\n",
}


@pytest.fixture
def index(vault):
    for rel, text in EXTRA.items():
        (vault.root / rel).write_text(text)
    vault.dimension = len(TopicEmbedder.TOPICS)
    conn = connect(vault)
    emb = TopicEmbedder()
    update(vault, conn, emb)
    return vault, conn, emb


def paths(hits):
    return [h.path for h in hits]


def test_fts_query_drops_stop_words():
    assert fts_query("What is the field stop of my eyepiece?") == '"field" OR "stop" OR "eyepiece"'
    assert fts_query("the of a") == ""


@pytest.mark.parametrize("mode", ["hybrid", "dense", "fts"])
def test_best_note_first(index, mode):
    cfg, conn, emb = index
    hits = search(cfg, conn, emb, "eyepiece field stop", mode=mode)
    assert hits[0].path == "Home/Astro/Theory/Field Stop.md"
    assert hits[0].heading.startswith("Field Stop")


def test_private_notes_are_excluded(index):
    cfg, conn, emb = index
    assert "Home/Budget/2026.md" not in paths(search(cfg, conn, emb, "boiler budget", limit=50))


def test_include_private_needs_the_config_gate(index):
    cfg, conn, emb = index
    with pytest.raises(PrivateNotAllowed):
        search(cfg, conn, emb, "boiler budget", include_private=True)
    allowed = replace(cfg, allow_private=True)
    assert "Home/Budget/2026.md" in paths(search(allowed, conn, emb, "boiler budget", include_private=True))


def test_filters(index):
    cfg, conn, emb = index
    hits = search(cfg, conn, emb, "boiler", vault="Work", limit=50)
    assert hits and all(h.vault == "Work" for h in hits)
    hits = search(cfg, conn, emb, "boiler", project="Heating", limit=50)
    assert hits and all(h.project == "Heating" for h in hits)
    hits = search(cfg, conn, emb, "old Q7", include_archived=False, limit=50)
    assert all(h.status == "active" for h in hits)


def test_hit_has_ranks_and_snippet(index):
    cfg, conn, emb = index
    hit = search(cfg, conn, emb, "boiler flow temperature")[0]
    assert hit.path == "Home/Heating/notes/boiler.md"
    assert hit.dense_rank == 1 and hit.fts_rank == 1
    assert "weather curve" in hit.text
