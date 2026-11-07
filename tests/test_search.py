import hashlib
from dataclasses import replace

import pytest

from ragnarok import search as search_module
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


@pytest.fixture
def labeled(vault):
    notes = {
        "Home/Heating/notes/boiler.md": "---\nclass: state\nvalidity: current\nas_of: 2026-11-01\n"
        "subject:\n  - '[[pump]]'\n  - '[[valve|v]]'\n---\n"
        "# Boiler\nThe boiler flow temperature is 45 degrees.\n",
        "Home/Heating/notes/boiler-2025.md": "---\nclass: state\nsuperseded_by: '[[boiler]]'\n---\n"
        "# Boiler 2025\nThe boiler flow temperature was 55 degrees.\n",
        "Home/Heating/notes/boiler-heat-pump.md": "---\nclass: proposal\nvalidity: rejected\n---\n"
        "# Boiler swap\nReplace the boiler with a heat pump.\n",
    }
    for rel, text in notes.items():
        (vault.root / rel).write_text(text)
    vault.dimension = len(TopicEmbedder.TOPICS)
    conn = connect(vault)
    emb = TopicEmbedder()
    update(vault, conn, emb)
    return vault, conn, emb


def test_superseded_and_rejected_notes_are_hidden_by_default(labeled):
    cfg, conn, emb = labeled
    found = paths(search(cfg, conn, emb, "boiler flow temperature", limit=50))
    assert "Home/Heating/notes/boiler.md" in found
    assert "Home/Heating/notes/boiler-2025.md" not in found
    assert "Home/Heating/notes/boiler-heat-pump.md" not in found


def test_include_superseded_shows_them(labeled):
    cfg, conn, emb = labeled
    found = paths(search(cfg, conn, emb, "boiler flow temperature", include_superseded=True, limit=50))
    assert {"Home/Heating/notes/boiler-2025.md", "Home/Heating/notes/boiler-heat-pump.md"} <= set(found)


def test_hit_shows_the_kb_labels(labeled):
    cfg, conn, emb = labeled
    hits = {h.path: h.as_dict() for h in search(cfg, conn, emb, "boiler", include_superseded=True, limit=50)}
    current = hits["Home/Heating/notes/boiler.md"]
    assert (current["class"], current["validity"], current["as_of"]) == ("state", "current", "2026-11-01")
    assert current["subject"] == ["pump", "valve"]
    old = hits["Home/Heating/notes/boiler-2025.md"]
    assert "subject" not in old
    assert (old["validity"], old["superseded_by"]) == ("superseded", "boiler")
    unlabeled = next(
        h for p, h in hits.items() if h["type"] == "note" and not p.startswith("Home/Heating/notes/boiler")
    )
    assert not {"class", "validity", "as_of", "superseded_by", "subject"} & unlabeled.keys()


BOILER, PUMP_2025, MOUNT = (
    "Home/Heating/notes/boiler.md",
    "Home/Heating/notes/pump-2025.md",
    "Home/Astro/Theory/mount.md",
)


@pytest.fixture
def subjects(labeled):
    """The labeled notes, an entity note Pump (alias circulator) and two more notes about it."""
    cfg, conn, emb = labeled
    notes = {
        "Home/Entities/Pump.md": "---\ntype: entity\nkind: device\naliases: [circulator]\n---\n# Pump\n",
        PUMP_2025: "---\nclass: state\nvalidity: superseded\nsubject: '[[circulator]]'\n---\n"
        "# Pump 2025\nThe boiler pump ran at speed 2.\n",
        MOUNT: "---\nsubject: '[[Home/Entities/Pump]]'\n---\n# Mount\nThe boiler room holds the mount.\n",
    }
    for rel, text in notes.items():
        (cfg.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (cfg.root / rel).write_text(text)
    update(cfg, conn, emb)
    return cfg, conn, emb


@pytest.mark.parametrize("value", ["Pump", "pump", "circulator", "CIRCULATOR", "Home/Entities/Pump.md", "[[Pump]]"])
def test_subject_filter_by_name_alias_or_path(subjects, value):
    cfg, conn, emb = subjects
    assert set(paths(search(cfg, conn, emb, "boiler", subject=value, limit=50))) == {BOILER, MOUNT}


def test_subject_filter_combines_with_the_other_filters(subjects):
    cfg, conn, emb = subjects
    found = paths(search(cfg, conn, emb, "boiler", subject="pump", include_superseded=True, limit=50))
    assert set(found) == {BOILER, PUMP_2025, MOUNT}
    found = paths(search(cfg, conn, emb, "boiler", subject="pump", project="Heating", limit=50))
    assert found == [BOILER]


@pytest.mark.parametrize("value", ["nobody", "valve"])  # valve is a subject target, but no entity note
def test_subject_filter_with_an_unknown_entity_gives_no_results(subjects, value):
    cfg, conn, emb = subjects
    assert search(cfg, conn, emb, "boiler", subject=value, limit=50) == []


def test_subject_filter_finds_a_note_below_the_unfiltered_depth(subjects, monkeypatch):
    cfg, conn, emb = subjects
    monkeypatch.setattr(search_module, "CANDIDATES", 1)
    monkeypatch.setattr(search_module, "FILTERED_CANDIDATES", 1)
    query = dict(query="boiler swap heat pump", include_superseded=True, mode="fts")
    assert paths(search(cfg, conn, emb, **query)) == ["Home/Heating/notes/boiler-heat-pump.md"]
    found = paths(search(cfg, conn, emb, subject="pump", **query))  # the depth applies to the notes about pump
    assert len(found) == 1 and found[0] in {BOILER, PUMP_2025, MOUNT}
