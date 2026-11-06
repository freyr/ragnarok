"""The authority factors of the rank stage (law L1), on a synthetic Kb vault."""

import json
from dataclasses import replace

import pytest
from test_search import TopicEmbedder

from ragnarok.config import AuthorityConfig, Config, KbConfig
from ragnarok.index import connect, update
from ragnarok.search import search
from ragnarok.tools import Tools
from ragnarok.trace import trace_dir


def entity(id_, name, aliases, extra=""):
    return f"---\nid: {id_}\ntype: entity\nkind: system\naliases: [{aliases}]\n{extra}---\n# {name}\nThe {name}.\n"


def howto(subject, task, extra, body):
    return f"---\ntype: how-to\nsubject: '[[{subject}]]'\ntask: {task}\n{extra}---\n# {task}\n{body}\n"


STRONG = "Deploy billing service."
WEAK = "Deploy the billing service with the pipeline, the review and the long list of checks of the team."

NOTES = {
    "Kb/entities/Billing System.md": entity("sys-billing", "Billing System", "Billing"),
    "Kb/entities/Ledger.md": entity("sys-ledger", "Ledger", "General Ledger"),
    # the same identity key (subject and task): the canonical one is older and weaker than the other
    "Kb/howto/deploy-billing.md": howto("Billing", "deploy", "canonical: true\nas_of: 2020-01-01\n", WEAK),
    "Kb/howto/deploy-billing-copy.md": howto("Billing", "deploy", "as_of: 2025-01-01\n", STRONG),
    # a state document with a review date in the past
    "Kb/howto/restart-billing.md": howto(
        "Billing", "restart", "review_by: 2021-01-01\n", "Restart the billing service."
    ),
    "Kb/decisions/old-choice.md": (
        "---\ntype: decision\nsubject: '[[Billing]]'\nsuperseded_by: '[[deploy-billing]]'\n---\n"
        "# Old choice\nRestart the billing service by hand.\n"
    ),
    "Kb/howto/archive-billing.md": howto("Billing", "archive", "status: archived\n", "Archive the billing service."),
    "Kb/howto/close-ledger.md": howto("General Ledger", "close", "", "Close the ledger of the billing service."),
}
DEPLOY = "Kb/howto/deploy-billing.md"
COPY = "Kb/howto/deploy-billing-copy.md"
Q = "deploy billing service"


@pytest.fixture
def kb(tmp_path):
    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=len(TopicEmbedder.TOPICS))
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    return cfg, conn, emb


def hits(kb, query=Q, mode="fts", **kwargs):
    cfg, conn, emb = kb
    return {h.path: h for h in search(cfg, conn, emb, query, limit=50, mode=mode, **kwargs)}


def test_authority_off_gives_the_rrf_order(kb):
    off = hits(kb, authority=False)
    order = list(off)
    assert order.index(COPY) < order.index(DEPLOY)  # the strong duplicate leads on RRF alone
    scores = [h.score for h in off.values()]
    assert scores == sorted(scores, reverse=True)


def test_canonical_document_moves_above_a_duplicate(kb):
    assert list(hits(kb)).index(DEPLOY) < list(hits(kb)).index(COPY)
    # the canonical document of a subject of the query gets the canonical factor
    on, off = hits(kb, subject="Billing"), hits(kb, subject="Billing", authority=False)
    assert on[DEPLOY].score == pytest.approx(off[DEPLOY].score * 1.5, abs=1e-5)
    assert on[COPY].score == pytest.approx(off[COPY].score * 0.5, abs=1e-5)
    assert on[COPY].duplicate and not on[DEPLOY].duplicate


def duplicates(kb):
    return [p for p, h in hits(kb).items() if h.duplicate]


def test_first_of_an_identity_key_is_the_canonical_one(kb):
    cfg, conn, emb = kb
    assert duplicates(kb) == [COPY]  # the canonical flag beats the later as_of
    # without a canonical flag, the latest as_of is first
    (cfg.root / DEPLOY).write_text(howto("Billing", "deploy", "as_of: 2020-01-01\n", WEAK))
    update(cfg, conn, emb)
    assert duplicates(kb) == [DEPLOY]
    # with the same as_of, the latest file time is first
    for path in (DEPLOY, COPY):
        (cfg.root / path).write_text(howto("Billing", "deploy", "as_of: 2020-01-01\n", WEAK + path))
        update(cfg, conn, emb)
    assert duplicates(kb) == [DEPLOY]  # COPY was written last


def test_stale_state_document_gets_a_warning(kb):
    restart = "Kb/howto/restart-billing.md"
    on, off = hits(kb), hits(kb, authority=False)
    assert "2021-01-01" in on[restart].review_warning
    assert on[restart].score == pytest.approx(off[restart].score * 0.8, abs=1e-5)


def test_superseded_record_gives_its_successor(kb):
    old = "Kb/decisions/old-choice.md"
    on, off = hits(kb), hits(kb, authority=False)
    assert on[old].successor["path"] == DEPLOY
    # no superseded factor: a record gets the record factor of a `now` question (spec C2), 0.95
    assert on[old].score == pytest.approx(off[old].score * 0.95, abs=1e-5)


def test_a_superseded_state_note_gets_its_validity_factor_and_a_note_with_no_kind_none(kb):
    cfg, conn, emb = kb
    (cfg.root / "Kb/howto/old-state.md").write_text(
        howto("Billing", "retire", "superseded_by: '[[deploy-billing]]'\n", "Retire the billing service.")
    )
    (cfg.root / "Kb/notes").mkdir()
    (cfg.root / "Kb/notes/old-note.md").write_text(
        "---\ntype: note\nsuperseded_by: '[[deploy-billing]]'\n---\n# Old note\nRetire the billing service.\n"
    )
    update(cfg, conn, emb)
    classes = dict(conn.execute("SELECT path, class FROM doc WHERE path LIKE '%old-%'").fetchall())
    assert sorted(classes.values()) == ["none", "record", "state"]
    query = "retire billing service"
    on, off = hits(kb, query), hits(kb, query, authority=False)
    state, note = "Kb/howto/old-state.md", "Kb/notes/old-note.md"
    assert on[state].score == pytest.approx(off[state].score * 0.8, abs=1e-5)  # state, superseded: 0.8 for now
    assert on[note].score == pytest.approx(off[note].score, abs=1e-5)  # kind none: 1.0
    for path in (state, note):
        assert on[path].successor["path"] == DEPLOY
    history = hits(kb, query, when="history")
    assert history[state].score == pytest.approx(off[state].score, abs=1e-5)  # 1.0 for history


def test_an_archived_document_gets_no_factor(kb):
    archived = "Kb/howto/archive-billing.md"
    on, off = hits(kb), hits(kb, authority=False)
    assert on[archived].score == pytest.approx(off[archived].score, abs=1e-5)


def test_subjects_in_the_query_text_count_only_when_the_knob_is_on(kb):
    cfg, conn, emb = kb
    ledger = "Kb/howto/close-ledger.md"
    (cfg.root / "Kb/entities/Ledger.md").write_text(
        entity("sys-ledger", "Ledger", "General Ledger", "canonical:\n  close: '[[close-ledger]]'\n")
    )
    update(cfg, conn, emb)
    query = "close the general ledger billing service"
    base = hits(kb, query, authority=False)[ledger].score
    assert hits(kb, query)[ledger].score == pytest.approx(base, abs=1e-5)
    knob = replace(cfg, kb=KbConfig(authority=AuthorityConfig(subject_from_text=True)))
    found = {h.path: h for h in search(knob, conn, emb, query, limit=50, mode="fts")}
    assert found[ledger].score == pytest.approx(base * 1.5, abs=1e-5)


def test_the_trace_records_each_factor_of_each_candidate(kb):
    cfg, conn, emb = kb
    trace_id = Tools(cfg, conn, emb).search(Q, subject="Billing")["trace_id"]
    lines = [json.loads(x) for f in trace_dir(cfg).glob("*.jsonl") for x in f.read_text().splitlines()]
    (entry,) = [x for x in lines if x["trace_id"] == trace_id]
    factors = {c["path"]: c["authority"] for c in entry["ragnarok.candidates"]}
    assert factors[DEPLOY] == {"canonical": 1.5, "when": 1.0}
    assert factors[COPY] == {"duplicate_identity": 0.5, "when": 1.0}
    assert factors["Kb/decisions/old-choice.md"] == {"when": 0.95}
    assert factors["Kb/howto/restart-billing.md"] == {"review_overdue": 0.8, "when": 1.0}
    assert entry["ragnarok.when"] == {"kind": "now", "source": "rule", "matched": None}


# the kind of question (spec C2): synthetic notes of each kind with near relevance to one query
def spec(name, extra, body):
    return f"---\ntype: spec\n{extra}---\n# {name}\n{body}\n"


WHEN_NOTES = {
    "Kb/entities/Vault.md": entity("sys-vault", "Vault", "Key Vault", "canonical:\n  rotate: '[[rotate-keys]]'\n"),
    # the state note matches best; the proposal and the record are a little longer, so a little weaker
    "Kb/howto/rotate-keys.md": howto("Key Vault", "rotate", "", "Rotate the vault keys."),
    "Kb/specs/rotate-keys-spec.md": spec("Rotation", "", "Rotate the vault keys with a new tool."),
    "Kb/decisions/rotate-keys-record.md": (
        "---\ntype: decision\n---\n# Rotation record\nRotate the vault keys with a new tool and a list of steps.\n"
    ),
    "Kb/howto/archived-rotate.md": howto("Key Vault", "rotate-old", "status: archived\n", "Rotate vault keys by hand."),
}
STATE, PROPOSAL, RECORD = (
    "Kb/howto/rotate-keys.md",
    "Kb/specs/rotate-keys-spec.md",
    "Kb/decisions/rotate-keys-record.md",
)
VAULT_DOC, ARCHIVED = "Kb/entities/Vault.md", "Kb/howto/archived-rotate.md"
WQ = "rotate the vault keys"


@pytest.fixture
def wkb(tmp_path):
    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=len(TopicEmbedder.TOPICS))
    for rel, text in WHEN_NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    return cfg, conn, emb


def ranked(wkb, query=WQ, **kwargs):
    cfg, conn, emb = wkb
    return [h.path for h in search(cfg, conn, emb, query, limit=50, mode="fts", entity_first=False, **kwargs)]


@pytest.mark.parametrize(
    "query, kind",
    [(f"how do we {w} rotate keys", "future") for w in ("improve", "better", "plan", "next", "upgrade")]
    + [(f"rotate keys {w}", "future") for w in ("should", "could", "proposal", "would", "IMPROVE")]
    + [(f"{w} rotate keys", "history") for w in ("why did we", "when did we", "what was the")]
    + [(f"rotate keys {w}", "history") for w in ("before", "previously", "history", "decided", "originally")]
    + [
        ("rotate the vault keys", "now"),
        ("the planner rotates keys", "now"),  # a word inside another word is no match
        ("what was the plan", "history"),  # both lists match: history wins
    ],
)
def test_each_rule_word_gives_its_kind_of_question(query, kind):
    from ragnarok.search import question_kind

    assert question_kind(Config(), query)[:2] == (kind, "rule")


def test_the_when_parameter_overrides_the_rule():
    from ragnarok.search import question_kind

    assert question_kind(Config(), "why did we plan this", "now") == ("now", "parameter", None)
    assert question_kind(Config(), "rotate keys", "future") == ("future", "parameter", None)
    with pytest.raises(ValueError):
        question_kind(Config(), "rotate keys", "later")


def test_the_rule_list_is_configuration():
    from ragnarok.search import question_kind

    cfg = replace(Config(), kb=replace(Config().kb, when_rules={"future": ["roadmap"], "history": []}))
    assert question_kind(cfg, "the roadmap of keys") == ("future", "rule", "roadmap")
    assert question_kind(cfg, "why did we plan this") == ("now", "rule", None)


def test_the_factor_table_is_configuration(wkb):
    cfg, conn, emb = wkb
    table = {"proposal_open": {"now": 1.0, "future": 1.0, "history": 1.0}}
    flat = replace(cfg, kb=replace(cfg.kb, when_factors=table))
    found = [h.path for h in search(flat, conn, emb, WQ, limit=50, mode="fts", entity_first=False, when="future")]
    assert found.index(STATE) < found.index(PROPOSAL)


def test_relevance_on_rrf_alone_puts_the_state_note_first(wkb):
    order = ranked(wkb, authority=False)
    assert order.index(STATE) < order.index(PROPOSAL) < order.index(RECORD)


def test_an_open_proposal_ranks_above_the_state_note_only_for_future(wkb):
    future, now = ranked(wkb, when="future"), ranked(wkb, when="now")
    assert future.index(PROPOSAL) < future.index(STATE)
    assert now.index(STATE) < now.index(PROPOSAL)
    # with no `when`, a future word in the query decides the same way
    upgrade = ranked(wkb, WQ + " next")
    assert upgrade.index(PROPOSAL) < upgrade.index(STATE)
    assert ranked(wkb).index(STATE) < ranked(wkb).index(PROPOSAL)


def test_a_record_ranks_first_for_history(wkb):
    assert ranked(wkb, when="history")[0] == RECORD
    now = ranked(wkb, when="now")
    assert now.index(STATE) < now.index(RECORD)


def test_the_entity_document_goes_first_only_for_now(wkb):
    cfg, conn, emb = wkb
    for when, first in (("now", True), ("future", False), ("history", False)):
        found = search(cfg, conn, emb, WQ, limit=50, mode="fts", when=when)
        assert any(h.entity_first for h in found) is first
        assert (found[0].path == VAULT_DOC) is first


def test_the_canonical_factor_applies_only_for_now(wkb):
    cfg, conn, emb = wkb
    off = {h.path: h.score for h in search(cfg, conn, emb, WQ, limit=50, mode="fts", authority=False)}
    for when, factor in (("now", 1.5), ("future", 1.0), ("history", 1.0)):
        found = {
            h.path: h.score
            for h in search(
                cfg, conn, emb, WQ, limit=50, mode="fts", subject="Key Vault", entity_first=False, when=when
            )
        }
        assert found[STATE] == pytest.approx(off[STATE] * factor, abs=1e-5)


def test_an_archived_note_gets_no_factor_of_its_own(wkb):
    cfg, conn, emb = wkb
    off = {h.path: h.score for h in search(cfg, conn, emb, WQ, limit=50, mode="fts", authority=False)}
    for when in ("now", "future", "history"):  # an archived state note, current: 1.0 in each column
        found = {h.path: h.score for h in search(cfg, conn, emb, WQ, limit=50, mode="fts", when=when)}
        assert found[ARCHIVED] == pytest.approx(off[ARCHIVED], abs=1e-5)


def traced(cfg, trace_id):
    lines = [json.loads(x) for f in trace_dir(cfg).glob("*.jsonl") for x in f.read_text().splitlines()]
    (entry,) = [x for x in lines if x["trace_id"] == trace_id]
    return entry


@pytest.mark.parametrize(
    "extra, path, factors",
    [
        ("", STATE, (1.0, 1.0, 1.0)),
        ("validity: superseded\n", STATE, (0.8, 0.8, 1.0)),
        ("", PROPOSAL, (0.9, 1.15, 1.0)),
        ("validity: accepted\n", PROPOSAL, (0.9, 0.95, 1.0)),
        ("validity: rejected\n", PROPOSAL, (0.8, 0.8, 1.0)),
        ("validity: superseded\n", PROPOSAL, (0.8, 0.8, 1.0)),
        ("", RECORD, (0.95, 0.95, 1.15)),
        ("class: none\n", RECORD, (1.0, 1.0, 1.0)),
    ],
)
def test_the_trace_records_the_kind_of_question_and_the_factor_of_each_result(wkb, extra, path, factors):
    cfg, conn, emb = wkb
    text = (cfg.root / path).read_text()
    (cfg.root / path).write_text(text.replace("---\n", "---\n" + extra, 1))
    update(cfg, conn, emb)
    tools = Tools(cfg, conn, emb)
    for when, factor in zip(("now", "future", "history"), factors, strict=True):
        entry = traced(cfg, tools.search(WQ, when=when)["trace_id"])
        assert entry["ragnarok.when"] == {"kind": when, "source": "parameter", "matched": None}
        assert {c["path"]: c["authority"] for c in entry["ragnarok.candidates"]}[path]["when"] == factor
    # with no `when`, the source is the rule and the trace gives the rule word
    entry = traced(cfg, tools.search(WQ + " previously")["trace_id"])
    assert entry["ragnarok.when"] == {"kind": "history", "source": "rule", "matched": "previously"}


FACT_NOTES = {
    "Kb/entities/Heating.md": entity(
        "sys-heating", "Heating", "CH, Pump Room", "facts:\n  - attribute: pump_off_delay_min\n    value: 5\n"
    ),
    "Kb/entities/Boiler.md": entity(
        "dev-boiler", "Boiler", "Neodens", "facts:\n  - attribute: max_flow_temperature\n    value: 65 C\n"
    ),
    "Kb/howto/heating-sensor.md": howto("Heating", "sensor", "", "The outdoor sensor model of the heating."),
}


@pytest.fixture
def fkb(tmp_path):
    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=len(TopicEmbedder.TOPICS))
    for rel, text in FACT_NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    return cfg, conn, emb


@pytest.mark.parametrize(
    "query, first",
    [
        ("what is the pump delay of the heating", ["Kb/entities/Heating.md"]),
        ("the CH pump off delay", ["Kb/entities/Heating.md"]),
        # the entity is named, but no fact of it matches the query
        ("which outdoor sensor model does the heating use", []),
        ("the max flow temperature of the CH", []),
        # one shared word of a longer attribute is not enough: two words, or all words, of it are
        ("the heating delay", []),
        ("the heating off delay", ["Kb/entities/Heating.md"]),
        # only the alias phrase that the query holds is removed: `pump` of the alias `pump room` stays
        ("the pump off time of the CH", ["Kb/entities/Heating.md"]),
        # a light stem: `delayed` matches `delay`
        ("is the CH pump delayed", ["Kb/entities/Heating.md"]),
        # a value word of a fact matches too
        ("is the Neodens at 65", ["Kb/entities/Boiler.md"]),
    ],
)
def test_an_entity_goes_first_only_when_a_fact_of_it_matches_the_query(fkb, query, first):
    cfg, conn, emb = fkb
    found = search(cfg, conn, emb, query, limit=50, mode="fts", when="now")
    assert [h.path for h in found if h.entity_first] == first
