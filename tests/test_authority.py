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
    assert on[old].score == pytest.approx(off[old].score * 0.5, abs=1e-5)


def test_an_archived_document_is_halved(kb):
    archived = "Kb/howto/archive-billing.md"
    on, off = hits(kb), hits(kb, authority=False)
    assert on[archived].score == pytest.approx(off[archived].score * 0.5, abs=1e-5)


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
    assert factors[DEPLOY] == {"canonical": 1.5}
    assert factors[COPY] == {"duplicate_identity": 0.5}
    assert factors["Kb/decisions/old-choice.md"] == {"superseded": 0.5}
    assert factors["Kb/howto/restart-billing.md"] == {"review_overdue": 0.8}
