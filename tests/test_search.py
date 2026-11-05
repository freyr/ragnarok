import hashlib
import json
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


# The knowledge-base filters and the scent fields, on a synthetic Kb vault.

KB_NOTES = {
    "Kb/entities/Billing System.md": (
        "---\nid: sys-billing\ntype: entity\nkind: system\naliases: [Billing, Core]\n---\n"
        "# Billing System\nIt sends the invoice.\n"
    ),
    "Kb/entities/Loyalty Tiers.md": (
        "---\nid: feat-tiers\ntype: entity\nkind: feature\naliases: [Tiers, Core]\n---\n# Loyalty Tiers\nThe ladder.\n"
    ),
    "Kb/howto/deploy-billing.md": (
        "---\ntype: how-to\nsubject: '[[Billing]]'\ntask: deploy\ncanonical: true\nas_of: 2020-01-01\n"
        "review_by: 2021-01-01\ndispute: {owner: '[[Team Ops]]', due: 2026-12-01}\n---\n"
        "# Deploy billing\nDeploy the billing service with the pipeline.\n"
    ),
    "Kb/howto/deploy-tiers.md": (
        "---\ntype: how-to\nsubject: '[[Tiers]]'\ntask: deploy\n---\n# Deploy tiers\nDeploy the tier service.\n"
    ),
    "Kb/howto/deploy-billing-old.md": (
        "---\ntype: how-to\nsubject: '[[sys-billing]]'\ntask: deploy by hand\nstatus: redirect\n"
        "superseded_by: '[[deploy-billing]]'\n---\n# Deploy billing by hand\nDeploy the billing service by hand.\n"
    ),
    "Kb/decisions/deploy-choice.md": (
        "---\ntype: decision\nsubject: ['[[Billing]]', '[[Tiers]]']\nsuperseded_by: '[[deploy-billing]]'\n---\n"
        "# Deploy choice\nWe deploy the billing service and the tier service with one pipeline.\n"
    ),
}
BILLING = {"Kb/howto/deploy-billing.md", "Kb/decisions/deploy-choice.md"}
TIERS = {"Kb/howto/deploy-tiers.md", "Kb/decisions/deploy-choice.md"}
BILLING_DOC = "Kb/entities/Billing System.md"  # the entity documents: the subject filter admits them
TIERS_DOC = "Kb/entities/Loyalty Tiers.md"


@pytest.fixture
def kb_index(tmp_path):
    return kb_setup(tmp_path)


def kb_setup(tmp_path):
    from ragnarok.config import Config

    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=len(TopicEmbedder.TOPICS))
    for rel, text in KB_NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    return cfg, conn, emb


def doc_id(conn, path):
    return conn.execute("SELECT id FROM doc WHERE path = ?", (path,)).fetchone()[0]


def found(kb_index, query="deploy service", **filters):
    cfg, conn, emb = kb_index
    return set(paths(search(cfg, conn, emb, query, limit=50, **filters)))


def test_subject_filter_keeps_only_documents_of_the_subject(kb_index):
    assert "Kb/howto/deploy-tiers.md" in found(kb_index)
    assert found(kb_index, subject="sys-billing") == BILLING | {BILLING_DOC}
    assert found(kb_index, subject=["sys-billing"]) == BILLING | {BILLING_DOC}
    # a list is an OR
    assert found(kb_index, subject=["sys-billing", "feat-tiers"]) == BILLING | TIERS | {BILLING_DOC, TIERS_DOC}


def test_an_empty_list_filter_is_no_filter(kb_index):
    every = found(kb_index)
    for key in ("type", "collection", "class_", "subject", "status"):
        assert found(kb_index, **{key: []}) == every, key


def test_a_private_entity_resolves_as_a_subject_when_private_is_allowed(tmp_path):
    from ragnarok.config import Config

    cfg = Config(
        root=tmp_path,
        index_path=tmp_path / "index.db",
        vaults=["Kb"],
        dimension=len(TopicEmbedder.TOPICS),
        private_patterns=["Kb/entities/Billing System.md"],
        allow_private=True,
    )
    for rel, text in KB_NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    assert found((cfg, conn, emb), subject="Billing") == BILLING


def test_subject_name_resolves_through_aliases(kb_index):
    cfg, conn, emb = kb_index
    resolved = []
    hits = search(cfg, conn, emb, "deploy service", subject="  BILLING ", limit=50, resolved=resolved)
    assert set(paths(hits)) == BILLING | {BILLING_DOC}
    assert resolved == [{"name": "  BILLING ", "id": "sys-billing", "matched_by": "alias"}]
    resolved = []
    hits = search(cfg, conn, emb, "deploy service", subject=["Loyalty Tiers"], limit=50, resolved=resolved)
    assert set(paths(hits)) == TIERS | {TIERS_DOC}
    assert resolved == [{"name": "Loyalty Tiers", "id": "feat-tiers", "matched_by": "alias"}]


def test_ambiguous_subject_gives_the_candidates(kb_index):
    from ragnarok.tools import Tools
    from ragnarok.trace import trace_dir

    cfg, conn, emb = kb_index
    with pytest.raises(LookupError, match="ambiguous") as info:
        search(cfg, conn, emb, "deploy", subject="core")
    assert "feat-tiers" in str(info.value) and "sys-billing" in str(info.value)
    assert sorted(c["id"] for c in info.value.candidates) == ["feat-tiers", "sys-billing"]
    with pytest.raises(LookupError, match="No entity"):
        search(cfg, conn, emb, "deploy", subject="Payments")

    tools = Tools(cfg, conn, emb)
    with pytest.raises(LookupError):
        tools.search("deploy", subject=["sys-billing", "core"])
    (day,) = trace_dir(cfg).glob("*.jsonl")
    line = json.loads(day.read_text().splitlines()[-1])
    assert line["gen_ai.tool.name"] == "search"
    assert line["error.type"] == "QUERY"  # not a silent miss
    entities = {e["name"]: e for e in line["ragnarok.entities"]}
    assert entities["sys-billing"]["selected"] == "sys-billing"
    assert entities["core"]["selected"] is None
    assert sorted(c["id"] for c in entities["core"]["candidates"]) == ["feat-tiers", "sys-billing"]


def test_redirect_is_never_eligible(kb_index):
    cfg, conn, emb = kb_index
    old = "Kb/howto/deploy-billing-old.md"
    assert conn.execute("SELECT count(*) FROM unit WHERE doc_id = ?", (doc_id(conn, old),)).fetchone()[0] > 0
    for mode in ("hybrid", "dense", "fts"):
        assert old not in paths(search(cfg, conn, emb, "deploy billing service by hand", mode=mode, limit=50))
    assert found(kb_index, "deploy billing by hand", status="redirect") == set()
    everything = found(kb_index, "deploy billing by hand")
    assert found(kb_index, "deploy billing by hand", status=["redirect", "active"]) == everything
    assert old not in found(kb_index, "deploy billing by hand", subject="sys-billing", include_archived=True)


def test_type_status_class_and_collection_filters(kb_index):
    howtos = {"Kb/howto/deploy-billing.md", "Kb/howto/deploy-tiers.md"}
    assert found(kb_index, type="decision") == {"Kb/decisions/deploy-choice.md"}
    assert found(kb_index, type=["decision", "how-to"]) == howtos | {"Kb/decisions/deploy-choice.md"}
    assert found(kb_index, class_="record") == {"Kb/decisions/deploy-choice.md"}
    assert found(kb_index, class_="state", subject="Tiers") == {"Kb/howto/deploy-tiers.md", TIERS_DOC}
    assert found(kb_index, collection="Kb") == found(kb_index) and found(kb_index, collection="Other") == set()
    assert found(kb_index, status="active") == found(kb_index)
    assert found(kb_index, status=["archived", "superseded"]) == set()


def test_result_has_the_scent_fields(kb_index):
    from ragnarok.tools import Tools

    cfg, conn, emb = kb_index
    result = Tools(cfg, conn, emb).search("deploy billing service", subject="Billing", limit=10)
    assert result["trace_id"]
    assert result["subjects"] == [{"name": "Billing", "id": "sys-billing", "matched_by": "alias"}]
    by_path = {r["path"]: r for r in result["results"]}
    assert set(by_path) == BILLING | {BILLING_DOC}
    assert result["results"][0]["path"] == BILLING_DOC and result["results"][0]["entity_first"] is True
    current = by_path["Kb/howto/deploy-billing.md"]
    assert current["id"] == doc_id(conn, "Kb/howto/deploy-billing.md")
    assert (current["type"], current["class"], current["status"]) == ("how-to", "state", "active")
    assert current["subjects"] == [{"id": "sys-billing", "name": "Billing System"}]
    assert current["canonical"] is True
    assert current["as_of"] == "2020-01-01"
    assert "2021-01-01" in current["review_warning"]
    assert current["successor"] is None
    assert current["dispute"] == {"owner": "Team Ops", "due": "2026-12-01"}
    assert {"heading", "text", "score", "dense_rank", "fts_rank"} <= set(current)
    record = by_path["Kb/decisions/deploy-choice.md"]
    assert record["class"] == "record"
    assert record["subjects"] == [
        {"id": "sys-billing", "name": "Billing System"},
        {"id": "feat-tiers", "name": "Loyalty Tiers"},
    ]
    assert (record["canonical"], record["as_of"], record["review_warning"], record["dispute"]) == (
        False,
        None,
        None,
        None,
    )
    assert record["successor"] == {"id": current["id"], "path": "Kb/howto/deploy-billing.md"}


def test_stat_counts_docs_and_units_and_follows_the_update(kb_index):
    cfg, conn, emb = kb_index

    def stat(column, value):
        row = conn.execute('SELECT n_docs, n_units FROM stat WHERE "column" = ? AND value = ?', (column, value))
        return row.fetchone()

    def units(where):
        return conn.execute(f"SELECT count(*) FROM unit u JOIN doc d ON d.id = u.doc_id WHERE {where}").fetchone()[0]

    assert stat("type", "how-to") == (3, units("d.type = 'how-to'"))
    assert stat("class", "record") == (1, units("d.class = 'record'"))
    assert stat("status", "redirect") == (1, units("d.status = 'redirect'"))
    assert stat("subject", "sys-billing")[0] == 3
    assert stat("collection", "Kb")[0] == 6 and stat("sensitivity", "normal")[0] == 6
    (cfg.root / "Kb/howto/deploy-tiers.md").unlink()
    update(cfg, conn, emb)
    assert stat("type", "how-to")[0] == 2
    assert stat("subject", "feat-tiers")[0] == 1


def test_plan_reads_the_selectivity_from_stat(kb_index):
    from ragnarok.trace import Trace

    cfg, conn, emb = kb_index
    trace = Trace(cfg, conn, "search", {})
    search(cfg, conn, emb, "deploy", subject="sys-billing", type="how-to", trace=trace)
    plan = trace.fields["ragnarok.plan"]
    n_units = conn.execute("""SELECT n_units FROM stat WHERE "column" = 'subject' AND value = 'sys-billing'""")
    assert plan["selectivity"]["subject"]["units"] == n_units.fetchone()[0]
    n_units = conn.execute("""SELECT n_units FROM stat WHERE "column" = 'type' AND value = 'how-to'""")
    assert plan["selectivity"]["type"]["units"] == n_units.fetchone()[0]
    eligible = conn.execute(
        "SELECT count(*) FROM unit u JOIN doc d ON d.id = u.doc_id WHERE d.type = 'how-to' AND d.status != 'redirect'"
        " AND EXISTS (SELECT 1 FROM doc_subject s WHERE s.doc_id = d.id AND s.entity_id = 'sys-billing')"
    )
    assert plan["eligible_units"] == eligible.fetchone()[0]
    checks = {c["path"]: c["filters"] for c in trace.fields["ragnarok.candidates"]}
    assert checks == {"Kb/howto/deploy-billing.md": {"type": True, "subject": True, "private": True}}


def test_filter_checks_name_the_filter_that_drops_a_document():
    from ragnarok.search import Filters, filter_checks

    doc = {
        "type": "how-to",
        "class": "state",
        "status": "active",
        "sensitivity": "normal",
        "subjects": ["feat-tiers"],
        "entity": None,
    }
    filters = Filters(type=["how-to", "decision"], subject=("sys-billing",))
    assert filter_checks(doc, filters) == {"type": True, "subject": False, "private": True}
    redirect = {**doc, "status": "redirect"}
    assert filter_checks(redirect, Filters(class_="record")) == {"class": False, "private": True, "redirect": False}
    # the subject filter admits the entity document of a subject entity
    entity_doc = {**doc, "type": "entity", "subjects": [], "entity": "sys-billing"}
    assert filter_checks(entity_doc, Filters(subject=("sys-billing",)))["subject"] is True
    assert filter_checks(entity_doc, Filters(subject=("feat-tiers",)))["subject"] is False


# R1: the entity documents of the query subjects go first.

LEDGER_DOC = "Kb/entities/Ledger.md"


def first(hits) -> list[str]:
    return [h.path for h in hits if h.entity_first]


def test_an_alias_in_the_query_puts_its_entity_document_first(kb_index):
    cfg, conn, emb = kb_index
    hits = search(cfg, conn, emb, "deploy the tiers service", limit=50)
    assert hits[0].path == TIERS_DOC and hits[0].entity_first and hits[0].score is None
    assert first(hits) == [TIERS_DOC]
    assert paths(hits).count(TIERS_DOC) == 1
    # the entities of the text go first in the order of their first position in the query
    assert first(search(cfg, conn, emb, "billing then tiers", limit=50)) == [BILLING_DOC, TIERS_DOC]
    assert first(search(cfg, conn, emb, "tiers then billing", limit=50)) == [TIERS_DOC, BILLING_DOC]


def test_the_subject_filter_admits_the_entity_document_of_the_subject(kb_index):
    cfg, conn, emb = kb_index
    hits = search(cfg, conn, emb, "deploy service", subject="Tiers", limit=50)
    assert first(hits) == [TIERS_DOC] and hits[0].path == TIERS_DOC
    assert set(paths(hits)) == TIERS | {TIERS_DOC}
    # the subject entities go first in their order, before the entities of the text
    hits = search(cfg, conn, emb, "tiers and billing", subject=["sys-billing", "feat-tiers"], limit=50)
    assert first(hits) == [BILLING_DOC, TIERS_DOC]
    # a text entity that the subject filter does not admit gets no place
    hits = search(cfg, conn, emb, "deploy billing service", subject="Tiers", limit=50)
    assert first(hits) == [TIERS_DOC] and BILLING_DOC not in paths(hits)
    # with entity-first off, the subject filter still admits the entity document in the RRF candidates
    off = replace(cfg, kb=replace(cfg.kb, entity_first=False))
    hits = search(off, conn, emb, "loyalty tiers ladder", subject="Tiers", mode="fts", limit=50)
    assert TIERS_DOC in paths(hits) and first(hits) == []


def test_a_filter_that_does_not_admit_the_entity_document_keeps_the_normal_rank(kb_index):
    cfg, conn, emb = kb_index
    hits = search(cfg, conn, emb, "deploy the tiers service", type="how-to", limit=50)
    assert first(hits) == [] and TIERS_DOC not in paths(hits)
    hits = search(cfg, conn, emb, "deploy the tiers service", type=["how-to", "entity"], limit=50)
    assert first(hits) == [TIERS_DOC]


def test_at_most_three_entity_documents_go_first_and_the_limit_counts_them(kb_index):
    cfg, conn, emb = kb_index
    for name, alias in (("Ledger", "GL"), ("Payroll", "Pay Run")):
        (cfg.root / f"Kb/entities/{name}.md").write_text(
            f"---\nid: sys-{name.lower()}\ntype: entity\nkind: system\naliases: [{alias}]\n---\n# {name}\nA book.\n"
        )
    update(cfg, conn, emb)
    query = "pay run billing gl tiers"
    hits = search(cfg, conn, emb, query, limit=50)
    assert first(hits) == ["Kb/entities/Payroll.md", BILLING_DOC, LEDGER_DOC]
    assert paths(hits)[:3] == first(hits)
    assert TIERS_DOC in paths(hits)  # the fourth entity keeps its normal rank
    short = search(cfg, conn, emb, query, limit=2)
    assert paths(short) == ["Kb/entities/Payroll.md", BILLING_DOC]


def test_entity_first_is_off_with_the_knob_or_without_authority(kb_index):
    cfg, conn, emb = kb_index
    off = replace(cfg, kb=replace(cfg.kb, entity_first=False))
    for c, authority in ((off, True), (cfg, False)):
        hits = search(c, conn, emb, "deploy the tiers service", limit=50, authority=authority)
        assert first(hits) == []
        assert all(h.score is not None for h in hits)
    # the keyword of a caller turns it off as well
    assert first(search(cfg, conn, emb, "deploy the tiers service", limit=50, entity_first=False)) == []


def test_the_trace_records_the_entity_first_documents_at_their_final_rank(kb_index):
    from ragnarok.trace import Trace

    cfg, conn, emb = kb_index
    (cfg.root / LEDGER_DOC).write_text(
        "---\nid: sys-ledger\ntype: entity\nkind: system\naliases: [GL]\n---\n# Ledger\nA book.\n"
    )
    update(cfg, conn, emb)
    # fts mode: the ledger document has no word of the query, so it is outside the RRF list
    trace = Trace(cfg, conn, "search", {})
    hits = search(cfg, conn, emb, "deploy tiers gl", mode="fts", limit=50, trace=trace)
    assert first(hits) == [TIERS_DOC, LEDGER_DOC]
    ledger = hits[1]
    assert (ledger.dense_rank, ledger.fts_rank, ledger.score, ledger.heading) == (None, None, None, "Ledger > Ledger")
    f = trace.fields
    assert f["ragnarok.entity_first"] == [
        {"id": "feat-tiers", "doc_id": doc_id(conn, TIERS_DOC), "reason": "text"},
        {"id": "sys-ledger", "doc_id": doc_id(conn, LEDGER_DOC), "reason": "text"},
    ]
    documents = f["gen_ai.retrieval.documents"]
    assert [d["path"] for d in documents] == paths(hits)
    assert [d["score"] for d in documents[:2]] == [None, None]
    assert all(d["score"] is not None for d in documents[2:])
    assert [s["id"] for s in f["ragnarok.returned.status"]] == [h.id for h in hits]
    candidates = {c["path"]: c for c in f["ragnarok.candidates"]}
    assert (candidates[TIERS_DOC]["final_rank"], candidates[TIERS_DOC]["entity_first"]) == (1, True)
    assert (candidates[LEDGER_DOC]["final_rank"], candidates[LEDGER_DOC]["rrf"]) == (2, None)
    assert "entity_first" not in candidates["Kb/howto/deploy-tiers.md"]
    # the subject flag of the trace admits the entity document of a subject entity
    trace = Trace(cfg, conn, "search", {})
    hits = search(cfg, conn, emb, "deploy tiers gl", subject="Tiers", mode="fts", limit=50, trace=trace)
    assert first(hits) == [TIERS_DOC]  # the subject filter does not admit the ledger
    checks = {c["path"]: c["filters"] for c in trace.fields["ragnarok.candidates"]}
    assert checks[TIERS_DOC]["subject"] is True
    assert trace.fields["ragnarok.entity_first"] == [
        {"id": "feat-tiers", "doc_id": doc_id(conn, TIERS_DOC), "reason": "subject"}
    ]
