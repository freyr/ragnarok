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
    assert found(kb_index, subject="sys-billing") == BILLING
    assert found(kb_index, subject=["sys-billing"]) == BILLING
    assert found(kb_index, subject=["sys-billing", "feat-tiers"]) == BILLING | TIERS  # a list is an OR


def test_subject_name_resolves_through_aliases(kb_index):
    cfg, conn, emb = kb_index
    resolved = []
    hits = search(cfg, conn, emb, "deploy service", subject="  BILLING ", limit=50, resolved=resolved)
    assert set(paths(hits)) == BILLING
    assert resolved == [{"name": "  BILLING ", "id": "sys-billing", "matched_by": "alias"}]
    resolved = []
    hits = search(cfg, conn, emb, "deploy service", subject=["Loyalty Tiers"], limit=50, resolved=resolved)
    assert set(paths(hits)) == TIERS
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
    assert found(kb_index, class_="state", subject="Tiers") == {"Kb/howto/deploy-tiers.md"}
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
    assert set(by_path) == BILLING
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

    doc = {"type": "how-to", "class": "state", "status": "active", "sensitivity": "normal", "subjects": ["feat-tiers"]}
    filters = Filters(type=["how-to", "decision"], subject=("sys-billing",))
    assert filter_checks(doc, filters) == {"type": True, "subject": False, "private": True}
    redirect = {**doc, "status": "redirect"}
    assert filter_checks(redirect, Filters(class_="record")) == {"class": False, "private": True, "redirect": False}
