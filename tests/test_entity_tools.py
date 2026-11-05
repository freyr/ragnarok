"""find_entity, get_entity and the entity_fts rows, on a synthetic vault."""

import json

import pytest

from ragnarok.config import AttributeDef, Config, KbConfig
from ragnarok.index import connect, rebuild, update
from ragnarok.tools import Tools
from ragnarok.trace import find_line

NOTES = {
    "Kb/entities/Stage.md": (
        "---\nid: env-stage\ntype: entity\nkind: environment\naliases: [Staging, STG]\n"
        "facts:\n"
        "  - {attribute: aws_account, value: '1234', valid_until: 2026-06-30}\n"
        "  - {attribute: aws_account, value: '5678', valid_from: 2026-07-01}\n"
        "  - {attribute: monthly_cost, value: LEAKED-VALUE, source: 'cost-explorer:1234'}\n"
        "partOf: '[[Billing System]]'\n---\n# Stage\nThe stage environment.\n"
    ),
    "Kb/entities/Billing System.md": (
        "---\nid: sys-billing\ntype: entity\nkind: system\naliases: [Billing, Core]\n"
        "canonical: {deploy: '[[how-to-deploy-billing]]'}\n---\n# Billing System\nIt sends the invoice.\n"
    ),
    "Kb/entities/Billing Reports.md": (
        "---\nid: sys-reports\ntype: entity\nkind: system\n---\n# Billing Reports\nIt shows the invoice report.\n"
    ),
    "Kb/entities/Loyalty Tiers.md": (
        "---\nid: feat-tiers\ntype: entity\nkind: feature\naliases: [Tiers, Core]\n---\n"
        "# Loyalty Tiers\nThe tier ladder.\n"
    ),
    "Kb/entities/Payment Gateway.md": (
        "---\nid: sys-gateway\ntype: entity\nkind: system\n---\n# Payment Gateway\nIt does the settlement of cards.\n"
    ),
    "Kb/howto/how-to-deploy-billing.md": (
        "---\ntype: how-to\nsubject: '[[Billing]]'\ntask: Deploy the service\ncanonical: true\n---\n"
        "# Deploy\nDeploy it.\n"
    ),
}


class TopicEmbedder:
    """Vectors from the topic words of a text, so that the dense scan is meaningful."""

    TOPICS = ["settlement", "invoice", "tier", "stage"]

    def embed_docs(self, items):
        return [self.vector(h + " " + t) for h, t in items]

    def embed_query(self, query):
        return self.vector(query)

    def vector(self, text):
        v = [1.0 if t in text.lower() else 0.0 for t in self.TOPICS]
        return v if any(v) else [0.5] * len(v)


def put(cfg, rel, text):
    path = cfg.root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def cfg(tmp_path):
    attributes = {
        "environment": {
            "aws_account": AttributeDef(),
            "monthly_cost": AttributeDef(live=True, system="cost-explorer", skill="aws-cost"),
        }
    }
    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=len(TopicEmbedder.TOPICS),
                 kb=KbConfig(attributes=attributes))  # fmt: skip
    for rel, text in NOTES.items():
        put(cfg, rel, text)
    return cfg


@pytest.fixture
def tools(cfg):
    conn = connect(cfg)
    emb = TopicEmbedder()
    update(cfg, conn, emb)
    return Tools(cfg, conn, emb)


def ids(result):
    return [r["id"] for r in result["results"]]


def test_find_entity_by_alias_first(tools):
    # "Billing" is an alias of the system, and a word of the name of another entity: the alias wins alone
    result = tools.find_entity("  BILLING ")
    assert ids(result) == ["sys-billing"]
    hit = result["results"][0]
    assert (hit["kind"], hit["name"], hit["matched_by"]) == ("system", "Billing System", "alias")
    assert hit["aliases"] == ["Billing System", "Billing", "Core"]
    assert {c["aspect"]: c["path"] for c in hit["canonical"]}["deploy"] == "Kb/howto/how-to-deploy-billing.md"
    assert hit["score"] > 0
    # no alias: the words of the name
    by_name = tools.find_entity("billing invoice")
    assert by_name["results"][0]["matched_by"] == "name"
    assert set(ids(by_name)) == {"sys-billing", "sys-reports"}


def test_find_entity_by_text_when_no_alias_agrees(tools):
    result = tools.find_entity("settlement")
    assert ids(result)[0] == "sys-gateway"
    assert result["results"][0]["matched_by"] == "text"
    assert result["results"][0]["score"] > result["results"][1]["score"]
    assert ids(tools.find_entity("settlement", limit=2)) == ids(result)[:2]


def test_ambiguous_alias_returns_all_entities(tools):
    result = tools.find_entity("core")
    assert sorted(ids(result)) == ["feat-tiers", "sys-billing"]
    assert {r["matched_by"] for r in result["results"]} == {"alias"}
    assert ids(tools.find_entity("core", kind="feature")) == ["feat-tiers"]


def test_get_entity_gives_a_live_pointer_and_no_value(tools):
    result = tools.get_entity("env-stage", as_of="2026-01-01")
    assert result["live"] == [
        {"attribute": "monthly_cost", "system": "cost-explorer", "skill": "aws-cost", "key": "cost-explorer:1234"}
    ]
    # the row of the file has a value; the result never gives it
    assert "LEAKED-VALUE" not in json.dumps(result)
    assert [f["attribute"] for f in result["facts"]] == ["aws_account"]
    assert (result["id"], result["kind"], result["name"]) == ("env-stage", "environment", "Stage")
    assert result["relations"] == {"out": {"partOf": 1}, "in": {}}
    billing = tools.get_entity("sys-billing")
    assert billing["relations"] == {"out": {}, "in": {"partOf": 1}}
    assert "deploy" in {c["aspect"] for c in billing["canonical"]}
    assert tools.get_entity("sys-gateway")["live"] == []


def test_get_entity_filters_facts_by_as_of(tools):
    def accounts(as_of):
        result = tools.get_entity("env-stage", as_of=as_of)
        return [(f["value"], f["valid_until"], f["valid_from"]) for f in result["facts"]]

    assert accounts("2026-01-01") == [("1234", "2026-06-30", None)]
    assert accounts("2026-06-30") == [("1234", "2026-06-30", None)]
    assert accounts("2026-07-01") == [("5678", None, "2026-07-01")]
    # the default is today
    assert accounts(None) == [("5678", None, "2026-07-01")]


def test_get_entity_with_an_unknown_id_is_an_error(tools):
    with pytest.raises(LookupError, match="no-such"):
        tools.get_entity("no-such")


def test_find_entity_result_has_a_trace_id(tools):
    result = tools.find_entity("core")
    record = find_line(tools.cfg, result["trace_id"])
    assert record["gen_ai.tool.name"] == "find_entity"
    (entry,) = record["ragnarok.entities"]
    assert entry["name"] == "core"
    assert sorted(c["id"] for c in entry["candidates"]) == ["feat-tiers", "sys-billing"]
    assert {c["matched_by"] for c in entry["candidates"]} == {"alias"}
    # two equal candidates: no entity is selected
    assert entry["selected"] is None
    one = find_line(tools.cfg, tools.find_entity("Staging")["trace_id"])
    assert one["ragnarok.entities"][0]["selected"] == "env-stage"


def test_get_entity_result_has_a_trace_id(tools):
    result = tools.get_entity("env-stage")
    record = find_line(tools.cfg, result["trace_id"])
    assert record["gen_ai.tool.name"] == "get_entity"
    (entry,) = record["ragnarok.entities"]
    assert (entry["name"], entry["selected"]) == ("env-stage", "env-stage")
    assert entry["candidates"] == [{"id": "env-stage", "matched_by": "id", "score": None}]


def test_private_entity_is_left_out(tools):
    tools.cfg.private_patterns.append("Kb/private/*")
    put(tools.cfg, "Kb/private/Secret.md", "---\nid: ent-secret\ntype: entity\nkind: system\nsensitivity: private\n"
        "aliases: [Hush]\n---\n# Secret\nA private system.\n")  # fmt: skip
    assert "ent-secret" not in ids(tools.find_entity("hush"))
    with pytest.raises(LookupError):
        tools.get_entity("ent-secret")


def fts_rows(conn):
    return conn.execute("SELECT entity_id, name FROM entity_fts ORDER BY entity_id").fetchall()


def test_entity_fts_has_one_row_for_each_entity(tools):
    conn = tools.conn
    rows = fts_rows(conn)
    assert [i for i, _ in rows] == sorted(i for (i,) in conn.execute("SELECT id FROM entity"))
    assert dict(rows)["sys-billing"] == "Billing System"
    aliases = conn.execute("SELECT aliases FROM entity_fts WHERE entity_id = 'sys-billing'").fetchone()[0]
    assert aliases == "Billing Core"


def test_a_changed_entity_document_replaces_its_fts_row(tools):
    conn, cfg = tools.conn, tools.cfg
    count = len(fts_rows(conn))
    # the name of an entity is the file name: a moved file with the same id is a rename
    (cfg.root / "Kb/entities/Stage.md").unlink()
    put(cfg, "Kb/entities/Stagehouse.md", NOTES["Kb/entities/Stage.md"])
    update(cfg, conn, tools.embedder)
    rows = dict(fts_rows(conn))
    assert len(rows) == count
    assert rows["env-stage"] == "Stagehouse"
    # a removed document removes its row
    (cfg.root / "Kb/entities/Payment Gateway.md").unlink()
    update(cfg, conn, tools.embedder)
    assert "sys-gateway" not in dict(fts_rows(conn))


def test_a_rebuild_fills_entity_fts_again(tools):
    cfg = tools.cfg
    tools.conn.close()
    conn = rebuild(cfg)
    assert fts_rows(conn) == []
    update(cfg, conn, tools.embedder)
    assert len(fts_rows(conn)) == 5
    assert Tools(cfg, conn, tools.embedder).find_entity("staging")["results"][0]["id"] == "env-stage"
