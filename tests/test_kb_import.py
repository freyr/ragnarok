"""The entity tables and the reference pass, on a synthetic vault: one environment, one system, one feature, two
clients and one group."""

import dataclasses
import re

import pytest
from test_index import FakeEmbedder

from ragnarok.config import AttributeDef, Config, DocType, KbConfig, RelationType
from ragnarok.index import connect, update
from ragnarok.kb import normalize_alias

NOTES = {
    "Kb/entities/Stage.md": (
        "---\nid: env-stage\ntype: entity\nkind: environment\naliases: [Staging, '  STG  ']\n"
        "owner: '[[Platform Team]]'\nas_of: 2026-10-01\n"
        "facts:\n"
        "  - {attribute: aws_account, value: '1234', as_of: 2026-10-01}\n"
        "  - {attribute: monthly_cost, source: 'cost-explorer:1234'}\n"
        "  - {attribute: monthly_cost, value: '99', source: 'cost-explorer:5678'}\n"
        "partOf: '[[Billing System]]'\n---\n# Stage\nThe stage environment.\n"
    ),
    "Kb/entities/Billing System.md": (
        "---\nid: sys-billing\ntype: entity\nkind: system\naliases: [Billing, Core]\nowner: '[[team-platform]]'\n"
        "canonical: {deploy: '[[how-to-deploy-billing]]'}\n---\n# Billing System\nIt bills.\n"
    ),
    "Kb/entities/Loyalty Tiers.md": (
        "---\nid: feat-tiers\ntype: entity\nkind: feature\naliases: [Tiers, Core]\npartOf: '[[sys-billing]]'\n"
        "---\n# Loyalty Tiers\nTiers.\n"
    ),
    "Kb/entities/Client Ąlpha.md": (
        "---\nid: client-alpha\ntype: entity\nkind: client\naliases: ['Ålpha   Corp']\n"
        "requests: '[[Loyalty Tiers]]'\n---\n# Client Ąlpha\nA client.\n"
    ),
    "Kb/entities/Client Beta.md": (
        "---\nid: client-beta\ntype: entity\nkind: client\nrequests: ['[[feat-tiers]]', '[[Unknown Feature]]']\n"
        "---\n# Client Beta\nAnother client.\n"
    ),
    "Kb/entities/Platform Team.md": (
        "---\nid: team-platform\ntype: entity\nkind: group\n---\n# Platform Team\nA team.\n"
    ),
    "Kb/howto/how-to-deploy-billing.md": (
        "---\ntype: how-to\nsubject: '[[Billing]]'\ntask: Deploy the service\nowner: '[[team-platform]]'\n"
        "canonical: true\n---\n# Deploy\nDeploy to [[Stage]] first, see [[env-stage]] and [[Staging]].\n"
    ),
    "Kb/meetings/call.md": "---\ntype: meeting\nrequests: '[[feat-tiers]]'\n---\n# Call\nThe client asked.\n",
}

STAGE = "Kb/entities/Stage.md"
HOW_TO = "Kb/howto/how-to-deploy-billing.md"


@pytest.fixture
def kb(tmp_path):
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    attributes = {
        "environment": {
            "aws_account": AttributeDef(),
            "monthly_cost": AttributeDef(live=True, system="cost-explorer", skill="aws-cost"),
        }
    }
    return Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=8,
                  kb=KbConfig(attributes=attributes))  # fmt: skip


def build(cfg):
    embedder = FakeEmbedder()
    conn = connect(cfg)
    update(cfg, conn, embedder)
    return conn, embedder


def doc_id(conn, path):
    return conn.execute("SELECT id FROM doc WHERE path = ?", (path,)).fetchone()[0]


def rows(conn, sql, *args):
    return conn.execute(sql, args).fetchall()


def test_entity_document_gives_an_entity_and_a_doc_node(kb):
    conn, _ = build(kb)
    nodes = rows(conn, "SELECT id, node_type, collection, mode, provisional FROM node WHERE id LIKE 'env-stage%'")
    assert sorted(nodes) == [("env-stage", "entity", "Kb", "files", 0), ("env-stage~doc", "doc", "Kb", "files", 0)]
    entity = rows(conn, "SELECT id, kind, name, status, as_of, doc_id FROM entity WHERE id = 'env-stage'")
    assert entity == [("env-stage", "environment", "Stage", "active", "2026-10-01", "env-stage~doc")]
    assert doc_id(conn, STAGE) == "env-stage~doc"
    assert rows(conn, "SELECT type FROM doc WHERE id = 'env-stage~doc'") == [("entity",)]
    assert conn.execute("SELECT count(*) FROM entity").fetchone()[0] == 6
    # a document that is not an entity document gives no entity
    assert rows(conn, "SELECT count(*) FROM entity WHERE doc_id = ?", doc_id(conn, HOW_TO)) == [(0,)]


def test_aliases_are_normalized(kb):
    conn, _ = build(kb)
    sql = "SELECT alias_norm, alias FROM entity_alias WHERE entity_id = ? ORDER BY alias_norm"
    assert rows(conn, sql, "client-alpha") == [("alpha corp", "Ålpha   Corp"), ("client alpha", "Client Ąlpha")]
    assert rows(conn, sql, "env-stage") == [("stage", "Stage"), ("staging", "Staging"), ("stg", "STG")]


def test_live_fact_keeps_only_the_source(kb):
    conn, _ = build(kb)
    facts = rows(
        conn,
        "SELECT attribute, value, as_of, source_ref, origin, source_doc FROM fact WHERE entity_id = 'env-stage'"
        " ORDER BY attribute, source_ref",
    )
    assert facts == [
        ("aws_account", "1234", "2026-10-01", None, "declared", "env-stage~doc"),
        ("monthly_cost", None, None, "cost-explorer:1234", "declared", "env-stage~doc"),
        # a live fact with a value in the file keeps the value: the later check reports it
        ("monthly_cost", "99", None, "cost-explorer:5678", "declared", "env-stage~doc"),
    ]


def test_relation_keys_give_declared_relations(kb):
    conn, _ = build(kb)
    call = doc_id(conn, "Kb/meetings/call.md")
    got = rows(conn, "SELECT src, rel, dst, dst_text, origin, source_doc FROM relation")
    assert sorted(got, key=lambda r: (r[0], r[3])) == sorted(
        [
            ("env-stage", "partOf", "sys-billing", "Billing System", "declared", "env-stage~doc"),
            ("feat-tiers", "partOf", "sys-billing", "sys-billing", "declared", "feat-tiers~doc"),
            ("client-alpha", "requests", "feat-tiers", "Loyalty Tiers", "declared", "client-alpha~doc"),
            ("client-beta", "requests", "feat-tiers", "feat-tiers", "declared", "client-beta~doc"),
            ("client-beta", "requests", None, "Unknown Feature", "declared", "client-beta~doc"),
            (call, "requests", "feat-tiers", "feat-tiers", "declared", call),
        ],
        key=lambda r: (r[0], r[3]),
    )


def test_subject_link_resolves_to_the_entity(kb):
    conn, _ = build(kb)
    how_to = doc_id(conn, HOW_TO)
    assert rows(conn, "SELECT entity_id, target_text FROM doc_subject WHERE doc_id = ?", how_to) == [
        ("sys-billing", "Billing")
    ]
    assert rows(conn, "SELECT primary_subject, owner FROM doc WHERE id = ?", how_to) == [
        ("sys-billing", "team-platform")
    ]
    # an owner by path (the file name) and by entity ID
    assert rows(conn, "SELECT id, owner FROM entity WHERE owner IS NOT NULL ORDER BY id") == [
        ("env-stage", "team-platform"),
        ("sys-billing", "team-platform"),
    ]
    # a body link goes to the document node: by path, by entity ID and by alias
    links = rows(conn, "SELECT target_text, to_node FROM link WHERE from_doc = ? ORDER BY target_text", how_to)
    assert links == [("Stage", "env-stage~doc"), ("Staging", "env-stage~doc"), ("env-stage", "env-stage~doc")]
    # the canonical document of its primary subject and task, and the canonical list of the entity document
    got = rows(conn, "SELECT entity_id, aspect, doc_id, source_doc FROM entity_canonical ORDER BY source_doc")
    assert sorted(got) == sorted(
        [("sys-billing", "Deploy the service", how_to, how_to), ("sys-billing", "deploy", how_to, "sys-billing~doc")]
    )


def test_unresolved_reference_keeps_its_text(kb):
    (kb.root / "Kb/howto/orphan.md").write_text(
        "---\ntype: how-to\nsubject: ['[[Nobody]]', '[[Core]]']\ntask: Fix it\nowner: '[[ghost]]'\n"
        "superseded_by: '[[missing-note]]'\n---\n# Orphan\nText.\n"
    )
    conn, _ = build(kb)
    orphan = doc_id(conn, "Kb/howto/orphan.md")
    # an absent target and an ambiguous alias (Core names two entities) stay null
    assert rows(conn, "SELECT ord, entity_id, target_text FROM doc_subject WHERE doc_id = ? ORDER BY ord", orphan) == [
        (0, None, "Nobody"),
        (1, None, "Core"),
    ]
    sql = "SELECT primary_subject, owner, owner_text, superseded_by, superseded_by_text, identity_key FROM doc"
    assert rows(conn, sql + " WHERE id = ?", orphan) == [(None, None, "ghost", None, "missing-note", None)]
    assert rows(conn, "SELECT dst, dst_text FROM relation WHERE dst IS NULL") == [(None, "Unknown Feature")]


def test_changed_entity_document_replaces_its_rows(kb):
    conn, emb = build(kb)
    (kb.root / STAGE).write_text(
        "---\nid: env-stage\ntype: entity\nkind: environment\naliases: [Preprod]\n"
        "facts:\n  - {attribute: aws_account, value: '9999'}\n"
        "dependsOn: '[[Platform Team]]'\n---\n# Stage\nThe stage environment, new text.\n"
    )
    result = update(kb, conn, emb)
    assert result.changed == 1
    assert rows(conn, "SELECT alias_norm FROM entity_alias WHERE entity_id = 'env-stage' ORDER BY alias_norm") == [
        ("preprod",),
        ("stage",),
    ]
    assert rows(conn, "SELECT attribute, value FROM fact WHERE entity_id = 'env-stage'") == [("aws_account", "9999")]
    assert rows(conn, "SELECT rel, dst FROM relation WHERE src = 'env-stage'") == [("dependsOn", "team-platform")]
    assert rows(conn, "SELECT owner FROM entity WHERE id = 'env-stage'") == [(None,)]
    # a deleted entity document removes its rows, and the references to it become null
    (kb.root / STAGE).unlink()
    update(kb, conn, emb)
    for table, column in (("node", "id"), ("entity", "id"), ("entity_alias", "entity_id"), ("fact", "entity_id")):
        assert rows(conn, f"SELECT count(*) FROM {table} WHERE {column} LIKE 'env-stage%'") == [(0,)], table
    assert rows(conn, "SELECT count(*) FROM relation WHERE src = 'env-stage'") == [(0,)]
    how_to = doc_id(conn, HOW_TO)
    assert rows(conn, "SELECT count(*) FROM link WHERE from_doc = ? AND to_node IS NOT NULL", how_to) == [(0,)]


def test_vocabulary_tables_follow_the_configuration(kb):
    conn, _ = build(kb)
    assert rows(conn, "SELECT standard FROM kind WHERE name = 'environment'") == [("local",)]
    assert rows(conn, "SELECT count(*) FROM kind") == [(12,)]
    assert rows(conn, "SELECT class, one_per, standard FROM doc_type WHERE name = 'how-to'") == [
        ("state", "subject+task", "Diataxis")
    ]
    assert rows(conn, "SELECT src_kinds, dst_kinds, inverse FROM relation_type WHERE name = 'requests'") == [
        ('["doc", "client"]', '["feature"]', "requestedBy")
    ]
    assert rows(conn, "SELECT kind, name, live, system, skill FROM attribute_def ORDER BY name") == [
        ("environment", "aws_account", 0, "", ""),
        ("environment", "monthly_cost", 1, "cost-explorer", "aws-cost"),
    ]
    conn.close()
    kb.kb = KbConfig(
        kinds={"team": "local"},
        doc_types={"note": DocType("none", "none", review_months=6)},
        relation_types={"ownedBy": RelationType(["team"], [], "owns")},
        attributes={},
    )
    conn = connect(kb)
    assert rows(conn, "SELECT name, standard FROM kind") == [("team", "local")]
    assert rows(conn, "SELECT name, class, one_per, review_months FROM doc_type") == [("note", "none", "none", 6)]
    assert rows(conn, "SELECT name, src_kinds, dst_kinds, inverse FROM relation_type") == [
        ("ownedBy", '["team"]', "[]", "owns")
    ]
    assert rows(conn, "SELECT count(*) FROM attribute_def") == [(0,)]


def test_identity_key_uses_the_resolved_subject(kb):
    (kb.root / "Kb/howto/deploy-again.md").write_text(
        "---\ntype: how-to\nsubject: '[[Billing System]]'\ntask: '  deploy the   SERVICE'\n---\n# Again\nText.\n"
    )
    (kb.root / "Kb/howto/unknown.md").write_text(
        "---\ntype: how-to\nsubject: '[[Nobody]]'\ntask: Deploy the service\n---\n# Unknown\nText.\n"
    )
    conn, _ = build(kb)
    keys = dict(rows(conn, "SELECT path, identity_key FROM doc WHERE type = 'how-to'"))
    assert keys == {
        HOW_TO: "Kb|how-to|sys-billing|deploy the service",
        "Kb/howto/deploy-again.md": "Kb|how-to|sys-billing|deploy the service",
        "Kb/howto/unknown.md": None,
    }


def test_supersedes_sets_superseded_by_of_the_target(kb):
    (kb.root / "Kb/howto/deploy-v2.md").write_text(
        "---\ntype: how-to\nsubject: '[[Billing]]'\ntask: Deploy\nsupersedes: ['[[how-to-deploy-billing]]', "
        "'[[call]]']\n---\n# V2\nText.\n"
    )
    (kb.root / "Kb/meetings/call.md").write_text(
        "---\ntype: meeting\nsuperseded_by: '[[Stage]]'\n---\n# Call\nThe client asked.\n"
    )
    conn, _ = build(kb)
    v2 = doc_id(conn, "Kb/howto/deploy-v2.md")
    sql = "SELECT superseded_by, superseded_by_text FROM doc WHERE path = ?"
    assert rows(conn, sql, HOW_TO) == [(v2, None)]
    # a target that gives its own successor keeps it
    assert rows(conn, sql, "Kb/meetings/call.md") == [("env-stage~doc", "Stage")]


def test_removed_supersedes_clears_the_derived_successor(kb):
    v2_path = kb.root / "Kb/howto/deploy-v2.md"
    v2_path.write_text("---\ntype: how-to\nsupersedes: '[[how-to-deploy-billing]]'\n---\n# V2\nText.\n")
    conn, emb = build(kb)
    sql = "SELECT superseded_by FROM doc WHERE path = ?"
    assert rows(conn, sql, HOW_TO) == [(doc_id(conn, "Kb/howto/deploy-v2.md"),)]
    v2_path.write_text("---\ntype: how-to\n---\n# V2\nText.\n")
    result = update(kb, conn, emb)
    assert result.changed == 1
    assert rows(conn, sql, HOW_TO) == [(None,)]


def test_provisional_entity_document_gets_its_id_when_it_is_free(kb):
    """An entity document whose id another doc holds gets a provisional pair of IDs, and the free id later."""
    (kb.root / "Kb/entities/Stage copy.md").write_text(NOTES[STAGE])
    conn, emb = build(kb)
    ids = {doc_id(conn, STAGE): STAGE, doc_id(conn, "Kb/entities/Stage copy.md"): "Kb/entities/Stage copy.md"}
    assert "env-stage~doc" in ids
    (provisional,) = set(ids) - {"env-stage~doc"}
    assert re.fullmatch(r"~[0-9a-f]{12}~doc", provisional)
    assert rows(conn, "SELECT id, doc_id FROM entity WHERE id = ?", provisional.removesuffix("~doc")) == [
        (provisional.removesuffix("~doc"), provisional)
    ]
    # the holder goes, and the other document takes the id with its entity, facts and relations
    holder = ids["env-stage~doc"]
    (kb.root / holder).unlink()
    update(kb, conn, emb)
    assert rows(conn, "SELECT id, doc_id FROM entity WHERE name LIKE 'Stage%'") == [("env-stage", "env-stage~doc")]
    assert rows(conn, "SELECT count(*) FROM fact WHERE entity_id = 'env-stage' AND source_doc = 'env-stage~doc'") == [
        (3,)
    ]
    assert rows(conn, "SELECT src, source_doc FROM relation WHERE rel = 'partOf' AND dst_text = 'Billing System'") == [
        ("env-stage", "env-stage~doc")
    ]
    assert rows(conn, "SELECT count(*) FROM node WHERE node_type = 'entity' AND provisional = 1") == [(0,)]


def test_rules_change_with_no_file_change_rewrites_no_entity_document(kb):
    conn, emb = build(kb)
    kb.kb = dataclasses.replace(kb.kb, kinds={**kb.kb.kinds, "team": "local"})
    conn.close()
    conn = connect(kb)
    result = update(kb, conn, emb)
    assert (result.rederived, result.changed) == (0, 0)
    assert rows(conn, "SELECT count(*) FROM kind WHERE name = 'team'") == [(1,)]
    assert rows(conn, "SELECT identity_key FROM doc WHERE path = ?", HOW_TO) == [
        ("Kb|how-to|sys-billing|deploy the service",)
    ]


def test_normalize_alias_folds_letters_that_have_no_decomposition():
    # the expected strings are written by hand
    assert normalize_alias("Łódź Ørsted Đurđ") == "lodz orsted durd"
    assert normalize_alias("Straße Æther Œuvre Ħal") == "strasse aether oeuvre hal"


@pytest.mark.parametrize("entity_path", ["Kb/a-entity.md", "Kb/z-entity.md"])
def test_shared_id_of_an_entity_and_a_plain_document_warns(kb, entity_path):
    (kb.root / entity_path).write_text("---\nid: dup-1\ntype: entity\nkind: group\n---\n# Entity\nA team.\n")
    (kb.root / "Kb/m-plain.md").write_text("---\nid: dup-1\n---\n# Plain\nA note.\n")
    result = update(kb, connect(kb), FakeEmbedder())
    assert any("dup-1" in w for w in result.warnings)
