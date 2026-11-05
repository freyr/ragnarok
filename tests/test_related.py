"""related: exact joins through the declared relations, on a synthetic vault."""

import pytest

from ragnarok.config import Config, KbConfig
from ragnarok.index import connect, update
from ragnarok.tools import Tools
from ragnarok.trace import find_line


def entity(id_, name, kind, extra=""):
    return f"---\nid: {id_}\ntype: entity\nkind: {kind}\n{extra}---\n# {name}\nThe {name}.\n"


NOTES = {
    "Kb/entities/Loyalty Tiers.md": entity("feat-tiers", "Loyalty Tiers", "feature", "aliases: [Tiers, Core]\n"),
    "Kb/entities/Acme.md": entity("client-acme", "Acme", "client", "requests: '[[Loyalty Tiers]]'\n"),
    "Kb/entities/Beta.md": entity("client-beta", "Beta", "client", "requests: '[[Loyalty Tiers]]'\n"),
    "Kb/entities/Engine.md": entity("sys-engine", "Engine", "system", "partOf: '[[Core System]]'\n"),
    "Kb/entities/Core System.md": entity(
        "sys-core",
        "Core System",
        "system",
        "aliases: [Core]\npartOf: '[[Bet Home]]'\ndependsOn: ['[[Payment Gateway]]', '[[Loyalty Tiers]]']\n",
    ),  # fmt: skip
    "Kb/entities/Bet Home.md": entity("plat-bet", "Bet Home", "system"),
    "Kb/entities/Payment Gateway.md": entity("sys-gateway", "Payment Gateway", "system"),
    "Kb/entities/Cycle A.md": entity("cyc-a", "Cycle A", "system", "dependsOn: '[[Cycle B]]'\n"),
    "Kb/entities/Cycle B.md": entity("cyc-b", "Cycle B", "system", "dependsOn: '[[Cycle C]]'\n"),
    "Kb/entities/Kt S.md": entity("kt-s", "Kt S", "system", "dependsOn: ['[[Kt A]]', '[[Kt B]]']\n"),
    "Kb/entities/Kt A.md": entity("kt-a", "Kt A", "system", "dependsOn: '[[Kt B]]'\n"),
    "Kb/entities/Kt B.md": entity("kt-b", "Kt B", "feature"),
    "Kb/entities/Cycle C.md": entity("cyc-c", "Cycle C", "system", "dependsOn: '[[Cycle A]]'\n"),
}


class Embedder:
    def embed_docs(self, items):
        return [[1.0, 0.0] for _ in items]

    def embed_query(self, query):
        return [1.0, 0.0]


@pytest.fixture
def tools(tmp_path):
    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=2, kb=KbConfig())
    for rel, text in NOTES.items():
        path = cfg.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    conn = connect(cfg)
    emb = Embedder()
    update(cfg, conn, emb)
    return Tools(cfg, conn, emb)


def far(result, direction="out"):
    end = "target" if direction == "out" else "source"
    return sorted((e[end]["id"], e["depth"]) for e in result["edges"])


def test_related_in_direction_gives_the_clients_of_a_feature(tools):
    result = tools.related("Loyalty Tiers", relation="requests", direction="in")
    assert far(result, "in") == [("client-acme", 1), ("client-beta", 1)]
    edge = next(e for e in result["edges"] if e["source"]["id"] == "client-acme")
    assert edge["relation"] == "requests"
    assert edge["source"] == {
        "id": "client-acme",
        "kind": "client",
        "name": "Acme",
        "path": "Kb/entities/Acme.md",
    }
    assert edge["target"] == {
        "id": "feat-tiers",
        "kind": "feature",
        "name": "Loyalty Tiers",
        "path": "Kb/entities/Loyalty Tiers.md",
    }
    assert (edge["origin"], edge["confidence"], edge["valid_from"], edge["valid_until"]) == (
        "declared",
        None,
        None,
        None,
    )
    assert edge["source_doc"] == "client-acme~doc"
    # the entity is an ID or a name, and an alias that names two entities is an error with the candidates
    assert far(tools.related("feat-tiers", relation="requests", direction="in"), "in") == far(result, "in")
    with pytest.raises(LookupError, match="feat-tiers.*sys-core|sys-core.*feat-tiers"):
        tools.related("core")


def test_related_depth_two_follows_part_of(tools):
    assert far(tools.related("sys-engine", relation="partOf")) == [("sys-core", 1)]
    assert far(tools.related("sys-engine", relation="partOf", depth=2)) == [("plat-bet", 2), ("sys-core", 1)]
    with pytest.raises(ValueError, match="depth"):
        tools.related("sys-engine", depth=4)


def test_related_stops_at_a_cycle(tools):
    result = tools.related("cyc-a", relation="dependsOn", depth=3)
    assert [(e["source"]["id"], e["target"]["id"]) for e in result["edges"]] == [("cyc-a", "cyc-b"), ("cyc-b", "cyc-c")]


def test_related_filters_by_kind_and_as_of(tools):
    tools.conn.execute("UPDATE relation SET valid_until = '2026-01-31' WHERE src = 'sys-core' AND dst = 'sys-gateway'")
    both = far(tools.related("sys-core", relation="dependsOn", as_of="2026-01-01"))
    assert both == [("feat-tiers", 1), ("sys-gateway", 1)]
    assert far(tools.related("sys-core", relation="dependsOn", as_of="2026-02-01")) == [("feat-tiers", 1)]
    by_kind = tools.related("sys-core", relation="dependsOn", kind="system", as_of="2026-01-01")
    assert far(by_kind) == [("sys-gateway", 1)]
    with pytest.raises(ValueError, match="ISO date"):
        tools.related("sys-core", as_of="2026-02")


def test_related_result_has_a_trace_id(tools):
    result = tools.related("sys-engine")
    record = find_line(tools.cfg, result["trace_id"])
    assert record["gen_ai.tool.name"] == "related"


def test_related_trace_records_the_edges_before_the_filters_and_the_removing_filter(tools):
    tools.conn.execute("UPDATE relation SET valid_until = '2026-01-31' WHERE src = 'sys-core' AND dst = 'sys-gateway'")
    result = tools.related("Core System", relation="dependsOn", kind="feature", as_of="2026-02-01")
    assert far(result) == [("feat-tiers", 1)]
    (entry,) = find_line(tools.cfg, result["trace_id"])["ragnarok.entities"]
    assert (entry["name"], entry["selected"]) == ("Core System", "sys-core")
    assert sorted((e["src"], e["dst"]) for e in entry["edges_before"]) == [
        ("sys-core", "feat-tiers"),
        ("sys-core", "sys-gateway"),
    ]
    assert [(e["src"], e["dst"]) for e in entry["edges"]] == [("sys-core", "feat-tiers")]
    assert entry["removed"] == [{"id": "sys-gateway", "filter": "as_of"}]
    # a kind filter removes the entity with its own name
    kind_only = tools.related("Core System", relation="dependsOn", kind="feature", as_of="2026-01-01")
    (entry,) = find_line(tools.cfg, kind_only["trace_id"])["ragnarok.entities"]
    assert entry["removed"] == [{"id": "sys-gateway", "filter": "kind"}]


def test_related_depth_is_the_shortest_valid_path(tools):
    conn = tools.conn
    add = (
        "INSERT INTO relation (src, rel, dst, dst_text, source_doc, origin)"
        " VALUES (?, 'dependsOn', ?, ?, ?, 'declared')"
    )
    # Gateway -> Bet Home is reached through Core -> Gateway (2 hops) and Core -> Tiers -> Gateway (3 hops)
    conn.execute(add, ("feat-tiers", "sys-gateway", "Payment Gateway", "feat-tiers~doc"))
    conn.execute(add, ("sys-gateway", "plat-bet", "Bet Home", "sys-gateway~doc"))
    conn.execute("UPDATE relation SET valid_until = '2020-01-01' WHERE src = 'sys-core' AND dst = 'sys-gateway'")
    result = tools.related("sys-core", relation="dependsOn", depth=3)
    edge = next(e for e in result["edges"] if e["source"]["id"] == "sys-gateway" and e["target"]["id"] == "plat-bet")
    assert edge["depth"] == 3


def test_related_with_an_unknown_relation_is_an_error(tools):
    with pytest.raises(ValueError, match="dependson.*dependsOn"):
        tools.related("sys-core", relation="dependson")


def ends(result):
    return sorted((e["source"]["id"], e["relation"], e["target"]["id"], e["depth"]) for e in result["edges"])


def test_related_both_at_depth_one_gives_the_out_and_in_edges_once(tools):
    result = tools.related("sys-core", direction="both")
    assert ends(result) == [
        ("sys-core", "dependsOn", "feat-tiers", 1),
        ("sys-core", "dependsOn", "sys-gateway", 1),
        ("sys-core", "partOf", "plat-bet", 1),
        ("sys-engine", "partOf", "sys-core", 1),
    ]


def test_related_both_at_depth_two_goes_out_then_in(tools):
    # Core -> Tiers (out), then Acme -> Tiers and Beta -> Tiers (in)
    result = tools.related("sys-core", direction="both", kind="client", depth=2)
    assert ends(result) == [
        ("client-acme", "requests", "feat-tiers", 2),
        ("client-beta", "requests", "feat-tiers", 2),
    ]
    assert tools.related("sys-core", direction="both", kind="client")["edges"] == []


def test_related_both_stops_at_a_cycle(tools):
    result = tools.related("cyc-a", relation="dependsOn", direction="both", depth=3)
    assert ends(result) == [
        ("cyc-a", "dependsOn", "cyc-b", 1),
        ("cyc-b", "dependsOn", "cyc-c", 2),
        ("cyc-c", "dependsOn", "cyc-a", 1),
    ]


def test_related_both_trace_records_the_edges_before_the_filters(tools):
    result = tools.related("Core System", direction="both", kind="feature")
    assert ends(result) == [("sys-core", "dependsOn", "feat-tiers", 1)]
    (entry,) = find_line(tools.cfg, result["trace_id"])["ragnarok.entities"]
    assert sorted((e["src"], e["dst"]) for e in entry["edges_before"]) == [
        ("sys-core", "feat-tiers"),
        ("sys-core", "plat-bet"),
        ("sys-core", "sys-gateway"),
        ("sys-engine", "sys-core"),
    ]
    assert [(e["src"], e["dst"]) for e in entry["edges"]] == [("sys-core", "feat-tiers")]
    # the reached entity of the in edge is its source
    assert sorted(entry["removed"], key=lambda r: r["id"]) == [
        {"id": "plat-bet", "filter": "kind"},
        {"id": "sys-engine", "filter": "kind"},
        {"id": "sys-gateway", "filter": "kind"},
    ]


def test_related_both_with_a_kind_keeps_an_edge_that_out_keeps(tools):
    # Kt S -> Kt A, Kt S -> Kt B, Kt A -> Kt B: the edge A -> B has the ends A and B at depth 2 with both
    out = tools.related("kt-s", direction="out", kind="feature", depth=2)
    assert ends(out) == [("kt-a", "dependsOn", "kt-b", 2), ("kt-s", "dependsOn", "kt-b", 1)]
    both = tools.related("kt-s", direction="both", kind="feature", depth=2)
    assert ends(both) == ends(out)


def test_related_both_at_depth_three_through_a_hub_is_fast(tools):
    import time

    add = (
        "INSERT INTO relation (src, rel, dst, dst_text, source_doc, origin)"
        " VALUES (?, 'partOf', ?, ?, 'sys-core~doc', 'declared')"
    )
    for i in range(400):
        for step in (1, 7, 31):
            tools.conn.execute(add, (f"n{i}", f"n{(i * step + 1) % 400}", "x"))
        tools.conn.execute(add, (f"n{i}", "sys-core", "Core System"))
    start = time.perf_counter()
    result = tools.related("sys-core", direction="both", depth=3)
    assert time.perf_counter() - start < 0.15
    assert len(result["edges"]) > 1000


def test_related_gives_no_path_of_a_private_document(tools):
    tools.cfg.private_patterns.append("Kb/private/*")
    path = tools.cfg.root / "Kb/private/Secret.md"
    path.parent.mkdir(parents=True)
    path.write_text(entity("sys-secret", "Secret", "system", "sensitivity: private\ndependsOn: '[[Loyalty Tiers]]'\n"))
    path = tools.cfg.root / "Kb/entities/Open.md"
    path.write_text(entity("sys-open", "Open", "system", "dependsOn: '[[Secret]]'\n"))
    update(tools.cfg, tools.conn, tools.embedder)
    for result in (
        tools.related("Open"),
        tools.related("Loyalty Tiers", direction="in"),
        tools.related("Loyalty Tiers", direction="both", depth=2),
    ):
        assert "Kb/private" not in str(result)
    tools.cfg.allow_private = True
    edge = tools.related("Open")["edges"][0]
    assert edge["target"]["path"] == "Kb/private/Secret.md"
