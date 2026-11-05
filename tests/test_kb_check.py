"""The derived violation rows and kb_check, on a synthetic vault."""

import json

import pytest
from test_index import FakeEmbedder

from ragnarok.config import AttributeDef, Config, KbConfig
from ragnarok.index import connect, update
from ragnarok.tools import Tools

SYS_A = "---\nid: sys-a\ntype: entity\nkind: system\naliases: [Alpha]\n---\n# Sys A\nA system.\n"
NOTES = {
    "Kb/entities/Sys A.md": SYS_A,
    "Kb/entities/Sys B.md": "---\nid: sys-b\ntype: entity\nkind: system\n---\n# Sys B\nAnother system.\n",
}


def howto(subject: str | None, task: str, body: str = "Text.", extra: str = "") -> str:
    front = f"type: how-to\ntask: {task}\n{extra}"
    if subject is not None:
        front += f"subject: {subject}\n"
    return f"---\n{front}---\n# Doc\n{body}\n"


@pytest.fixture
def cfg(tmp_path):
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    attributes = {"system": {"channel": AttributeDef(live=True, system="chat", skill="chat-read")}}
    return Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=8,
                  kb=KbConfig(attributes=attributes))  # fmt: skip


def put(cfg, name: str, text: str) -> None:
    path = cfg.root / "Kb" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(cfg):
    conn = connect(cfg)
    emb = FakeEmbedder()
    update(cfg, conn, emb)
    return conn, emb


def violations(conn, rule):
    rows = conn.execute("SELECT node_ids, detail FROM violation WHERE rule = ? ORDER BY id", (rule,)).fetchall()
    return [(json.loads(ids), detail) for ids, detail in rows]


def test_link_in_a_code_block_is_not_m1(cfg):
    body = "Prose [[missing-prose]] and [[both]].\n\n```\n[[in-fence]]\n```\n\nSpan `[[in-span]]` and `[[both]]`.\n"
    put(cfg, "a.md", f"# A\n{body}")
    conn, _ = build(cfg)
    marks = dict(conn.execute("SELECT target_text, in_code FROM link").fetchall())
    # a target with one prose occurrence is not in code
    assert marks == {"missing-prose": 0, "both": 0, "in-fence": 1, "in-span": 1}
    assert sorted(d for _, d in violations(conn, "M1")) == ["both", "missing-prose"]


def test_two_active_state_docs_with_one_key_give_m3(cfg):
    put(cfg, "one.md", howto("'[[Sys A]]'", "Deploy"))
    put(cfg, "two.md", howto("'[[Alpha]]'", "deploy"))
    put(cfg, "three.md", howto("'[[Sys B]]'", "Deploy"))
    conn, _ = build(cfg)
    ids = {p: i for p, i in conn.execute("SELECT path, id FROM doc")}
    found = violations(conn, "M3")
    assert len(found) == 1
    assert sorted(found[0][0]) == sorted([ids["Kb/one.md"], ids["Kb/two.md"]])
    assert found[0][1].startswith("Kb|how-to|sys-a")


def test_unknown_kind_gives_a_vocab_violation(cfg):
    put(cfg, "entities/Gadget.md", "---\nid: gadget-1\ntype: entity\nkind: gadget\n---\n# Gadget\nA gadget.\n")
    conn, _ = build(cfg)
    found = violations(conn, "vocab")
    assert [ids for ids, _ in found] == [["gadget-1"]]
    assert "gadget" in found[0][1]


def test_live_fact_with_a_value_gives_a_live_value_violation(cfg):
    facts = "facts:\n  - {attribute: channel, value: '#ops'}\n"
    put(cfg, "entities/Sys A.md", SYS_A.replace("---\n# Sys A", f"{facts}---\n# Sys A"))
    conn, _ = build(cfg)
    assert [ids for ids, _ in violations(conn, "live_value")] == [["sys-a"]]


def test_unresolved_references_give_ref_violations(cfg):
    put(cfg, "orphan.md", howto("'[[Nobody]]'", "Fix", extra="owner: '[[ghost]]'\nsuperseded_by: '[[gone]]'\n"))
    conn, _ = build(cfg)
    details = sorted(d for _, d in violations(conn, "ref"))
    assert details == ["owner: ghost", "subject: Nobody", "successor: gone"]


def test_relation_with_a_wrong_end_kind_gives_relation_kinds(cfg):
    # `requests` goes from a client or a doc to a feature: a system is the wrong end
    put(
        cfg,
        "entities/Client.md",
        "---\nid: client-1\ntype: entity\nkind: client\nrequests: '[[sys-a]]'\n---\n# C\nC.\n",
    )
    conn, _ = build(cfg)
    assert [ids for ids, _ in violations(conn, "relation_kinds")] == [["client-1", "sys-a"]]


def test_two_canonical_docs_give_canonical_twice(cfg):
    put(cfg, "one.md", howto("'[[Sys A]]'", "Deploy", extra="canonical: true\n"))
    put(cfg, "two.md", howto("'[[Sys A]]'", "Deploy", extra="canonical: true\n"))
    conn, _ = build(cfg)
    found = violations(conn, "canonical_twice")
    assert len(found) == 1
    assert len(found[0][0]) == 2


def test_state_doc_with_no_subject_gives_subjects(cfg):
    # the importer keeps at most 3 subjects, so only the missing subject can occur here
    put(cfg, "none.md", howto(None, "Alpha task"))
    put(cfg, "ok.md", howto("'[[Sys B]]'", "Gamma task"))
    conn, _ = build(cfg)
    ids = {p: i for p, i in conn.execute("SELECT path, id FROM doc")}
    assert [(i, d) for i, d in violations(conn, "subjects")] == [([ids["Kb/none.md"]], "0 subjects")]


def test_check_replaces_the_violations_after_a_second_import(cfg):
    put(cfg, "a.md", "# A\n[[missing-one]] [[missing-two]]\n")
    conn, emb = build(cfg)
    assert len(violations(conn, "M1")) == 2
    put(cfg, "a.md", "# A\n[[missing-one]]\n")
    update(cfg, conn, emb)
    assert [d for _, d in violations(conn, "M1")] == ["missing-one"]
    assert conn.execute("SELECT count(*) FROM violation WHERE rule = 'M1'").fetchone()[0] == 1


def test_kb_check_filters_by_subject(cfg):
    put(cfg, "one.md", howto("'[[Sys A]]'", "Deploy", "See [[broken-a]]."))
    put(cfg, "two.md", howto("'[[Sys B]]'", "Deploy", "See [[broken-b]]."))
    conn, emb = build(cfg)
    tools = Tools(cfg, conn, emb)
    everything = tools.kb_check()
    assert everything["M1"]["count"] == 2
    assert everything["trace_id"]
    only_a = tools.kb_check(subject="Alpha")
    assert only_a["M1"]["count"] == 1
    assert [v["detail"] for v in only_a["M1"]["violations"]] == ["broken-a"]
    assert tools.kb_check(collection="Other")["M1"]["count"] == 0
    with pytest.raises(LookupError):
        tools.kb_check(subject="Nobody Here")
