"""The derived violation rows and kb_check, on a synthetic vault."""

import json

import pytest
from test_index import FakeEmbedder

from ragnarok import kb
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


def test_table_pipe_escape_and_attachment_are_not_m1(cfg):
    put(cfg, "Plan 2026.11.md", "# Plan\nText.\n")
    body = (
        "| Note | Use |\n|---|---|\n| [[Plan 2026.11\\|the plan]] | x |\n\n"
        "![[floor.svg]] and [[photo.PNG]] and [[Plan 2026.11]] and [[missing\\|label]].\n"
    )
    put(cfg, "a.md", f"# A\n{body}")
    conn, _ = build(cfg)
    targets = {t for (t,) in conn.execute("SELECT target_text FROM link")}
    # the escape of a pipe in a table is not part of the target; an attachment is not a note link
    assert targets == {"Plan 2026.11", "missing"}
    assert [d for _, d in violations(conn, "M1")] == ["missing"]


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
    put(cfg, "none.md", howto(None, "Alpha task"))
    put(cfg, "ok.md", howto("'[[Sys B]]'", "Gamma task"))
    conn, _ = build(cfg)
    ids = {p: i for p, i in conn.execute("SELECT path, id FROM doc")}
    assert [(i, d) for i, d in violations(conn, "subjects")] == [([ids["Kb/none.md"]], "0 subjects")]


def test_state_doc_with_four_subjects_gives_one_subjects_violation(cfg):
    four = "['[[Sys A]]', '[[Sys B]]', '[[Third]]', '[[Fourth]]']"
    put(cfg, "many.md", howto(four, "Delta task"))
    conn, _ = build(cfg)
    ids = {p: i for p, i in conn.execute("SELECT path, id FROM doc")}
    assert violations(conn, "subjects") == [([ids["Kb/many.md"]], "4 subjects")]
    assert conn.execute("SELECT count(*) FROM doc_subject WHERE doc_id = ?", (ids["Kb/many.md"],)).fetchone()[0] == 4
    assert conn.execute("SELECT primary_subject FROM doc WHERE id = ?", (ids["Kb/many.md"],)).fetchone()[0] == "sys-a"


def test_unresolved_owner_of_an_entity_document_gives_one_ref(cfg):
    put(cfg, "entities/Sys A.md", SYS_A.replace("kind: system\n", "kind: system\nowner: '[[ghost]]'\n"))
    conn, _ = build(cfg)
    assert violations(conn, "ref") == [(["sys-a"], "owner: ghost")]


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
    # a broken link in the entity document counts for the entity as the subject
    put(cfg, "entities/Sys A.md", SYS_A + "See [[broken-in-entity]].\n")
    update(cfg, conn, emb)
    own = tools.kb_check(subject="sys-a")
    assert sorted(v["detail"] for v in own["M1"]["violations"]) == ["broken-a", "broken-in-entity"]
    assert tools.kb_check(subject="sys-b")["M1"]["count"] == 1
    with pytest.raises(LookupError):
        tools.kb_check(subject="Nobody Here")


# --- the kind and the validity (V1 to V6) -----------------------------------------------------------------------


def note(front: str, body: str = "Text.") -> str:
    return f"---\n{front}subject: '[[Sys A]]'\n---\n# Doc\n{body}\n"


def ids_of(conn) -> dict[str, str]:
    return {p: i for p, i in conn.execute("SELECT path, id FROM doc")}


def test_the_index_holds_the_kind_and_the_validity(cfg):
    put(cfg, "plan-a.md", note("type: plan\n"))
    put(cfg, "spec-a.md", note("type: spec\nvalidity: rejected\n"))
    put(cfg, "how.md", note("type: how-to\ntask: t\nsuperseded_by: '[[plan-a]]'\n"))
    put(cfg, "rec.md", note("type: research\nclass: state\n"))
    put(cfg, "dec.md", note("type: decision\nas_of: 2026-01-02\n"))
    conn, emb = build(cfg)
    sql = "SELECT path, class, validity FROM doc WHERE path NOT LIKE '%/entities/%'"
    rows = {p: (c, v) for p, c, v in conn.execute(sql)}
    assert rows == {
        "Kb/plan-a.md": ("proposal", "open"),
        "Kb/spec-a.md": ("proposal", "rejected"),
        "Kb/how.md": ("state", "superseded"),
        "Kb/rec.md": ("state", "current"),
        "Kb/dec.md": ("record", None),
    }
    # a change of the validity key alone is a change of the note
    put(cfg, "spec-a.md", note("type: spec\nvalidity: superseded\nsuperseded_by: '[[plan-a]]'\n"))
    update(cfg, conn, emb)
    assert conn.execute("SELECT validity FROM doc WHERE path = 'Kb/spec-a.md'").fetchone()[0] == "superseded"


def test_an_entity_document_keeps_its_entity_kind(cfg):
    room = "---\nid: room-1\ntype: entity\nkind: room\nclass: record\nvalidity: open\n---\n# Room\nA room.\n"
    put(cfg, "entities/Room.md", room)
    conn, _ = build(cfg)
    assert conn.execute("SELECT kind FROM entity WHERE id = 'room-1'").fetchone()[0] == "room"
    row = conn.execute("SELECT class, validity FROM doc WHERE id = 'room-1~doc'").fetchone()
    assert row == ("state", "current")
    for rule in ("V1", "V5", "V6"):
        assert violations(conn, rule) == [], rule


def test_implemented_in_and_followed_by_are_relations_to_a_note_or_an_entity(cfg):
    put(cfg, "how.md", note("type: how-to\ntask: t\n"))
    put(cfg, "rec-2.md", note("type: research\nas_of: 2026-02-01\n"))
    put(cfg, "spec-a.md", note("type: spec\nvalidity: accepted\nimplemented_in: ['[[how]]', '[[Sys B]]']\n"))
    put(cfg, "rec-1.md", note("type: research\nas_of: 2026-01-01\nfollowed_by: '[[rec-2]]'\n"))
    conn, _ = build(cfg)
    ids = ids_of(conn)
    rows = set(conn.execute("SELECT src, rel, dst, dst_text FROM relation"))
    assert rows == {
        (ids["Kb/spec-a.md"], "implemented_in", ids["Kb/how.md"], "how"),
        (ids["Kb/spec-a.md"], "implemented_in", "sys-b", "Sys B"),
        (ids["Kb/rec-1.md"], "followed_by", ids["Kb/rec-2.md"], "rec-2"),
    }
    assert violations(conn, "ref") == []
    assert violations(conn, "vocab") == []
    assert violations(conn, "relation_kinds") == []
    assert violations(conn, "V3") == []


def test_v1_a_note_with_no_kind(cfg):
    put(cfg, "loose.md", note(""))
    put(cfg, "typo.md", note("type: how-to\ntask: t\nclass: sate\n"))
    put(cfg, "given.md", note("type: note\nclass: state\n"))
    conn, _ = build(cfg)
    ids = ids_of(conn)
    assert violations(conn, "V1") == [([ids["Kb/loose.md"]], "class: none"), ([ids["Kb/typo.md"]], "class: sate")]


def test_v2_a_superseded_note_with_no_superseded_by(cfg):
    put(cfg, "how.md", note("type: how-to\ntask: a\nvalidity: superseded\n"))
    put(cfg, "how-ok.md", note("type: how-to\ntask: b\nvalidity: superseded\nsuperseded_by: '[[how]]'\n"))
    put(cfg, "plan-a.md", note("type: plan\nvalidity: superseded\n"))
    conn, _ = build(cfg)
    ids = ids_of(conn)
    assert sorted(i[0] for i, _ in violations(conn, "V2")) == sorted([ids["Kb/how.md"], ids["Kb/plan-a.md"]])


def test_v3_an_accepted_proposal_with_no_implemented_in(cfg):
    put(cfg, "spec-a.md", note("type: spec\nvalidity: accepted\n"))
    put(cfg, "spec-b.md", note("type: spec\nvalidity: accepted\nimplemented_in: '[[Sys B]]'\n"))
    conn, _ = build(cfg)
    assert violations(conn, "V3") == [([ids_of(conn)["Kb/spec-a.md"]], "validity: accepted")]


def test_v4_a_record_with_no_as_of(cfg):
    put(cfg, "rec.md", note("type: research\n"))
    put(cfg, "rec-ok.md", note("type: research\nas_of: 2026-01-01\n"))
    put(cfg, "state.md", note("type: research\nclass: state\n"))
    conn, _ = build(cfg)
    assert violations(conn, "V4") == [([ids_of(conn)["Kb/rec.md"]], None)]


def test_v5_a_validity_that_is_not_valid_for_the_kind(cfg):
    put(cfg, "how.md", note("type: how-to\ntask: t\nvalidity: open\n"))
    put(cfg, "plan-a.md", note("type: plan\nvalidity: current\n"))
    put(cfg, "spec-a.md", note("type: spec\nvalidity: rejected\n"))
    put(cfg, "ref.md", note("type: reference\ntask: r\nvalidity: current\n"))
    conn, _ = build(cfg)
    ids = ids_of(conn)
    expected = [([ids["Kb/how.md"]], "state: open"), ([ids["Kb/plan-a.md"]], "proposal: current")]
    assert sorted(violations(conn, "V5")) == sorted(expected)


def test_v6_a_validity_key_on_a_record(cfg):
    put(cfg, "dec.md", note("type: decision\nas_of: 2026-01-01\nvalidity: accepted\n"))
    put(cfg, "dec-ok.md", note("type: decision\nas_of: 2026-01-01\n"))
    conn, _ = build(cfg)
    assert violations(conn, "V6") == [([ids_of(conn)["Kb/dec.md"]], "validity: accepted")]
    assert violations(conn, "V5") == []


def test_kb_check_scope_keeps_the_notes_under_a_folder_prefix(cfg):
    put(cfg, "Area/rec.md", note("type: research\n", "See [[broken-in]]."))
    put(cfg, "Area/loose.md", note(""))
    put(cfg, "Other/rec.md", note("type: research\n", "See [[broken-out]]."))
    conn, _ = build(cfg)
    every = kb.report(conn)
    assert every["M1"]["count"] == 2
    scoped_all = kb.report(conn, scope="Kb/")
    assert (scoped_all["V4"]["count"], scoped_all["M1"]["count"]) == (2, 2)
    scoped = kb.report(conn, scope="Kb/Area/")
    assert (scoped["V4"]["count"], scoped["V1"]["count"], scoped["M1"]["count"]) == (1, 1, 1)
    assert [v["detail"] for v in scoped["M1"]["violations"]] == ["broken-in"]
    assert [v["nodes"] for v in scoped["V4"]["violations"]] == [[ids_of(conn)["Kb/Area/rec.md"]]]
    assert kb.report(conn, scope="Kb/Nothing/")["V4"]["count"] == 0
    # the scope and the collection filter together
    assert kb.report(conn, "Other", scope="Kb/Area/")["V4"]["count"] == 0


def test_cli_kb_check_takes_a_scope(cfg, capsys):
    from ragnarok.cli import build_parser, cmd_kb

    put(cfg, "Area/rec.md", note("type: research\n"))
    put(cfg, "Other/rec.md", note("type: research\n"))
    build(cfg)[0].close()
    args = build_parser().parse_args(["kb", "check", "--scope", "Kb/Area/", "--ids"])
    assert cmd_kb(cfg, args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["V4"]["count"] == 1
    assert len(out["V4"]["violations"]) == 1


def test_kb_check_with_no_scope_lists_no_v_finding(cfg):
    # the checks of the kind and the validity apply to the notes of a scope only
    put(cfg, "rec.md", note("type: research\n"))
    put(cfg, "loose.md", note(""))
    conn, emb = build(cfg)
    assert violations(conn, "V4") and violations(conn, "V1")
    for result in (kb.report(conn), Tools(cfg, conn, emb).kb_check(), kb.report(conn, collection="Kb")):
        assert not [check for check in result if check.startswith("V")], result.keys()
    assert kb.report(conn, scope="Kb/")["V4"]["count"] == 1


def test_a_custom_relation_table_keeps_the_note_relations(cfg, tmp_path):
    from ragnarok.config import load_config

    path = tmp_path / "config.toml"
    path.write_text("[kb.relation_types.runs]\nsrc_kinds = []\ndst_kinds = []\ninverse = 'runBy'\n")
    custom = load_config(path)
    cfg.kb = custom.kb
    assert set(cfg.kb.relation_types) == {"runs", "implemented_in", "followed_by"}
    put(cfg, "how.md", note("type: how-to\ntask: t\n"))
    put(cfg, "spec-a.md", note("type: spec\nvalidity: accepted\nimplemented_in: '[[how]]'\n"))
    put(cfg, "rec-1.md", note("type: research\nas_of: 2026-01-01\nfollowed_by: '[[how]]'\n"))
    conn, _ = build(cfg)
    assert conn.execute("SELECT count(*) FROM relation").fetchone()[0] == 2
    assert violations(conn, "vocab") == []
