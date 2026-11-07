import json

import pytest
from test_index import FakeEmbedder

from ragnarok import validate
from ragnarok.cli import main
from ragnarok.config import Config, WriteConfig, dump_config, load_config
from ragnarok.index import connect, update

PUMP = "V/Entities/Heat pump.md"
ANNA = "V/Entities/Anna.md"
BOILER = "V/Heating/boiler.md"
NOTE = "V/Heating/guide.md"
DRAFT_LABEL = "(draft)"

BASE = {
    PUMP: "---\ntype: entity\nkind: device\naliases: [HP]\nfacts: [a fact]\n---\n# Heat pump\n",
    ANNA: "---\ntype: entity\nkind: person\n---\n# Anna\n",
    BOILER: "# Boiler\nThe boiler heats the water.\n",
}

GOOD = (
    "---\ntype: guide\nclass: state\nvalidity: current\nas_of: 2026-01-05\nsubject: ['[[Heat pump]]']\n"
    "task: defrost\nowner: '[[Anna]]'\nproject: heating\narea: home\nstatus: active\ndescription: How to.\n---\n"
    "# Defrost\nSee [[boiler]].\n"
)


@pytest.fixture
def cfg(tmp_path):
    for rel, text in BASE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["V"], dimension=8)


def write(cfg, notes: dict) -> None:
    for rel, text in notes.items():
        path = cfg.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def index(cfg):
    conn = connect(cfg)
    update(cfg, conn, FakeEmbedder())
    return conn


def check_note(cfg, text, rel=NOTE, extra=None):
    write(cfg, {rel: text, **(extra or {})})
    return validate.check_paths(cfg, index(cfg), [rel])


def found(report, check=None, severity="error"):
    """(path, check, kind) of the issues of one check (all checks when None) and one severity."""
    return [
        (i.path, i.check, i.kind)
        for i in report.issues
        if (check is None or i.check == check) and i.severity == severity
    ]


# --- check 1: keys and values of the model -----------------------------------------------------------------------


def test_a_note_with_all_model_keys_has_no_issue(cfg):
    report = check_note(cfg, GOOD)
    assert report.issues == []
    assert report.errors == 0


@pytest.mark.parametrize(
    "lines, kind",
    [
        ("class: idea\n", "bad-class"),
        ("class: record\nvalidity: current\n", "bad-validity"),
        ("class: state\nvalidity: accepted\n", "bad-validity"),
        ("class: proposal\nvalidity: current\n", "bad-validity"),
        ("class: state\nvalidity: current\nas_of: 05.01.2026\n", "bad-as-of"),
        ("class: state\nvalidity: current\nas_of: '2026-02-30'\n", "bad-as-of"),
        ("class: state\nvalidity: current\nas_of: 2026-01-05 10:00:00\n", "bad-as-of"),
        ("class: state\nvalidity: current\nsubject: '[[Heat pump]]'\n", "bad-subject"),
        ("class: state\nvalidity: current\nsubject: [Heat pump]\n", "bad-subject"),
        ("class: record\nsupersedes: '[[boiler]]'\n", "bad-supersedes"),
        ("class: state\nvalidity: superseded\nsuperseded_by: ['[[boiler]]', '[[Anna]]']\n", "bad-superseded-by"),
        ("class: state\nvalidity: current\nowner: Anna\n", "bad-owner"),
        ("class: state\nvalidity: current\ntask: [a, b]\n", "bad-task"),
    ],
)
def test_a_wrong_value_of_the_model_is_an_error(cfg, lines, kind):
    report = check_note(cfg, f"---\n{lines}---\n# Note\n")
    assert found(report, 1) == [(NOTE, 1, kind)]


def test_an_unquoted_wiki_link_is_a_link(cfg):
    text = "---\nclass: state\nvalidity: current\nsubject: [[Heat pump]]\nowner: [[Anna]]\n---\n# Note\n"
    assert check_note(cfg, text).issues == []


def test_malformed_frontmatter_is_an_error(cfg):
    report = check_note(cfg, "---\nclass: [state\n---\n# Note\n")
    assert (NOTE, 1, "bad-yaml") in found(report)


# --- check 2: required keys ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines, kinds",
    [
        ("", ["no-class"]),
        ("type: guide\n", ["no-class"]),  # the default class from type is not an explicit class
        ("class: state\n", ["no-validity"]),
        ("class: proposal\n", ["no-validity"]),
        ("class: record\n", []),
        ("class: state\nvalidity: current\n", []),
    ],
)
def test_a_given_path_needs_class_and_validity(cfg, lines, kinds):
    report = check_note(cfg, f"---\n{lines}---\n# Note\n")
    assert [k for _, _, k in found(report, 2)] == kinds


def test_the_scan_does_not_count_a_missing_class_as_an_error(cfg):
    write(cfg, {NOTE: "---\ntype: guide\n---\n# Note\n", "V/Heating/plan.md": "---\nclass: proposal\n---\n# P\n"})
    report = validate.check_index(cfg, index(cfg))
    assert found(report, 2) == []
    assert report.errors == 0
    # the two entities, the boiler, the guide and the plan: none has a complete label
    assert report.counts["unlabeled"] == 5
    assert report.counts["m4"] == {"share": 1.0, "passed": 5, "notes": 5}


# --- check 3: unknown keys ------------------------------------------------------------------------------------------


def test_an_unknown_key_is_a_warning(cfg):
    report = check_note(cfg, "---\nclass: record\ncolour: red\n---\n# Note\n")
    assert found(report, 3, "warning") == [(NOTE, 3, "unknown-key")]
    assert report.errors == 0
    assert report.warnings == 1


def test_a_known_key_of_the_config_is_no_warning(cfg):
    cfg.write = WriteConfig(known_keys=["colour"])
    report = check_note(cfg, "---\nclass: record\ncolour: red\n---\n# Note\n")
    assert report.issues == []


def test_entity_keys_are_known_only_on_an_entity_note(cfg):
    write(cfg, {NOTE: "---\nclass: record\nkind: device\n---\n# Note\n"})
    report = validate.check_paths(cfg, index(cfg), [PUMP, NOTE])
    assert found(report, 3, "warning") == [(NOTE, 3, "unknown-key")]


def test_the_default_list_of_known_keys_is_empty(tmp_path):
    from ragnarok.bootstrap import example_config

    assert Config().write.known_keys == []
    assert load_config(example_config()).write.known_keys == []
    path = tmp_path / "config.toml"
    path.write_text('vaults = ["V"]\n\n[write]\nknown_keys = ["tags", "created"]\n')
    assert load_config(path).write.known_keys == ["tags", "created"]


def test_known_keys_round_trip_through_toml(cfg, tmp_path):
    cfg.write = WriteConfig(known_keys=["tags"])
    path = tmp_path / "copy.toml"
    path.write_text(dump_config(cfg, root="vaults"))
    assert load_config(path).write == cfg.write


# --- check 4: wiki-links -----------------------------------------------------------------------------------------


def test_a_link_that_resolves_is_no_error(cfg):
    report = check_note(cfg, "---\nclass: record\nparent: '[[Anna]]'\n---\n# Note\nSee [[boiler|the boiler]].\n")
    assert found(report) == []  # parent is an unknown key: a warning only


@pytest.mark.parametrize(
    "text",
    [
        "---\nclass: record\n---\n# Note\nSee [[Nowhere]].\n",
        "---\nclass: record\n---\n# Note\nSee [[Nowhere#Part|there]].\n",
        "---\nclass: record\nparent: '[[Nowhere]]'\n---\n# Note\n",
        "---\nclass: record\nproject: '[[Nowhere/SUMMARY]]'\n---\n# Note\n",
    ],
)
def test_a_broken_link_is_an_error(cfg, text):
    assert found(check_note(cfg, text), 4) == [(NOTE, 4, "broken-link")]


def test_links_in_code_and_attachments_are_not_checked(cfg):
    text = (
        "---\nclass: record\n---\n# Note\n```\n[[Nowhere]]\n```\n~~~md\n[[Nowhere 2]]\n~~~\n"
        "Inline `[[Nowhere 3]]` and ``a [[Nowhere 4]] b``.\n![[diagram.png]]\n"
    )
    assert check_note(cfg, text).issues == []


@pytest.mark.parametrize("target", ["../Heating/boiler", "./boiler", "../Entities/Anna"])
def test_a_relative_link_resolves_from_the_folder_of_the_note(cfg, target):
    report = check_note(cfg, f"---\nclass: record\n---\n# Note\nSee [[{target}]].\n")
    assert report.issues == []
    draft = validate.check_draft(cfg, index(cfg), f"---\nclass: record\n---\n# D\n[[{target}]]\n", "V/Heating/new.md")
    assert draft.issues == []


def test_a_relative_link_to_no_note_is_an_error(cfg):
    report = check_note(cfg, "---\nclass: record\n---\n# Note\nSee [[../boiler]].\n")  # V/boiler.md is no note
    assert found(report, 4) == [(NOTE, 4, "broken-link")]
    draft = validate.check_draft(cfg, index(cfg), "---\nclass: record\n---\n# D\n[[./boiler]]\n")
    assert found(draft, 4) == [(DRAFT_LABEL, 4, "broken-link")]  # no intended path: no folder


# --- check 5: reference targets ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines, key",
    [
        ("class: state\nvalidity: current\nsubject: ['[[Nothing]]']\n", "subject"),
        ("class: state\nvalidity: current\nsubject: ['[[boiler]]']\n", "subject"),  # not an entity
        ("class: state\nvalidity: current\nowner: '[[Nothing]]'\n", "owner"),
        ("class: state\nvalidity: current\nowner: '[[boiler]]'\n", "owner"),  # not an entity
        ("class: record\nsupersedes: ['[[Nothing]]']\n", "supersedes"),
        ("class: state\nvalidity: superseded\nsuperseded_by: '[[Nothing]]'\n", "superseded_by"),
    ],
)
def test_a_missing_reference_target_is_an_error(cfg, lines, key):
    report = check_note(cfg, f"---\n{lines}---\n# Note\n")
    assert found(report, 5) == [(NOTE, 5, "missing-target")]
    assert key in report.issues[0].message


def test_reference_targets_resolve_by_path_and_alias(cfg):
    text = (
        "---\nclass: state\nvalidity: current\nsubject: ['[[HP]]', '[[Entities/Heat pump]]']\n"
        "owner: '[[anna]]'\nsupersedes: ['[[boiler]]']\n---\n# Note\n"
    )
    assert found(check_note(cfg, text), 5) == []


def test_an_owner_must_be_a_person_or_a_team(cfg):
    report = check_note(cfg, "---\nclass: state\nvalidity: current\nowner: '[[HP]]'\n---\n# Note\n")
    assert found(report, 5) == [(NOTE, 5, "bad-owner-kind")]


# --- check 6: superseded_by needs a matching supersedes ------------------------------------------------------------

OLD = "---\nclass: state\nvalidity: superseded\nsuperseded_by: '[[new]]'\n---\n# Old\n"


def test_a_successor_that_lists_the_note_is_no_error(cfg):
    new = "---\nclass: state\nvalidity: current\nsupersedes: ['[[guide]]']\n---\n# New\n"
    report = check_note(cfg, OLD, extra={"V/Heating/new.md": new})
    assert report.issues == []


def test_a_successor_that_does_not_list_the_note_is_an_error(cfg):
    new = "---\nclass: state\nvalidity: current\nsupersedes: ['[[boiler]]']\n---\n# New\n"
    report = check_note(cfg, OLD, extra={"V/Heating/new.md": new})
    assert found(report, 6) == [(NOTE, 6, "no-matching-supersedes")]


# --- check 7: double identity --------------------------------------------------------------------------------------


def state(subject="[[Heat pump]]", task=None, more=""):
    task_line = f"task: {task}\n" if task else ""
    return f"---\ntype: guide\nclass: state\nvalidity: current\nsubject: ['{subject}']\n{task_line}{more}---\n# S\n"


TWIN = "V/Heating/twin.md"


@pytest.mark.parametrize(
    "first, second",
    [
        (state(), state()),
        (state(), state(subject="[[HP]]")),  # an alias resolves to the same entity
        (state(task="Defrost"), state(task="defrost ")),
        # the default class of a guide is state, and no validity counts as current
        (state(), "---\ntype: guide\nsubject: ['[[Heat pump]]']\n---\n# S\n"),
    ],
)
def test_two_current_state_notes_with_one_identity_key_are_an_error(cfg, first, second):
    write(cfg, {NOTE: first, TWIN: second})
    conn = index(cfg)
    assert found(validate.check_paths(cfg, conn, [NOTE]), 7) == [(NOTE, 7, "double-identity")]
    report = validate.check_index(cfg, conn)
    assert sorted(found(report, 7)) == [(NOTE, 7, "double-identity"), (TWIN, 7, "double-identity")]
    assert report.counts["m3"] == 1


@pytest.mark.parametrize(
    "second",
    [
        state(task="install"),
        state(subject="[[Anna]]"),
        state(more="status: archived\n").replace("validity: current", "validity: current"),
        state().replace("validity: current", "validity: superseded"),
        state().replace("class: state", "class: proposal").replace("validity: current", "validity: open"),
        state().replace("type: guide", "type: reference"),
        "---\ntype: guide\nclass: state\nvalidity: current\n---\n# S\n",  # no subject: no identity key
    ],
)
def test_other_identity_keys_and_other_notes_are_no_error(cfg, second):
    write(cfg, {NOTE: state(), TWIN: second})
    conn = index(cfg)
    assert found(validate.check_paths(cfg, conn, [NOTE]), 7) == []
    report = validate.check_index(cfg, conn)
    assert found(report, 7) == []
    assert report.counts["m3"] == 0


# --- draft ---------------------------------------------------------------------------------------------------------


def test_a_valid_draft_has_no_issue(cfg):
    report = validate.check_draft(cfg, index(cfg), GOOD, "V/Heating/new.md")
    assert report.issues == []
    assert report.notes == 1


def test_a_draft_needs_class_and_valid_links(cfg):
    report = validate.check_draft(cfg, index(cfg), "---\ntype: guide\n---\n# D\nSee [[Nowhere]].\n")
    assert found(report) == [("(draft)", 2, "no-class"), ("(draft)", 4, "broken-link")]


def test_a_draft_at_the_path_of_its_note_replaces_it_in_the_identity_check(cfg):
    write(cfg, {NOTE: state()})
    conn = index(cfg)
    assert found(validate.check_draft(cfg, conn, state(), NOTE), 7) == []
    assert found(validate.check_draft(cfg, conn, state(), TWIN), 7) == [(TWIN, 7, "double-identity")]


def test_a_draft_link_resolves_with_and_without_an_intended_path(cfg):
    write(cfg, {"V/Other/boiler.md": "# Other boiler\n", "V/Other/x.md": "# X\n"})
    conn = index(cfg)
    text = "---\nclass: record\n---\n# D\nSee [[boiler]].\n"
    assert validate.check_draft(cfg, conn, text, "V/Other/new.md").issues == []
    assert validate.check_draft(cfg, conn, text).issues == []


# --- the scan counts, the CLI and the MCP tool ---------------------------------------------------------------------


SCAN = {
    # broken body link (M1) and a missing subject (M1); an unknown key (warning)
    "V/A/one.md": "---\nclass: record\ncolour: red\nsubject: ['[[Nothing]]']\n---\n# One\n[[Nowhere]]\n",
    # a successor that does not list it (check 6)
    "V/A/two.md": "---\nclass: state\nvalidity: superseded\nsuperseded_by: '[[three]]'\n---\n# Two\n",
    "V/A/three.md": "---\nclass: state\nvalidity: current\n---\n# Three\n",
    # two current state notes for one key (M3)
    "V/A/four.md": state(),
    "V/A/five.md": state(),
}


def test_the_scan_gives_the_counts(cfg):
    write(cfg, SCAN)
    report = validate.check_index(cfg, index(cfg))
    # notes: 3 base + 5; errors in one, two, four, five
    assert report.counts == {
        "m1": 2,
        "broken_links": 1,
        "missing_targets": 1,
        "m3": 1,
        "m4": {"share": 0.5, "passed": 4, "notes": 8},
        "unlabeled": 3,
        "warnings": 1,
        "check6": 1,
    }
    assert report.errors == 5


def config_file(cfg, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path))
    return path


def test_cli_scan_prints_json_and_fails_on_an_error(cfg, tmp_path, capsys):
    write(cfg, SCAN)
    index(cfg).close()
    code = main(["--config", str(config_file(cfg, tmp_path)), "validate", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["counts"]["m3"] == 1
    assert {"path", "check", "severity", "kind", "message"} <= set(out["issues"][0])


def test_cli_paths_print_text_and_pass(cfg, tmp_path, capsys):
    write(cfg, {NOTE: GOOD})
    index(cfg).close()
    absolute = str(cfg.root / NOTE)
    code = main(["--config", str(config_file(cfg, tmp_path)), "validate", absolute, BOILER])
    out = capsys.readouterr().out
    assert code == 1  # the boiler has no class
    assert f"{BOILER}: error [2]" in out
    assert NOTE not in out.replace(f"{BOILER}", "")
    write(cfg, {BOILER: "---\nclass: record\n---\n# Boiler\n"})
    index(cfg).close()
    assert main(["--config", str(config_file(cfg, tmp_path)), "validate", absolute, BOILER]) == 0


def test_cli_unknown_path_fails(cfg, tmp_path, capsys):
    index(cfg).close()
    assert main(["--config", str(config_file(cfg, tmp_path)), "validate", "V/none.md"]) == 1
    assert "not a note" in capsys.readouterr().err


def test_mcp_tool_validates_a_draft_and_paths(cfg):
    import asyncio

    from ragnarok.server import build
    from ragnarok.tools import Tools

    write(cfg, {NOTE: GOOD})
    conn = index(cfg)
    tools = Tools(cfg, conn, FakeEmbedder())
    result = tools.validate(draft="---\nclass: record\n---\n# D\n[[Nowhere]]\n")
    assert [(i["check"], i["kind"]) for i in result["issues"]] == [(4, "broken-link")]
    assert tools.validate(paths=[NOTE])["error_count"] == 0
    server = build(tools=tools)
    out = asyncio.run(server.call_tool("validate", {"paths": [BOILER]}))
    data = json.loads(out.content[0].text)
    assert [(i["path"], i["kind"]) for i in data["issues"]] == [(BOILER, "no-class")]


def test_mcp_tool_refuses_a_private_path_and_no_input(cfg):
    from ragnarok.search import PrivateNotAllowed
    from ragnarok.tools import Tools

    cfg.private_patterns = ["V/Heating/**"]
    tools = Tools(cfg, index(cfg), FakeEmbedder())
    with pytest.raises(PrivateNotAllowed):
        tools.validate(paths=[BOILER])
    with pytest.raises(ValueError):
        tools.validate()


@pytest.fixture
def secret(cfg):
    write(cfg, {"V/Secret/s.md": "---\nclass: SECRETVALUE\n---\n# S\n[[secret-target]]\n"})
    (cfg.root / "outside.md").write_text("---\nclass: OUTSIDE\n---\n# O\n")
    cfg.private_patterns = ["V/Secret/**"]
    return cfg


@pytest.mark.parametrize("path", ["V/Heating/../Secret/s.md", "V/../outside.md", "outside.md", "V/Heating/none.md"])
def test_mcp_tool_refuses_a_path_that_is_not_an_indexed_note(secret, path):
    from ragnarok.tools import Tools

    tools = Tools(secret, index(secret), FakeEmbedder())
    with pytest.raises(FileNotFoundError, match="not a note in the index"):
        tools.validate(paths=[path])


def test_mcp_tool_refuses_an_indexed_private_note_and_reads_an_indexed_public_note(secret):
    from ragnarok.search import PrivateNotAllowed
    from ragnarok.tools import Tools

    tools = Tools(secret, index(secret), FakeEmbedder())
    with pytest.raises(PrivateNotAllowed):
        tools.validate(paths=["V/Secret/s.md"])
    with pytest.raises(PrivateNotAllowed):
        tools.validate(paths=[str(secret.root / "V/Heating/../Secret/s.md")])
    assert [i["kind"] for i in tools.validate(paths=[BOILER])["issues"]] == ["no-class"]


def test_mcp_draft_never_reads_the_file_at_its_intended_path(secret):
    from ragnarok.tools import Tools

    tools = Tools(secret, index(secret), FakeEmbedder())
    result = tools.validate(draft="---\nclass: record\n---\n# D\n", path="V/Secret/s.md")
    assert result["issues"] == []
    with pytest.raises(ValueError):
        tools.validate(draft="---\nclass: record\n---\n# D\n", path="V/../outside.md")
