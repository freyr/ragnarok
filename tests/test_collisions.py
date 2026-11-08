import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from test_search import TopicEmbedder

from ragnarok import collisions
from ragnarok.cli import main
from ragnarok.config import Config, SystemOfRecord, VaultRules, WriteConfig, dump_config, load_config
from ragnarok.index import connect, update
from ragnarok.search import PrivateNotAllowed
from ragnarok.tools import Tools

PUMP = "V/Entities/Heat pump.md"
IDENT_STATE = "V/Heating/defrost-a.md"
IDENT_RECORD = "V/Heating/defrost-b.md"
OTHER_TASK = "V/Heating/defrost-c.md"
BOILER = "V/Heating/boiler.md"
WIRING = "V/Heating/wiring.md"
OLD = "V/Heating/old-plan.md"
LINKER = "V/Other/linker.md"
SECRET = "V/Secret/s.md"

NOTES = {
    PUMP: "---\ntype: entity\nkind: device\n---\n# Heat pump\n",
    IDENT_STATE: (
        "---\ntype: guide\nclass: state\nvalidity: current\nsubject: ['[[Heat pump]]']\ntask: defrost\n---\n"
        "# A\nThe eyepiece cleaning.\n"
    ),
    IDENT_RECORD: (
        "---\ntype: guide\nclass: record\nsubject: ['[[Heat pump]]']\ntask: Defrost\n---\n# B\nThe person on call.\n"
    ),
    OTHER_TASK: "---\ntype: guide\nsubject: ['[[Heat pump]]']\ntask: cleaning\n---\n# C\nThe queue length.\n",
    BOILER: "# Boiler\nThe boiler defrost cycle runs at night.\n",
    WIRING: "# Wiring\nThe boiler incident log.\n",
    OLD: "# Old plan\nThe budget plan.\n",
    LINKER: "# Linker\nSee [[old-plan]] for the database.\n",
    SECRET: ("---\ntype: guide\nsubject: ['[[Heat pump]]']\ntask: defrost\n---\n# S\nThe boiler defrost cycle.\n"),
}

DRAFT = (
    "---\ntype: guide\nsubject: ['[[Heat pump]]']\ntask: defrost\nsupersedes: ['[[old-plan]]']\n---\n"
    "# Defrost\nThe boiler defrost cycle. See [[wiring]], [[defrost-a]] and [[s]].\n"
)


@pytest.fixture
def cfg(tmp_path):
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Config(
        root=tmp_path,
        index_path=tmp_path / "index.db",
        vaults=["V"],
        private_patterns=["V/Secret/**"],
        dimension=len(TopicEmbedder.TOPICS),
    )


def write(cfg, notes: dict) -> None:
    for rel, text in notes.items():
        path = cfg.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.fixture
def conn(cfg):
    c = connect(cfg)
    update(cfg, c, TopicEmbedder())
    return c


def run(cfg, conn, **kwargs):
    return collisions.find_collisions(cfg, conn, TopicEmbedder(), **kwargs)


def by_path(candidates):
    return {c["path"]: c for c in candidates}


def drop_vectors(conn, path):
    conn.execute(
        "DELETE FROM chunks_vec WHERE chunk_id IN"
        " (SELECT c.id FROM chunks c JOIN notes n ON n.id = c.note_id WHERE n.path = ?)",
        (path,),
    )


# --- the sources -------------------------------------------------------------------------------------------------


def test_identity_candidates_have_the_same_key_in_any_class(cfg, conn):
    found = by_path(run(cfg, conn, draft=DRAFT))
    assert "identity" in found[IDENT_STATE]["reasons"]
    assert "identity" in found[IDENT_RECORD]["reasons"]  # a record, task in another case
    assert "identity" not in found.get(OTHER_TASK, {"reasons": []})["reasons"]  # another task
    assert "identity" not in found.get(PUMP, {"reasons": []})["reasons"]


@pytest.mark.parametrize("subject", ["", "subject: none\nsubject_reason: a cycle, no device\n"])
def test_a_draft_with_no_subject_has_no_identity_candidates(cfg, conn, subject):
    draft = f"---\ntype: guide\n{subject}task: defrost\n---\n# Defrost\nThe boiler defrost cycle.\n"
    assert all("identity" not in c["reasons"] for c in run(cfg, conn, draft=draft))


def test_search_candidates_keep_their_rank_and_distance(cfg, conn):
    found = by_path(run(cfg, conn, draft=DRAFT))
    assert found[BOILER]["reasons"] == ["search"]
    assert found[BOILER]["rank"] == 1
    assert found[BOILER]["distance"] == pytest.approx(0.0, abs=1e-6)  # the same topic vector (boiler)


def test_search_includes_superseded_notes(cfg, conn):
    write(cfg, {BOILER: "---\nclass: state\nvalidity: superseded\n---\n# Boiler\nThe boiler defrost cycle.\n"})
    update(cfg, conn, TopicEmbedder())
    found = by_path(run(cfg, conn, draft=DRAFT))
    assert found[BOILER]["reasons"] == ["search"]
    assert found[BOILER]["validity"] == "superseded"


def test_links_out_are_candidates_with_relative_links(cfg, conn):
    draft = "---\ntype: guide\nsee: '[[wiring]]'\n---\n# D\nThe text. See [[../Other/linker]].\n"
    found = by_path(run(cfg, conn, draft=draft, draft_path="V/Heating/new.md"))
    assert "links_out" in found[WIRING]["reasons"]  # a frontmatter link
    assert "links_out" in found[LINKER]["reasons"]  # a relative body link


def test_the_notes_that_link_to_a_supersedes_target_are_candidates(cfg, conn):
    found = by_path(run(cfg, conn, draft=DRAFT))
    assert "links_to_supersedes" in found[OLD]["reasons"]
    assert "links_to_supersedes" in found[LINKER]["reasons"]
    assert "links_out" not in found[OLD]["reasons"]  # supersedes is not a link of links_out


def test_a_candidate_is_listed_once_with_all_its_reasons(cfg, conn):
    result = run(cfg, conn, draft=DRAFT)
    paths = [c["path"] for c in result]
    assert len(paths) == len(set(paths))
    assert by_path(result)[IDENT_STATE]["reasons"] == ["identity", "search", "links_out"]


def test_each_candidate_has_its_labels(cfg, conn):
    c = by_path(run(cfg, conn, draft=DRAFT))[IDENT_STATE]
    assert {k: c[k] for k in ("title", "type", "status", "class", "validity", "as_of", "subject", "superseded_by")} == {
        "title": "defrost-a",
        "type": "guide",
        "status": "active",
        "class": "state",
        "validity": "current",
        "as_of": None,
        "subject": ["Heat pump"],
        "superseded_by": None,
    }


def test_a_note_with_no_vector_has_no_distance(cfg, conn):
    drop_vectors(conn, WIRING)
    assert by_path(run(cfg, conn, draft=DRAFT))[WIRING]["distance"] is None


def test_one_query_embedding(cfg, conn):
    class Counting(TopicEmbedder):
        calls = 0

        def embed_query(self, query):
            Counting.calls += 1
            return super().embed_query(query)

    collisions.find_collisions(cfg, conn, Counting(), draft=DRAFT)
    assert Counting.calls == 1


# --- the order and the limit -------------------------------------------------------------------------------------


@pytest.fixture
def fake_search(monkeypatch):
    """A search that gives these paths in this order, cut at the limit of the call."""
    calls = []

    def install(paths):
        def fake(cfg, conn, embedder, query, **kwargs):
            embedder.embed_query(query)
            calls.append(kwargs)
            return [SimpleNamespace(path=p) for p in paths[: kwargs["limit"]]]

        monkeypatch.setattr(collisions, "search", fake)
        return calls

    return install


@pytest.fixture
def fillers(cfg, conn):
    paths = [f"V/Fill/f{i:02}.md" for i in range(20)]
    write(cfg, {p: f"# F\nThe filler {p[-6:-3]}.\n" for p in paths})
    update(cfg, conn, TopicEmbedder())
    return paths


def test_tier_one_then_search_rank_then_link_only_by_distance(cfg, conn, fake_search):
    calls = fake_search([BOILER, OTHER_TASK, IDENT_RECORD, LINKER])
    result = run(cfg, conn, draft=DRAFT)
    # tier 1 (identity and the supersedes target): rank 3, then no rank by distance (1.0) and path;
    # search: ranks 1, 2, 4 (linker is a search candidate); link only: wiring
    assert [c["path"] for c in result] == [IDENT_RECORD, IDENT_STATE, OLD, BOILER, OTHER_TASK, LINKER, WIRING]
    assert [c["rank"] for c in result] == [3, None, None, 1, 2, 4, None]
    assert result[6]["distance"] == pytest.approx(1 - 2**-0.5, abs=1e-6)  # boiler of boiler and incident
    assert calls[0]["include_superseded"] is True
    assert calls[0]["include_private"] is False


def test_a_link_only_candidate_with_no_distance_is_last(cfg, conn, fake_search):
    fake_search([])
    drop_vectors(conn, WIRING)
    result = run(cfg, conn, draft=DRAFT)
    assert [c["path"] for c in result][-2:] == [LINKER, WIRING]


def test_at_most_ten_candidates(cfg, conn, fake_search, fillers):
    calls = fake_search(fillers)
    result = run(cfg, conn, draft=DRAFT)
    # tier 1: 3; link only: wiring (0.29), linker (1.0); search: the other 5 places
    assert [c["path"] for c in result] == [IDENT_STATE, IDENT_RECORD, OLD, *fillers[:5], WIRING, LINKER]
    assert calls[0]["limit"] >= 10 + 3


def test_a_supersedes_target_beyond_the_search_pool_is_first(cfg, conn, fake_search, fillers):
    draft = "---\ntype: guide\nsupersedes: ['[[old-plan]]']\n---\n# D\nThe boiler text.\n"
    calls = fake_search([*fillers, OLD])  # the target at rank 21, outside any pool of the call
    result = run(cfg, conn, draft=draft)
    assert result[0]["path"] == OLD
    assert result[0]["rank"] is None
    assert result[-1]["path"] == LINKER  # it links to the target: link only
    assert [c["path"] for c in result[1:-1]] == fillers[:8]
    assert calls[0]["limit"] >= 11


def test_a_links_out_note_outside_the_search_results_is_kept(cfg, conn, fake_search, fillers):
    fake_search(fillers)
    result = run(cfg, conn, draft="---\ntype: guide\n---\n# D\nSee [[wiring]].\n")
    assert [c["path"] for c in result] == [*fillers[:9], WIRING]
    assert result[-1]["reasons"] == ["links_out"]


def test_at_most_three_link_only_candidates_by_distance(cfg, conn, fake_search, fillers):
    fake_search(fillers)
    draft = "---\ntype: guide\n---\n# D\nThe queue. [[wiring]] [[linker]] [[boiler]] [[defrost-c]] [[old-plan]]\n"
    result = run(cfg, conn, draft=draft)
    # the query has boiler and queue: boiler 0.29, defrost-c (queue) 0.29, wiring 0.5, linker and old-plan 1.0
    assert [c["path"] for c in result] == [*fillers[:7], BOILER, OTHER_TASK, WIRING]


def test_with_no_link_candidates_the_search_fills_the_places(cfg, conn, fake_search, fillers):
    fake_search(fillers)
    assert [c["path"] for c in run(cfg, conn, draft="# D\nThe text.\n")] == fillers[:10]
    subject = "---\ntype: guide\nsubject: ['[[Heat pump]]']\ntask: defrost\n---\n# D\nThe text.\n"
    assert [c["path"] for c in run(cfg, conn, draft=subject)] == [IDENT_STATE, IDENT_RECORD, *fillers[:8]]


# --- the input note, a copy and privacy --------------------------------------------------------------------------


def test_the_input_note_is_not_a_candidate(cfg, conn):
    result = by_path(run(cfg, conn, path=IDENT_STATE))
    assert IDENT_STATE not in result
    assert "identity" in result[IDENT_RECORD]["reasons"]


def test_a_path_must_be_an_indexed_note(cfg, conn):
    with pytest.raises(FileNotFoundError, match="not a note in the index"):
        run(cfg, conn, path="V/Heating/none.md")


def test_a_copy_of_a_note_gives_that_note_first(cfg, conn):
    result = run(cfg, conn, draft=NOTES[IDENT_STATE])
    assert result[0]["path"] == IDENT_STATE
    assert result[0]["reasons"][0] == "identity"
    assert result[0]["distance"] == pytest.approx(0.0, abs=1e-6)
    copy = run(cfg, conn, draft=NOTES[BOILER])
    assert copy[0]["path"] == BOILER
    assert copy[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_a_private_note_is_no_candidate_by_default(cfg, conn):
    assert SECRET not in by_path(run(cfg, conn, draft=DRAFT))  # same key, same text, linked
    with pytest.raises(PrivateNotAllowed):
        run(cfg, conn, draft=DRAFT, allow_private=True)
    cfg.allow_private = True
    assert SECRET not in by_path(run(cfg, conn, draft=DRAFT))
    found = by_path(run(cfg, conn, draft=DRAFT, allow_private=True))
    assert found[SECRET]["reasons"] == ["identity", "search", "links_out"]


# --- the MCP tool and the CLI ------------------------------------------------------------------------------------


def test_mcp_tool_takes_a_draft_or_an_indexed_path(cfg, conn):
    from ragnarok.server import build

    tools = Tools(cfg, conn, TopicEmbedder())
    assert BOILER in by_path(tools.find_collisions(draft=DRAFT)["candidates"])
    assert IDENT_STATE not in by_path(tools.find_collisions(path=IDENT_STATE)["candidates"])
    with pytest.raises(FileNotFoundError):
        tools.find_collisions(path="V/Heating/none.md")
    with pytest.raises(PrivateNotAllowed):
        tools.find_collisions(path=SECRET)
    with pytest.raises(ValueError):
        tools.find_collisions()
    with pytest.raises(ValueError):
        tools.find_collisions(draft=DRAFT, path="V/../outside.md")
    out = asyncio.run(build(tools=tools).call_tool("find_collisions", {"draft": DRAFT}))
    data = json.loads(out.content[0].text)
    assert data["candidates"][0]["reasons"][0] == "identity"


def config_file(cfg, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path))
    return path


@pytest.fixture
def cli(cfg, conn, tmp_path, monkeypatch):
    conn.close()
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda c: TopicEmbedder())
    return str(config_file(cfg, tmp_path))


def test_cli_path_prints_json(cli, capsys):
    assert main(["--config", cli, "find-collisions", IDENT_STATE, "--json"]) == 0
    paths = [c["path"] for c in json.loads(capsys.readouterr().out)["candidates"]]
    assert IDENT_STATE not in paths
    assert IDENT_RECORD in paths


def test_cli_reads_a_draft_from_stdin(cli, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(NOTES[BOILER]))
    assert main(["--config", cli, "find-collisions", "-"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith(f"1. {BOILER}")
    assert "search" in first


def test_cli_unknown_path_fails(cli, capsys):
    assert main(["--config", cli, "find-collisions", "V/none.md"]) == 1
    assert "not a note in the index" in capsys.readouterr().err


# --- vault_rules -------------------------------------------------------------------------------------------------


def test_vault_rules_without_a_write_section(cfg, conn):
    result = Tools(cfg, conn, TopicEmbedder()).vault_rules()
    assert result["write"] == {"known_keys": [], "systems_of_record": []}


def test_vault_rules_give_the_path_rules_of_each_vault_and_the_write_section(cfg, conn, tmp_path):
    cfg.vaults = ["V", "W"]
    cfg.vault_paths = {"W": tmp_path / "elsewhere" / "kb"}
    cfg.private_patterns = ["V/Secret/**", "**/private/**", "W/x/**"]
    cfg.vault_rules = {"V": VaultRules(project_depth=1, ignore_folders=["docs"])}
    cfg.write = WriteConfig(
        known_keys=["tags"],
        systems_of_record=[SystemOfRecord("client tier", "CRM", "crm-read"), SystemOfRecord("invoices", "ERP")],
    )
    result = Tools(cfg, conn, TopicEmbedder()).vault_rules()
    common = {"project_markers": ["DRIVERS.md", "SUMMARY.md"], "archive_folder": "_archive"}
    defaults = {
        "area": "folder",
        "area_default": "",
        "project_fallback": True,
        "project_parents": [],
        "jira_prefixes": [],
    }
    assert result["vaults"] == [
        {
            "name": "V",
            "folder": "V",
            **defaults,
            "project_depth": 1,
            "ignore_folders": ["docs"],
            **common,
            "private_patterns": ["V/Secret/**", "**/private/**"],
        },
        {
            "name": "W",
            "folder": "kb",
            **defaults,
            "project_depth": 2,
            "ignore_folders": [],
            **common,
            "private_patterns": ["**/private/**", "W/x/**"],
        },
    ]
    assert str(tmp_path) not in json.dumps(result)
    assert result["write"] == {
        "known_keys": ["tags"],
        "systems_of_record": [
            {"data": "client tier", "system": "CRM", "skill": "crm-read"},
            {"data": "invoices", "system": "ERP", "skill": None},
        ],
    }


def test_systems_of_record_load_and_round_trip(cfg, tmp_path):
    from ragnarok.bootstrap import example_config

    assert Config().write.systems_of_record == []
    assert load_config(example_config()).write.systems_of_record == []
    path = tmp_path / "c.toml"
    path.write_text(
        'vaults = ["V"]\n\n[write]\nknown_keys = ["tags"]\n\n'
        '[[write.systems_of_record]]\ndata = "client tier"\nsystem = "CRM"\nskill = "crm-read"\n\n'
        '[[write.systems_of_record]]\ndata = "invoices"\nsystem = "ERP"\n'
    )
    loaded = load_config(path).write
    assert loaded == WriteConfig(
        known_keys=["tags"],
        systems_of_record=[SystemOfRecord("client tier", "CRM", "crm-read"), SystemOfRecord("invoices", "ERP")],
    )
    cfg.write = loaded
    again = tmp_path / "again.toml"
    again.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path))
    assert load_config(again).write == loaded


def test_mcp_server_gives_vault_rules(cfg, conn):
    from ragnarok.server import build

    out = asyncio.run(build(tools=Tools(cfg, conn, TopicEmbedder())).call_tool("vault_rules", {}))
    data = json.loads(out.content[0].text)
    assert [v["name"] for v in data["vaults"]] == ["V"]
    assert data["write"] == {"known_keys": [], "systems_of_record": []}
