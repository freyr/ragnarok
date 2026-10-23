import asyncio
import dataclasses

import pytest
from test_search import EXTRA, TopicEmbedder

from ragnarok.index import connect, update
from ragnarok.search import PrivateNotAllowed
from ragnarok.server import build
from ragnarok.tools import Tools

MORE = {
    "Home/Heating/SUMMARY.md": (
        "---\nstage: DELIVER\nnext: Wire the valves to the boiler relay.\n---\n# Heating\n\n"
        "Zone control for the boiler and the radiators.\n\nSee [[boiler]] and [[2026]].\n"
    ),
    "Home/Heating/notes/boiler.md": (
        "# Boiler\nThe boiler flow temperature follows the weather curve.\n\n"
        "## Wiring\nThe relay password: hunter2hunter2 is on the label.\n"
    ),
}


@pytest.fixture
def tools(vault):
    for rel, text in {**EXTRA, **MORE}.items():
        (vault.root / rel).write_text(text)
    vault.dimension = len(TopicEmbedder.TOPICS)
    conn = connect(vault)
    emb = TopicEmbedder()
    update(vault, conn, emb)
    return Tools(vault, conn, emb)


def test_search_excludes_private_and_refuses_include_private(tools):
    paths = [r["path"] for r in tools.search("boiler")["results"]]
    assert "Home/Heating/notes/boiler.md" in paths
    assert "Home/Budget/2026.md" not in paths
    with pytest.raises(PrivateNotAllowed):
        tools.search("boiler", include_private=True)


def test_find_projects_gives_the_project_summary(tools):
    result = tools.find_projects("boiler weather curve")["results"]
    top = result[0]
    assert (top["vault"], top["project"], top["folder"]) == ("Home", "Heating", "Home/Heating")
    assert top["stage"] == "DELIVER"
    assert top["next"] == "Wire the valves to the boiler relay."
    assert top["description"] == "Zone control for the boiler and the radiators."
    assert "Home/Heating/notes/boiler.md" in top["matching_notes"]


def test_list_project(tools):
    result = tools.list_project("Heating")["results"]
    assert len(result) == 1
    assert [n["path"] for n in result[0]["notes"]] == ["Home/Heating/SUMMARY.md", "Home/Heating/notes/boiler.md"]


def test_open_note_redacts_and_hides_private_links(tools):
    note = tools.open_note("Home/Heating/SUMMARY.md")
    assert note["frontmatter"]["stage"] == "DELIVER"
    assert {"target": "boiler", "path": "Home/Heating/notes/boiler.md"} in note["links"]
    assert all(link["target"] != "2026" for link in note["links"])
    section = tools.open_note("Home/Heating/notes/boiler.md", heading="Wiring")
    assert "hunter2" not in section["text"] and "[REDACTED]" in section["text"]
    assert "weather curve" not in section["text"]
    assert section["backlinks"] == ["Home/Heating/SUMMARY.md"]


def test_open_note_refuses_private_and_unknown_notes(tools):
    with pytest.raises(PrivateNotAllowed):
        tools.open_note("Home/Budget/2026.md")
    with pytest.raises(FileNotFoundError):
        tools.open_note("../outside.md")


def test_allow_private_opens_private_notes(tools):
    tools.cfg = dataclasses.replace(tools.cfg, allow_private=True)
    assert "budget" in tools.open_note("Home/Budget/2026.md")["text"].lower()


def test_find_by_jira(tools):
    assert [r["path"] for r in tools.find_by_jira("ops-77")["results"]] == ["Work/Devops/db-resize/research-cpu.md"]


def test_each_call_scans_first(tools):
    (tools.cfg.root / "Home/Heating/notes/valves.md").write_text("# Valves\nThe boiler valves arrived.\n")
    paths = [r["path"] for r in tools.search("valves arrived")["results"]]
    assert paths[0] == "Home/Heating/notes/valves.md"
    assert tools.last_scan["changed"] == 1


def test_server_lists_the_five_tools(tools):
    server = build(tools=tools)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"search", "find_projects", "list_project", "open_note", "find_by_jira"}


def test_server_errors_carry_their_reason(tools):
    from mcp.server.mcpserver.exceptions import ToolError

    server = build(tools=tools)
    with pytest.raises(ToolError, match="private note"):
        asyncio.run(server.call_tool("open_note", {"path": "Home/Budget/2026.md"}))
