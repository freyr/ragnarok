import asyncio
import dataclasses
import os
import time

import pytest
from test_search import EXTRA, TopicEmbedder

from ragnarok import server as server_module
from ragnarok.config import dump_config
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


def test_new_private_pattern_hides_an_indexed_note(tools):
    assert "Home/Astro/Theory/Field Stop.md" in [r["path"] for r in tools.search("eyepiece")["results"]]
    tools.cfg = dataclasses.replace(tools.cfg, private_patterns=[*tools.cfg.private_patterns, "Home/Astro/**"])
    assert "Home/Astro/Theory/Field Stop.md" not in [r["path"] for r in tools.search("eyepiece")["results"]]
    with pytest.raises(PrivateNotAllowed):
        tools.open_note("Home/Astro/Theory/Field Stop.md")


FIELD_STOP = {"path": "Home/Astro/Theory/Field Stop.md"}


def write_config(path, text, age=5):
    """Write the config file with a modification time `age` seconds in the past (a settled file)."""
    path.write_text(text)
    t = time.time_ns() - age * 10**9
    os.utime(path, ns=(t, t))


@pytest.fixture
def served(tools, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "OllamaEmbedder", lambda cfg: TopicEmbedder())
    cfg = tools.cfg
    config_file = tmp_path / "ragnarok-home" / "config.toml"
    config_file.parent.mkdir()
    write_config(config_file, dump_config(cfg, root=cfg.root, index_path=cfg.index_path))
    server = build(config_file=config_file)
    asyncio.run(server.call_tool("open_note", FIELD_STOP))
    return server, config_file, cfg


def private_text(cfg):
    private = dataclasses.replace(cfg, private_patterns=[*cfg.private_patterns, "Home/Astro/**"])
    return dump_config(private, root=cfg.root, index_path=cfg.index_path)


def test_server_reads_a_changed_config_file(served):
    from mcp.server.mcpserver.exceptions import ToolError

    server, config_file, cfg = served
    write_config(config_file, private_text(cfg))
    with pytest.raises(ToolError, match="private note"):
        asyncio.run(server.call_tool("open_note", FIELD_STOP))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "vaults = [",
        'index_path = "index.db"\ndimension = 8\n',
    ],
    ids=["empty", "invalid-toml", "no-vaults"],
)
def test_server_keeps_its_config_when_the_new_file_is_not_usable(served, text):
    server, config_file, cfg = served
    count = lambda: connect(cfg).execute("SELECT count(*), sum(sensitivity = 'private') FROM notes").fetchone()  # noqa: E731
    before = count()
    write_config(config_file, text)
    asyncio.run(server.call_tool("open_note", FIELD_STOP))
    assert count() == before


def test_server_waits_until_a_new_config_file_is_settled(served):
    from mcp.server.mcpserver.exceptions import ToolError

    server, config_file, cfg = served
    write_config(config_file, private_text(cfg), age=0)
    asyncio.run(server.call_tool("open_note", FIELD_STOP))  # the file can still be in the middle of a write
    t = time.time_ns() - 5 * 10**9
    os.utime(config_file, ns=(t, t))
    with pytest.raises(ToolError, match="private note"):
        asyncio.run(server.call_tool("open_note", FIELD_STOP))


def test_server_starts_with_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGNAROK_HOME", str(tmp_path))
    monkeypatch.setattr(server_module, "OllamaEmbedder", lambda cfg: TopicEmbedder())
    server = build(config_file=tmp_path / "missing.toml")
    asyncio.run(server.call_tool("find_by_jira", {"key": "ABC-1"}))
    assert (tmp_path / "data" / "index.db").exists()
