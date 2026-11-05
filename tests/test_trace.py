import asyncio
import dataclasses
import datetime as dt
import json

import pytest
from test_search import EXTRA, TopicEmbedder

from ragnarok.config import TraceConfig
from ragnarok.index import connect, update
from ragnarok.server import build
from ragnarok.tools import Tools
from ragnarok.trace import GOLD_ENV, purge, trace_dir

PEER = "Home/Heating/notes/n{:02}.md"


@pytest.fixture
def tools(vault):
    """Sixty notes and more that match the query "boiler", so that the pools hold more than 50 documents."""
    for rel, text in EXTRA.items():
        (vault.root / rel).write_text(text)
    for i in range(60):
        (vault.root / PEER.format(i)).write_text(f"# Note {i}\nThe boiler note number {i} has a relay.\n")
    vault.dimension = len(TopicEmbedder.TOPICS)
    conn = connect(vault)
    emb = TopicEmbedder()
    update(vault, conn, emb)
    return Tools(vault, conn, emb)


def lines(cfg) -> list[dict]:
    return [json.loads(line) for f in sorted(trace_dir(cfg).glob("*.jsonl")) for line in f.read_text().splitlines()]


def line_of(cfg, trace_id: str) -> dict:
    (found,) = [line for line in lines(cfg) if line["trace_id"] == trace_id]
    return found


def test_each_tool_result_has_a_trace_id(tools):
    results = {
        "search": tools.search("boiler"),
        "find_projects": tools.find_projects("boiler"),
        "list_project": tools.list_project("Heating"),
        "open_projects": tools.open_projects(),
        "open_note": tools.open_note("Home/Heating/notes/boiler.md"),
        "find_by_jira": tools.find_by_jira("OPS-77"),
    }
    for name, result in results.items():
        assert line_of(tools.cfg, result["trace_id"])["gen_ai.tool.name"] == name
    assert len({r["trace_id"] for r in results.values()}) == 6


def test_server_search_result_has_a_trace_id(tools):
    server = build(tools=tools)
    result = asyncio.run(server.call_tool("search", {"query": "boiler", "limit": 3}))
    trace_id = json.loads(result.content[0].text)["trace_id"]
    assert line_of(tools.cfg, trace_id)["gen_ai.tool.call.arguments"]["limit"] == 3


def test_trace_line_has_the_candidate_ranks_of_the_top_50(tools):
    result = tools.search("boiler", limit=5)
    line = line_of(tools.cfg, result["trace_id"])
    candidates = line["ragnarok.candidates"]
    assert len(candidates) >= 50
    for c in candidates:
        assert min(r for r in (c["dense_rank"], c["fts_rank"]) if r) <= 50
        assert set(c) >= {"doc_id", "path", "unit_id", "dense_rank", "fts_rank", "filters", "rrf", "authority"}
        assert c["rrf"] == pytest.approx(sum(1 / (60 + r) for r in (c["dense_rank"], c["fts_rank"]) if r), abs=1e-9)
    by_rank = {c["final_rank"]: c for c in candidates}
    assert [by_rank[i]["path"] for i in range(1, 6)] == [r["path"] for r in result["results"]]
    assert [d["score"] for d in line["gen_ai.retrieval.documents"]] == [r["score"] for r in result["results"]]


def test_trace_line_has_the_plan_the_version_and_the_stage_times(tools):
    tools.ids = ("session-1", "conversation-1")
    line = line_of(tools.cfg, tools.search("boiler", project="Heating")["trace_id"])
    assert line["gen_ai.retrieval.query.text"] == "boiler"
    assert (line["mcp.session.id"], line["gen_ai.conversation.id"]) == ("session-1", "conversation-1")
    plan = line["ragnarok.plan"]
    assert plan["mode"] == "exact"
    assert 60 <= plan["eligible_units"] < plan["total_units"]
    assert plan["selectivity"]["project"]["units"] == plan["eligible_units"]
    assert 0 < plan["selectivity"]["project"]["selectivity"] < 1
    assert plan["pools"]["dense"] == min(200, plan["eligible_units"])  # units; the dense pool holds 200 units at most
    assert 50 < plan["pools"]["fts"] <= plan["eligible_units"]
    version = line["ragnarok.index.version"]
    assert version["counts"]["units"] == plan["total_units"]
    assert set(version) >= {"commit", "config_hash", "model", "min_words", "max_words", "corpus_hash", "counts"}
    assert {"scan", "embed", "plan", "dense", "fts", "rank"} <= set(line["ragnarok.ms"])
    assert line["error.type"] is None


def test_a_call_without_a_context_has_null_ids(tools):
    line = line_of(tools.cfg, tools.search("boiler")["trace_id"])
    assert line["mcp.session.id"] is None and line["gen_ai.conversation.id"] is None


def test_failed_call_records_the_error_type(tools):
    with pytest.raises(FileNotFoundError):
        tools.open_note("Home/Nothing.md")
    assert lines(tools.cfg)[-1]["error.type"] == "FileNotFoundError"


def outside_top_50(tools) -> str:
    """A path that the pools hold, with both ranks above 50."""
    hits = tools.search("boiler", limit=100)["results"]
    far = [h["path"] for h in hits if (h["dense_rank"] or 99) > 50 and (h["fts_rank"] or 99) > 50]
    assert far
    return far[0]


def test_trace_records_a_gold_document_outside_the_top_50(tools, monkeypatch):
    gold = outside_top_50(tools)
    plain = line_of(tools.cfg, tools.search("boiler")["trace_id"])
    assert gold not in [c["path"] for c in plain["ragnarok.candidates"]]  # a production call keeps the top 50
    monkeypatch.setenv(GOLD_ENV, json.dumps([gold]))
    line = line_of(tools.cfg, tools.search("boiler")["trace_id"])
    (record,) = [c for c in line["ragnarok.candidates"] if c["path"] == gold]
    assert record["gold"] is True
    assert all(record["filters"].values()) and record["filters"]
    assert (record["dense_rank"] or record["fts_rank"]) > 50
    assert record["final_rank"] > 50
    assert [c["path"] for c in line["ragnarok.candidates"] if c.get("gold")] == [gold]


def test_trace_records_the_filter_that_removes_a_gold_document(tools, monkeypatch):
    gold = "Home/Astro/Theory/Field Stop.md"
    monkeypatch.setenv(GOLD_ENV, json.dumps([gold]))
    line = line_of(tools.cfg, tools.search("boiler eyepiece", project="Heating")["trace_id"])
    (record,) = [c for c in line["ragnarok.candidates"] if c["path"] == gold]
    assert record["gold"] is True
    assert record["filters"] == {"project": False, "private": True}
    assert record["dense_rank"] is None and record["fts_rank"] is None and record["final_rank"] is None


def test_trace_records_the_private_filter_for_a_private_gold_document(tools, monkeypatch):
    monkeypatch.setenv(GOLD_ENV, json.dumps(["Home/Budget/2026.md", "Home/Gone.md"]))
    line = line_of(tools.cfg, tools.search("boiler")["trace_id"])
    records = {c["path"]: c for c in line["ragnarok.candidates"] if c.get("gold")}
    assert records["Home/Budget/2026.md"]["filters"] == {"private": False}
    assert records["Home/Gone.md"]["doc_id"] is None  # not in the index


def test_old_trace_files_are_deleted(tools):
    cfg = dataclasses.replace(tools.cfg, trace=TraceConfig(retention_days=10))
    folder = trace_dir(cfg)
    folder.mkdir(parents=True)
    today = dt.datetime.now(dt.UTC).date()
    names = {
        "old": (today - dt.timedelta(days=11)).isoformat() + ".jsonl",
        "edge": (today - dt.timedelta(days=10)).isoformat() + ".jsonl",
        "new": today.isoformat() + ".jsonl",
        "other": "notes.txt",
    }
    for name in names.values():
        (folder / name).write_text("{}\n")
    build(tools=dataclasses.replace(tools, cfg=cfg))
    assert sorted(p.name for p in folder.iterdir()) == sorted(names[k] for k in ("edge", "new", "other"))
    assert purge(cfg) == []


def test_a_read_only_trace_file_does_not_change_the_result(tools, caplog):
    folder = trace_dir(tools.cfg)
    tools.search("boiler")  # makes today's file
    (file,) = folder.glob("*.jsonl")
    file.chmod(0o444)
    folder.chmod(0o555)
    try:
        result = tools.search("boiler", limit=3)
        assert len(result["results"]) == 3
        with pytest.raises(FileNotFoundError):  # the real error, not the trace error
            tools.open_note("Home/Nothing.md")
    finally:
        folder.chmod(0o755)
        file.chmod(0o644)
    assert "trace" in caplog.text


def test_a_bad_gold_value_does_not_change_the_result(tools, monkeypatch):
    monkeypatch.setenv(GOLD_ENV, "not json")
    assert len(tools.search("boiler", limit=3)["results"]) == 3


def test_a_failing_purge_does_not_stop_the_server(tools, monkeypatch):
    from ragnarok import server as server_module

    def broken(cfg):
        raise PermissionError("no")

    monkeypatch.setattr(server_module, "purge", broken)
    build(tools=tools)


def test_gold_removed_by_a_filter_has_ranks_from_before_the_filter(tools, monkeypatch):
    gold = "Home/Astro/Theory/Field Stop.md"
    monkeypatch.setenv(GOLD_ENV, json.dumps([gold, "Home/Heating/notes/n03.md"]))
    line = line_of(tools.cfg, tools.search("boiler eyepiece", project="Heating")["trace_id"])
    records = {c["path"]: c for c in line["ragnarok.candidates"] if c.get("gold")}
    removed = records[gold]
    assert removed["filters"]["project"] is False
    assert removed["dense_rank"] is None and removed["fts_rank"] is None  # after the filter
    assert removed["prefilter_dense_rank"] >= 1
    assert removed["prefilter_fts_rank"] == 1  # the only note with the word "eyepiece"
    kept = records["Home/Heating/notes/n03.md"]
    assert kept["prefilter_fts_rank"] >= 1 and kept["fts_rank"] >= 1


def test_gold_without_a_match_has_null_prefilter_fts_rank(tools, monkeypatch):
    monkeypatch.setenv(GOLD_ENV, json.dumps(["Home/Astro/Theory/Field Stop.md"]))
    line = line_of(tools.cfg, tools.search("boiler", project="Heating")["trace_id"])
    (record,) = [c for c in line["ragnarok.candidates"] if c.get("gold")]
    assert record["prefilter_fts_rank"] is None and record["prefilter_dense_rank"] >= 1


def test_every_read_tool_records_the_returned_documents(tools):
    boiler = "Home/Heating/notes/boiler.md"
    calls = {
        "search": (tools.search("boiler", limit=3), lambda r: [x["path"] for x in r["results"]]),
        "find_projects": (
            tools.find_projects("boiler"),
            lambda r: [p for x in r["results"] for p in x["matching_notes"]],
        ),
        "list_project": (
            tools.list_project("Heating"),
            lambda r: [n["path"] for x in r["results"] for n in x["notes"]],
        ),
        "open_projects": (
            tools.open_projects(),
            lambda r: [f"{p['folder']}/SUMMARY.md" for group in r["results"].values() for p in group],
        ),
        "open_note": (tools.open_note(boiler), lambda r: [boiler]),
        "find_by_jira": (tools.find_by_jira("OPS-77"), lambda r: [x["path"] for x in r["results"]]),
    }
    for name, (result, returned) in calls.items():
        line = line_of(tools.cfg, result["trace_id"])
        docs = line["gen_ai.retrieval.documents"]
        assert sorted(d["path"] for d in docs) == sorted(set(returned(result))), name
        assert docs and all(d["id"] for d in docs), name
        assert [s["id"] for s in line["ragnarok.returned.status"]] == [d["id"] for d in docs], name


@pytest.mark.parametrize("method", ["dense", "fts"])
def test_prefilter_rank_tells_inside_the_pool_from_outside(tools, monkeypatch, method):
    from ragnarok import search as search_module

    monkeypatch.setattr(search_module, "CANDIDATES", 10)
    hits = tools.search("boiler", limit=100)["results"]  # one unit for each note, so unit rank = note rank
    ranked = {h["path"] for h in hits if h[f"{method}_rank"] is not None}
    inside = sorted(ranked)[0]
    outside = next(PEER.format(i) for i in range(60) if PEER.format(i) not in ranked)
    monkeypatch.setenv(GOLD_ENV, json.dumps([inside, outside]))
    result = tools.search("boiler", project="Astro")
    line = line_of(tools.cfg, result["trace_id"])
    assert line["ragnarok.plan"]["pool_limit"] == 10
    records = {c["path"]: c for c in line["ragnarok.candidates"] if c.get("gold")}
    for path in (inside, outside):
        assert records[path]["filters"]["project"] is False
        assert records[path][f"{method}_rank"] is None
    assert 1 <= records[inside][f"prefilter_{method}_rank"] <= 10
    assert records[outside][f"prefilter_{method}_rank"] > 10
