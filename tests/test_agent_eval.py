import json
from pathlib import Path

import pytest
from test_index import FakeEmbedder

from ragnarok import agent_eval, embed
from ragnarok.index import connect, update


def test_parse_notes_takes_the_last_list_and_makes_paths_relative(tmp_path):
    text = (
        'First idea {"notes": ["Home/old.md"]}\nFinal:\n'
        f'{{"notes": ["{tmp_path}/Home/Heating/SUMMARY.md", "./Studio/a.md", "Studio/a.md"]}}'
    )
    assert agent_eval.parse_notes(text, tmp_path) == ["Home/Heating/SUMMARY.md", "Studio/a.md"]
    assert agent_eval.parse_notes("no list", tmp_path) == []


def test_commands_give_each_agent_only_its_tools(tmp_path):
    grep = agent_eval.command("grep", "q", 8, "sonnet", None, ["Work"])
    mcp = agent_eval.command("mcp-kb", "q", 8, "sonnet", tmp_path / "mcp.json", ["Work"])
    for cmd in (grep, mcp):
        assert {"--restricted", "--strict-mcp-config"} <= set(cmd)
        assert cmd[cmd.index("--max-turns") + 1] == "10"
    assert grep[grep.index("--tools") + 1] == "Grep,Glob,Read"
    assert "--mcp-config" not in grep
    assert mcp[mcp.index("--tools") + 1] == "Read"
    assert "mcp__ragnarok__find_projects" in mcp[mcp.index("--allowedTools") + 1]


def test_the_mirror_has_no_private_note_and_no_excluded_note(vault):
    vault.eval_exclude = ["Home/Projects/ragnarok/"]
    (vault.root / "Home/Projects/ragnarok").mkdir(parents=True)
    (vault.root / "Home/Projects/ragnarok/SUMMARY.md").write_text("# expected answers\n")
    notes = agent_eval.mirror_notes(vault)
    assert "Home/Heating/notes/boiler.md" in notes
    assert not [n for n in notes if n.startswith(("Home/Password", "Home/Budget", "Work/Personal"))]
    assert not [n for n in notes if "ragnarok" in n]


def test_search_latency_counts_only_good_calls_without_changes(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "call=search ms=120 scan_ms=60 scan_changed=0 result=ok\n"
        "call=search ms=1400 scan_ms=1300 scan_changed=4 result=ok\n"
        "call=search ms=80 scan_ms=80 scan_changed=0 result=PrivateNotAllowed\n"
        "call=find_projects ms=600 scan_ms=70 scan_changed=0 result=ok\n"
    )
    assert agent_eval.search_latency(log) == [120]


def stream(notes, trace_id="tr-1", turns=3, cost=0.01):
    """A synthetic stream-json output: one search call with its result, then the final result event."""
    final = json.dumps({"notes": notes, "entities": ["e1"], "route": "kb", "answer": "ok"})
    search = {"type": "tool_use", "id": "t1", "name": "mcp__ragnarok__search", "input": {"query": "boiler"}}
    read = {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "a.md"}}
    found = {"type": "tool_result", "tool_use_id": "t1"}
    found["content"] = [{"type": "text", "text": json.dumps({"trace_id": trace_id, "results": []})}]
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I search."}]}},
        {"type": "assistant", "message": {"content": [search]}},
        {"type": "user", "message": {"content": [found]}},
        {"type": "assistant", "message": {"content": [read]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": "# n"}]}},
        {
            "type": "result",
            "subtype": "success",
            "num_turns": turns,
            "total_cost_usd": cost,
            "result": f"Done.\n{final}",
        },
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_stream_json_gives_the_tool_sequence_and_trace_ids():
    run = agent_eval.parse_stream(stream(["Home/Heating/notes/boiler.md"], trace_id="tr-9", turns=3, cost=0.02))
    assert [t["name"] for t in run["tools"]] == ["mcp__ragnarok__search", "Read"]
    assert run["tools"][0]["input"] == {"query": "boiler"}
    assert run["tools"][0]["trace_ids"] == ["tr-9"]
    assert run["tools"][1]["trace_ids"] == []
    assert run["trace_ids"] == ["tr-9"]
    assert (run["turns"], run["cost"], run["error"]) == (3, 0.02, False)
    answer = agent_eval.parse_answer(run["answer_text"], Path("/m"))
    assert answer == {
        "notes": ["Home/Heating/notes/boiler.md"],
        "entities": ["e1"],
        "route": "kb",
        "answer": "ok",
        "answered": True,
    }
    assert run["tools"][1]["result"] == "# n"  # the text of each tool result, for the attribution
    assert agent_eval.parse_answer("Done, no object.")["answered"] is False
    assert agent_eval.parse_stream("not json\n")["error"] is True


def fake_run(
    vault,
    tmp_path,
    monkeypatch,
    questions,
    agents,
    runs=1,
    baseline=None,
    seen=None,
    out=None,
    answers=None,
    failure=None,
):
    monkeypatch.setattr(agent_eval, "prepare", lambda cfg, **kw: cfg)
    seen = [] if seen is None else seen

    def runner(cmd, cwd, env):
        seen.append(cmd)
        if failure:
            return failure
        notes = answers[len(seen) - 1] if answers else ["Home/Heating/notes/boiler.md"]
        return stream(notes, trace_id=f"tr-{len(seen)}"), "", 0

    sets = {"s": questions}
    results = agent_eval.run_all(
        vault, sets, out or tmp_path / "out", 8, "sonnet", agents, runs, workers=1, runner=runner, baseline=baseline
    )
    return sets, results, seen


def test_each_question_gets_its_own_gold_environment(vault, tmp_path, monkeypatch):
    questions = [
        {"id": "a", "class": "single-truth", "question": "boiler?", "gold": [{"path": "Home/Heating/notes/boiler.md"}]},
        {"id": "b", "class": "single-truth", "question": "bike?", "gold": [{"path": "Home/Bike/shortlist.md"}]},
    ]
    _, _, seen = fake_run(vault, tmp_path, monkeypatch, questions, ["mcp-kb"])
    envs = []
    for cmd in seen:
        server = json.loads(Path(arg(cmd, "--mcp-config")).read_text())["mcpServers"]["ragnarok"]
        envs.append(json.loads(server["env"]["RAGNAROK_TRACE_GOLD"]))
        assert server["env"]["RAGNAROK_HOME"] == str(agent_eval.eval_home(vault))
    assert envs == [["Home/Heating/notes/boiler.md"], ["Home/Bike/shortlist.md"]]
    assert '"route"' in arg(seen[0], "-p")  # the new sets ask for the four-key JSON object


def test_two_runs_are_kept_and_paired(vault, tmp_path, monkeypatch):
    questions = [{"id": "q1", "type": "fact", "question": "boiler?", "expected": ["Home/Heating/notes/boiler.md"]}]
    out = tmp_path / "out"
    base = (tmp_path / "base-checkout", tmp_path / "base-home")
    agents = ["mcp-kb", "mcp-base"]
    other = ["Home/Bike/shortlist.md"]  # the second run of mcp-kb misses the gold note
    answers = [["Home/Heating/notes/boiler.md"], other, ["Home/Heating/notes/boiler.md"]] * 2
    sets, results, seen = fake_run(vault, tmp_path, monkeypatch, questions, agents, 2, base, out=out, answers=answers)
    assert len(seen) == 4
    names = sorted(p.name for p in (out / "s").glob("*.jsonl"))
    assert names == ["q1.mcp-base.1.jsonl", "q1.mcp-base.2.jsonl", "q1.mcp-kb.1.jsonl", "q1.mcp-kb.2.jsonl"]
    assert (out / "s" / "q1.mcp-kb.2.jsonl").read_text().startswith('{"type": "system"')
    runs = results["s"]["q1"]["mcp-kb"]
    assert [r["notes"] for r in runs] == [["Home/Heating/notes/boiler.md"], other]
    assert all(r["trace_ids"] and r["tools"] for r in runs)
    lines = agent_eval.report(vault, sets, results, [], agents)
    # the unit is the question: one row, the mean of its two runs (recall 1 and 0 for mcp-kb, 1 and 1 for mcp-base)
    row = next(line for line in lines if line.startswith("| Retrieval, draft set"))
    assert row.startswith("| Retrieval, draft set (n=1)") and "mcp-base 1.00, mcp-kb 0.50" in row
    fake_run(vault, tmp_path, monkeypatch, questions, agents, 2, base, seen=seen, out=out)
    assert len(seen) == 4  # the result files are used again


def test_baseline_agent_uses_the_baseline_checkout(vault, tmp_path, monkeypatch):
    questions = [{"id": "q1", "type": "fact", "question": "boiler?", "expected": ["Home/Heating/notes/boiler.md"]}]
    base = (tmp_path / "base-checkout", tmp_path / "base-home")
    _, _, seen = fake_run(vault, tmp_path, monkeypatch, questions, ["mcp-kb", "mcp-base"], 1, base)
    by_agent = {}
    for cmd in seen:
        path = Path(arg(cmd, "--mcp-config"))
        server = json.loads(path.read_text())["mcpServers"]["ragnarok"]
        by_agent[path.name.split(".")[-2]] = (server, arg(cmd, "--allowedTools"))
    server, tools = by_agent["mcp-base"]
    assert server["args"][:2] == ["--directory", str(base[0])]
    assert server["env"]["RAGNAROK_HOME"] == str(base[1])
    assert "find_entity" not in tools and "mcp__ragnarok__search" in tools
    kb_server, kb_tools = by_agent["mcp-kb"]
    assert kb_server["args"][1] != str(base[0])
    assert "mcp__ragnarok__find_entity" in kb_tools and "mcp__ragnarok__related" in kb_tools
    with pytest.raises(ValueError, match="baseline"):
        fake_run(vault, tmp_path, monkeypatch, questions, ["mcp-base"], 1, None, out=tmp_path / "other")


def test_prepare_adds_generated_distractors(vault, monkeypatch):
    vault.dimension = 8
    monkeypatch.setattr(embed, "OllamaEmbedder", lambda cfg: FakeEmbedder())
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    conn.close()
    distractor = agent_eval.eval_home(vault) / "distractors" / "Home" / "Heating" / "old-boiler.md"
    distractor.parent.mkdir(parents=True)
    distractor.write_text("# Old boiler\n")
    ecfg = agent_eval.prepare(vault)
    assert (ecfg.root / "Home/Heating/old-boiler.md").read_text() == "# Old boiler\n"
    assert (ecfg.root / "Home/Heating/notes/boiler.md").exists()
    (ecfg.root / "Home/Heating/old-boiler.md").unlink()
    agent_eval.prepare(vault, reuse_mirror=True)  # an existing mirror gets nothing
    assert not (ecfg.root / "Home/Heating/old-boiler.md").exists()


def test_a_failed_run_keeps_its_stderr_and_exit_code(vault, tmp_path, monkeypatch):
    questions = [{"id": "q1", "type": "fact", "question": "boiler?", "expected": []}]
    out = tmp_path / "out"
    fake_run(vault, tmp_path, monkeypatch, questions, ["mcp-kb"], 1, out=out, failure=("", "login expired\n", 3))
    assert (out / "s" / "q1.mcp-kb.1.stderr").read_text() == "3\nlogin expired\n"


def test_each_mcp_config_is_written_once_before_the_runs(vault, tmp_path, monkeypatch):
    questions = [{"id": "q1", "type": "fact", "question": "boiler?", "expected": []}]
    written, seen, real = [], [], agent_eval.mcp_config

    def spy(cfg, agent, name, qid, gold, baseline):
        written.append((agent, qid, len(seen)))
        return real(cfg, agent, name, qid, gold, baseline)

    monkeypatch.setattr(agent_eval, "mcp_config", spy)
    fake_run(vault, tmp_path, monkeypatch, questions, ["mcp-kb"], 2, seen=seen)
    assert written == [("mcp-kb", "q1", 0)]  # once, and before the first run started


def test_the_new_prompt_asks_for_the_answer_entities_for_both_agents(tmp_path):
    for agent in ("mcp-kb", "mcp-base"):
        prompt = agent_eval.command(agent, "q", 8, "sonnet", tmp_path / "mcp.json", ["Work"], new_prompt=True)[2]
        assert "the entities that are the answer: for a list question each item of the list" in prompt
        assert "an empty list when the route is none; an ID or a name" in prompt
