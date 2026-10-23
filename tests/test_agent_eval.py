import json

from ragnarok import agent_eval


def test_parse_notes_takes_the_last_list_and_makes_paths_relative(tmp_path):
    text = (
        'First idea {"notes": ["Home/old.md"]}\nFinal:\n'
        f'{{"notes": ["{tmp_path}/Home/Heating/SUMMARY.md", "./Studio/a.md", "Studio/a.md"]}}'
    )
    assert agent_eval.parse_notes(text, tmp_path) == ["Home/Heating/SUMMARY.md", "Studio/a.md"]
    assert agent_eval.parse_notes("no list", tmp_path) == []


def test_commands_give_each_agent_only_its_tools(tmp_path):
    grep = agent_eval.command("grep", "q", 8, "sonnet", None, ["Work"])
    mcp = agent_eval.command("mcp", "q", 8, "sonnet", tmp_path / "mcp.json", ["Work"])
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


def test_run_all_scores_with_a_fake_runner_and_reuses_results(vault, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_eval, "prepare", lambda cfg: cfg)
    monkeypatch.setattr(agent_eval, "mcp_config", lambda cfg: tmp_path / "mcp.json")
    answers = {"grep": ["Home/Bike/shortlist.md"], "mcp": ["Home/Heating/notes/boiler.md"]}
    calls = []

    def runner(cmd, cwd, env):
        agent = "grep" if "Grep,Glob,Read" in cmd else "mcp"
        calls.append(agent)
        return {"result": json.dumps({"notes": answers[agent]}), "total_cost_usd": 0.01, "num_turns": 3}

    sets = {
        "draft": [{"id": "q1", "type": "fact", "question": "boiler?", "expected": ["Home/Heating/notes/boiler.md"]}],
        "hard": [
            {"id": "h1", "type": "continuation", "subtype": "x", "question": "heat", "expected": ["Home/Heating"]}
        ],
    }
    out = tmp_path / "out"
    results = agent_eval.run_all(vault, sets, out, 8, "sonnet", workers=2, runner=runner)
    assert len(calls) == 4
    assert results["draft"]["q1"]["mcp"]["notes"] == ["Home/Heating/notes/boiler.md"]
    lines = agent_eval.report(vault, sets, results, [100, 200])
    assert any(line.startswith("| Retrieval, draft set (n=1)") and "+1.00" in line for line in lines)
    assert any(line.startswith("| Project continuation (n=1)") and "MCP 1.00 (grep 0.00)" in line for line in lines)
    agent_eval.run_all(vault, sets, out, 8, "sonnet", workers=2, runner=runner)
    assert len(calls) == 4
