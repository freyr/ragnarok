import json
import re
from pathlib import Path

import pytest
from test_search import TopicEmbedder

from ragnarok import replay_run as rr
from ragnarok.cli import main
from ragnarok.config import Config, dump_config, load_config
from ragnarok.index import connect, update
from ragnarok.metadata import parse_note

# --- the scorer --------------------------------------------------------------------------------------------------

POSITIVE = {
    "id": "p001",
    "kind": "positive",
    "positive_kind": "content_change",
    "old": [{"path": "V/old-1.md", "expected": "supersede"}, {"path": "V/old-2.md", "expected": "supersede"}],
}
MERGE = {**POSITIVE, "id": "p002", "old": [{"path": "V/state.md", "expected": "merge"}]}
DUPLICATE = {
    "id": "d001",
    "kind": "duplicate",
    "positive_kind": None,
    "old": [{"path": "V/dup.md", "expected": "same"}],
}
NEGATIVE = {"id": "n001", "kind": "negative", "positive_kind": None, "old": []}
EMPTY = {"created": [], "changed": [], "deleted": []}


def rec(action, targets=(), written=False, outcome=None, **extra):
    outcome = outcome or ("written" if written else "nothing-written")
    return {
        "record": "kb-write/1",
        "action": action,
        "targets": list(targets),
        "written": written,
        "outcome": outcome,
        "run_id": None,
        **extra,
    }


def diff(created=(), changed=(), deleted=()):
    return {"created": list(created), "changed": list(changed), "deleted": list(deleted)}


def test_expected_action_of_each_kind():
    assert rr.expected_of(POSITIVE) == ("supersede", ["V/old-1.md", "V/old-2.md"])
    assert rr.expected_of(MERGE) == ("merge", ["V/state.md"])
    assert rr.expected_of(DUPLICATE) == ("same", ["V/dup.md"])
    assert rr.expected_of(NEGATIVE) == ("new", [])


def test_a_supersede_with_all_old_notes_and_the_change_on_disk_is_correct():
    record = rec("supersede", ["V/old-2.md", "V/old-1.md", "V/other.md"], written=True)
    s = rr.score(POSITIVE, record, diff(["V/new.md"], ["V/old-1.md", "V/old-2.md", "V/other.md"]))
    assert s["action_ok"] and s["target_ok"] and s["written_ok"] and s["correct"]
    assert s["category"] == "correct"


def test_a_supersede_that_misses_an_old_note_is_wrong():
    s = rr.score(POSITIVE, rec("supersede", ["V/old-1.md"], written=True), diff(["V/new.md"], ["V/old-1.md"]))
    assert s["action_ok"] and not s["target_ok"] and not s["correct"]
    assert s["category"] == "wrong"


def test_a_merge_counts_when_the_expected_action_is_merge():
    s = rr.score(MERGE, rec("merge", ["V/state.md"], written=True), diff(changed=["V/state.md"]))
    assert s["correct"] and s["category"] == "correct"


def test_a_merge_for_a_supersede_is_its_own_category():
    record = rec("merge", ["V/old-1.md", "V/old-2.md"], written=True)
    s = rr.score(POSITIVE, record, diff(changed=["V/old-1.md", "V/old-2.md"]))
    assert not s["action_ok"] and not s["correct"] and s["merge_for_supersede"]
    assert s["category"] == "merge-for-supersede"


def test_same_on_a_duplicate_with_nothing_written_is_correct():
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), EMPTY)
    assert s["correct"] and s["category"] == "correct"


def test_new_on_a_negative_needs_no_targets_and_a_created_file():
    assert rr.score(NEGATIVE, rec("new", [], written=True), diff(["V/fresh.md"]))["correct"]
    assert not rr.score(NEGATIVE, rec("new", ["V/x.md"], written=True), diff(["V/fresh.md"]))["target_ok"]


def test_ask_is_neither_correct_nor_wrong():
    s = rr.score(POSITIVE, rec("ask", ["V/old-1.md"]), EMPTY)
    assert not s["correct"] and s["category"] == "ask"


def test_a_missing_record_is_failed():
    s = rr.score(POSITIVE, None, EMPTY)
    assert s["action"] is None and not s["correct"] and s["category"] == "failed"


def test_a_refused_run_is_refused():
    s = rr.score(NEGATIVE, rec("refused", outcome="refused"), EMPTY)
    assert s["category"] == "refused" and not s["correct"]


def test_the_written_flag_must_match_the_snapshot():
    # the record says written, the snapshot shows no change
    s = rr.score(POSITIVE, rec("supersede", ["V/old-1.md", "V/old-2.md"], written=True), EMPTY)
    assert s["action_ok"] and s["target_ok"] and not s["written_ok"] and not s["correct"]
    # same, but a file changed
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), diff(changed=["V/dup.md"]))
    assert not s["written_ok"] and s["category"] == "wrong"
    # supersede written, but an old note is unchanged on disk
    s = rr.score(
        POSITIVE, rec("supersede", ["V/old-1.md", "V/old-2.md"], written=True), diff(["V/n.md"], ["V/old-1.md"])
    )
    assert not s["written_ok"]
    # a deleted file is never consistent
    s = rr.score(NEGATIVE, rec("new", [], written=True), diff(["V/fresh.md"], deleted=["V/gone.md"]))
    assert not s["written_ok"]


def test_the_summary_counts_the_categories_with_the_wilson_interval():
    rows = [
        {
            "kind": "positive",
            "positive_kind": "content_change",
            "cost_usd": 1.5,
            "score": rr.score(
                POSITIVE,
                rec("supersede", ["V/old-1.md", "V/old-2.md"], written=True),
                diff(["V/n.md"], ["V/old-1.md", "V/old-2.md"]),
            ),
        },
        {
            "kind": "positive",
            "positive_kind": "moved_copy",
            "cost_usd": 0.5,
            "score": rr.score(POSITIVE, rec("ask"), EMPTY),
        },
        {"kind": "duplicate", "positive_kind": None, "cost_usd": None, "score": rr.score(DUPLICATE, None, EMPTY)},
    ]
    s = rr.summarize(rows)
    assert s["all"]["correct"] == 1 and s["all"]["cases"] == 3
    assert s["all"]["low"] < 1 / 3 < s["all"]["high"]
    assert s["categories"] == {"correct": 1, "ask": 1, "failed": 1}
    assert s["by_kind"]["positive"]["correct"] == 1 and s["by_kind"]["positive"]["cases"] == 2
    assert s["by_positive_kind"]["moved_copy"]["correct"] == 0
    assert s["without_ask"]["cases"] == 2
    assert s["cost_usd"] == 2.0
    assert "correct" in rr.format_summary(s)


# --- the record parser -------------------------------------------------------------------------------------------


def test_the_record_is_the_last_record_line():
    first = json.dumps(rec("ask"))
    last = json.dumps(rec("same", ["V/dup.md"]))
    text = f"Some report.\nKB-WRITE-RECORD {first}\nmore text\nKB-WRITE-RECORD {last}\n\n"
    record, error = rr.parse_record(text)
    assert error is None and record["action"] == "same"


def test_a_malformed_or_absent_record_is_an_error():
    record, error = rr.parse_record("text\nKB-WRITE-RECORD {not json\n")
    assert record is None and "JSON" in error
    record, error = rr.parse_record("KB-WRITE-RECORD [1, 2]")
    assert record is None and "object" in error
    record, error = rr.parse_record("no record here")
    assert record is None and "KB-WRITE-RECORD" in error
    assert rr.parse_record(None)[0] is None


def test_the_agent_output_gives_the_result_cost_and_turns():
    out = json.dumps(
        {
            "result": "ok",
            "total_cost_usd": 0.7,
            "num_turns": 12,
            "is_error": False,
            "permission_denials": [{"tool_name": "Write"}],
        }
    )
    parsed = rr.parse_agent_output(out)
    assert parsed == {
        "result": "ok",
        "cost_usd": 0.7,
        "turns": 12,
        "is_error": False,
        "permission_denials": 1,
        "error": None,
    }
    assert rr.parse_agent_output("crash")["error"]


# --- snapshots ---------------------------------------------------------------------------------------------------


def test_the_snapshot_diff_gives_created_changed_and_deleted_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "keep.md").write_text("k")
    (tmp_path / "a" / "edit.md").write_text("1")
    (tmp_path / "gone.md").write_text("g")
    before = rr.snapshot(tmp_path)
    (tmp_path / "a" / "edit.md").write_text("2")
    (tmp_path / "gone.md").unlink()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "new.md").write_text("n")
    after = rr.snapshot(tmp_path)
    assert rr.diff_snapshots(before, after) == {
        "created": ["a/b/new.md"],
        "changed": ["a/edit.md"],
        "deleted": ["gone.md"],
    }
    assert rr.diff_snapshots(after, after) == EMPTY


def test_strip_keys_removes_only_the_named_keys():
    text = (
        "---\ntitle: T\nsuperseded_by: '[[x]]'\nsupersedes:\n  - '[[a]]'\n  - '[[b]]'\nvalidity: superseded\n"
        "type: guide\n---\n# Body\nvalidity: stays\n"
    )
    out = rr.strip_keys(text, ["supersedes", "superseded_by", "validity"])
    assert out == "---\ntitle: T\ntype: guide\n---\n# Body\nvalidity: stays\n"
    assert rr.strip_keys(text, []) == text
    assert rr.strip_keys("# no frontmatter\n", ["validity"]) == "# no frontmatter\n"


def test_as_of_falls_back_to_updated_created_and_date():
    import datetime

    assert rr.note_as_of({"as_of": "2026-01-02", "updated": "2025-01-01"}) == "2026-01-02"
    assert rr.note_as_of({"updated": datetime.date(2026, 3, 4), "created": "2020-01-01"}) == "2026-03-04"
    assert rr.note_as_of({"created": "2026-05-06"}) == "2026-05-06"
    assert rr.note_as_of({"date": datetime.datetime(2026, 7, 8, 9, 0)}) == "2026-07-08"
    assert rr.note_as_of({}) is None


# --- the case home -----------------------------------------------------------------------------------------------

OLD = "V/Docs/old.md"
NEWER = "V/Docs/newer.md"
OTHER = "V/Misc/other.md"
SRC_NOTES = {
    OLD: (
        "---\ntype: guide\nclass: state\nvalidity: superseded\nsuperseded_by: '[[newer]]'\n---\n"
        "# Old\nThe boiler runs at night and the pump follows the curve.\n"
    ),
    NEWER: (
        "---\ntype: guide\nclass: state\nvalidity: current\nas_of: 2026-08-16\nsupersedes: ['[[old]]']\n---\n"
        "# Newer\nThe boiler runs in the morning and the pump follows the curve.\n"
    ),
    OTHER: "# Other\nThe eyepiece field stop.\n",
}
CASE = {
    "id": "p001",
    "kind": "positive",
    "positive_kind": "content_change",
    "newer": NEWER,
    "old": [{"path": OLD, "expected": "supersede", "similarity": 0.8}],
    "drafts": {"bare": "# Newer\nThe boiler runs in the morning.\n", "typed": "---\ntype: guide\n---\n# Newer\n"},
    "start_state": {"exclude": NEWER, "remove_keys": {OLD: ["superseded_by", "validity"]}},
}


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A source bench home with a mirror, an index made with a test embedder, and a set folder."""
    mirror = tmp_path / "mirror"
    for rel, text in SRC_NOTES.items():
        path = mirror / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    (home / rr.MARKER).touch()
    cfg = Config(
        root=mirror,
        index_path=home / "data" / "index.db",
        vaults=["V"],
        dimension=len(TopicEmbedder.TOPICS),
        min_words=3,
    )
    (home / "config.toml").write_text(dump_config(cfg, root=mirror, data_dir="data"))
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    cfg = load_config(home / "config.toml")
    conn = connect(cfg)
    update(cfg, conn, TopicEmbedder())
    conn.close()
    set_dir = home / "replay" / "set-v1"
    set_dir.mkdir(parents=True)
    (set_dir / "cases.jsonl").write_text(json.dumps(CASE) + "\n")
    (set_dir / "manifest.json").write_text("{}\n")
    return cfg, home, mirror, set_dir


def test_the_case_home_has_the_start_state_the_marker_and_its_own_root(bench, tmp_path):
    cfg, home, mirror, set_dir = bench
    before = rr.snapshot(mirror)
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "bare")
    h = case_home.home
    assert h == set_dir / "runs" / "r1" / "p001"
    assert (h / rr.MARKER).is_file()
    assert not (case_home.mirror / NEWER).exists()
    fm, body = parse_note((case_home.mirror / OLD).read_text())
    assert "superseded_by" not in fm and "validity" not in fm and fm["class"] == "state"
    assert (case_home.mirror / OTHER).read_text() == SRC_NOTES[OTHER]
    case_cfg = load_config(h / "config.toml")
    assert case_cfg.root == case_home.mirror
    assert case_cfg.index_path == h / "data" / "index.db"
    assert case_home.draft.read_text() == CASE["drafts"]["bare"]
    assert case_home.as_of == "2026-08-16"
    assert case_home.index["embedded"] == 0
    conn = connect(case_cfg)
    paths = {p for (p,) in conn.execute("SELECT path FROM notes")}
    conn.close()
    assert paths == {OLD, OTHER}
    assert rr.snapshot(mirror) == before
    config = json.loads(case_home.mcp_config.read_text())
    server = config["mcpServers"][rr.SERVER]
    assert server["env"] == {"RAGNAROK_HOME": str(h)}
    assert server["args"][-1] == "serve"


def test_the_typed_form_writes_the_typed_draft(bench):
    cfg, home, mirror, set_dir = bench
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "typed")
    assert case_home.draft.read_text() == CASE["drafts"]["typed"]


def test_the_agent_command_allows_writes_only_under_the_case_mirror(bench):
    cfg, home, mirror, set_dir = bench
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "bare")
    plugin = Path("/opt/plugin")
    argv = rr.agent_command("claude", case_home, plugin, budget=2.0, model="opus")
    allowed = argv[argv.index("--allowedTools") + 1 :]
    writes = [a for a in allowed if a.startswith(("Write", "Edit"))]
    assert writes == [f"Write(/{case_home.mirror}/**)", f"Edit(/{case_home.mirror}/**)"]
    assert "--strict-mcp-config" in argv and argv[argv.index("--max-budget-usd") + 1] == "2.0"
    assert argv[argv.index("--model") + 1] == "opus"
    assert "--plugin-dir" not in argv
    prompt = rr.agent_prompt(case_home, plugin / "skills" / "kb-write" / "SKILL.md", plugin)
    assert f"mirror={case_home.mirror}" in prompt and "2026-08-16" in prompt and str(case_home.draft) in prompt


# --- the run -----------------------------------------------------------------------------------------------------


def fake_runner(action, write=True):
    """A runner that acts like the skill: it writes into the mirror of the prompt and prints a record."""

    def run(argv, prompt, cwd, env, timeout):
        mirror = Path(re.search(r"mirror=(\S+)", prompt).group(1))
        targets, written = [], False
        if write:
            (mirror / "V" / "Docs" / "fresh.md").write_text("# Fresh\n")
            (mirror / OLD).write_text(
                (mirror / OLD).read_text().replace("class: state", "class: state\nvalidity: superseded")
            )
            targets, written = [OLD], True
        record = rec(action, targets, written=written)
        out = json.dumps(
            {
                "result": f"done\nKB-WRITE-RECORD {json.dumps(record)}",
                "total_cost_usd": 0.25,
                "num_turns": 9,
                "is_error": False,
                "permission_denials": [],
            }
        )
        return rr.AgentOutput(exit_code=0, stdout=out, stderr="", duration_s=1.0)

    return run


def test_a_run_records_each_case_and_leaves_the_source_unchanged(bench, tmp_path):
    cfg, home, mirror, set_dir = bench
    before = rr.snapshot(mirror)
    rows, summary, run_dir = rr.run_cases(
        cfg, home, set_dir, [CASE], skill=tmp_path / "SKILL.md", runner=fake_runner("supersede"), parallel=2
    )
    assert rr.snapshot(mirror) == before
    assert run_dir.parent == set_dir / "runs"
    (row,) = rows
    assert row["case"] == "p001" and row["form"] == "bare" and row["as_of"] == "2026-08-16"
    assert row["diff"] == {"created": ["V/Docs/fresh.md"], "changed": [OLD], "deleted": []}
    assert row["record"]["action"] == "supersede" and row["record_error"] is None
    assert row["cost_usd"] == 0.25 and row["turns"] == 9 and row["exit_code"] == 0
    assert row["score"]["correct"]
    assert summary["all"]["correct"] == 1 and summary["source_unchanged"]
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["case"] == "p001"
    assert json.loads((run_dir / "summary.json").read_text())["all"]["cases"] == 1


def test_a_runner_error_gives_a_failed_record(bench, tmp_path):
    cfg, home, mirror, set_dir = bench

    def crash(argv, prompt, cwd, env, timeout):
        return rr.AgentOutput(exit_code=None, stdout="", stderr="", duration_s=5.0, timed_out=True)

    rows, summary, _ = rr.run_cases(cfg, home, set_dir, [CASE], skill=tmp_path / "SKILL.md", runner=crash)
    assert rows[0]["timed_out"] and rows[0]["record"] is None and rows[0]["score"]["category"] == "failed"
    assert summary["categories"] == {"failed": 1}


def test_a_changed_source_mirror_fails_the_run(bench, tmp_path):
    cfg, home, mirror, set_dir = bench

    def vandal(argv, prompt, cwd, env, timeout):
        (mirror / OTHER).write_text("changed\n")
        return fake_runner("same", write=False)(argv, prompt, cwd, env, timeout)

    with pytest.raises(rr.SourceChanged, match="other.md"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=tmp_path / "SKILL.md", runner=vandal)


def test_the_run_refuses_a_home_with_no_bench_marker(bench, tmp_path, capsys):
    cfg, home, mirror, set_dir = bench
    (home / rr.MARKER).unlink()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    assert main(["replay", "run", "--set", str(set_dir), "--skill", str(skill)]) == 1
    assert rr.MARKER in capsys.readouterr().err


def test_the_run_refuses_a_set_outside_the_home(bench, tmp_path, capsys):
    cfg, home, mirror, set_dir = bench
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    outside = tmp_path / "set"
    outside.mkdir()
    (outside / "cases.jsonl").write_text(json.dumps(CASE) + "\n")
    (outside / "manifest.json").write_text("{}\n")
    assert main(["replay", "run", "--set", str(outside), "--skill", str(skill)]) == 1
    assert "inside" in capsys.readouterr().err


def test_the_cli_runs_the_chosen_cases(bench, tmp_path, monkeypatch, capsys):
    cfg, home, mirror, set_dir = bench
    skill = tmp_path / "SKILL.md"
    skill.write_text("# skill\n")
    monkeypatch.setattr(rr, "subprocess_runner", fake_runner("supersede"))
    assert main(["replay", "run", "--set", str(set_dir), "--skill", str(skill), "--cases", "p001"]) == 0
    assert "correct" in capsys.readouterr().out
    assert main(["replay", "run", "--set", str(set_dir), "--skill", str(skill), "--cases", "p999"]) == 1
    assert "p999" in capsys.readouterr().err
