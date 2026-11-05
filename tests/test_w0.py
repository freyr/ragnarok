"""ragnarok w0 score: the step summary from a synthetic Claude Code transcript, one w0_step row, no evaluator."""

import hashlib
import io
import json
import sqlite3
import time

import pytest

from ragnarok import cli, w0
from ragnarok.config import load_config
from ragnarok.index import connect


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home folder with a config, a vault root with a private folder and a private path outside the root."""
    (tmp_path / "vaults" / "Work" / "Secret").mkdir(parents=True)
    (tmp_path / "config.toml").write_text(
        'root = "vaults"\nvaults = ["Work"]\ndata_dir = "data"\nprivate_patterns = ["Work/Secret/**"]\n'
        f'\n[w0]\nsummary_chars = 400\nprivate_paths = ["{tmp_path / "private"}", "Work/Diary"]\n'
    )
    monkeypatch.setenv("RAGNAROK_HOME", str(tmp_path))
    return tmp_path


def _user(text, origin=None, uuid=None, meta=False):
    line = {"type": "user", "message": {"role": "user", "content": text}}
    if origin:
        line["origin"] = {"kind": origin}
    if uuid:
        line["uuid"] = uuid
    if meta:
        line["isMeta"] = True
    return line


def _tool_use(name, **inputs):
    block = {"type": "tool_use", "id": f"t-{name}", "name": name, "input": inputs}
    return {"type": "assistant", "message": {"role": "assistant", "content": [block]}}


def _tool_result(name):
    block = {"type": "tool_result", "tool_use_id": f"t-{name}", "content": "ok"}
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _say(text):
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def _kb_save(args=""):
    """The real shape: the command line (origin human), then the isMeta line with the skill body."""
    text = "<command-message>kb-save</command-message>\n<command-name>/kb-save</command-name>"
    command = _user(text + (f"\n<command-args>{args}</command-args>" if args else ""), origin="human")
    return [command, _user("Base directory for this skill: /skills/kb-save\n# kb-save", meta=True)]


def _transcript(path, lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _score(home, monkeypatch, transcript, session="s-1", cwd=None):
    payload = {"session_id": session, "transcript_path": str(transcript), "cwd": str(cwd or home)}
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    return cli.main(["w0", "score"])


def _rows(home):
    conn = sqlite3.connect(home / "data" / "core.db")
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM w0_step ORDER BY id")]
    conn.close()
    return rows


EDIT_STEP = [
    _user("an old prompt"),
    _say("an old answer"),
    _user("fix the parser", origin="human"),
    _tool_use("Read", file_path="/work/src/parser.py"),
    _tool_result("Read"),
    _tool_use("Edit", file_path="/work/src/parser.py", old_string="a", new_string="b"),
    _tool_result("Edit"),
    _tool_use("Write", file_path="/work/tests/test_parser.py", content="x"),
    _tool_result("Write"),
    _say("The parser now accepts empty lines."),
]


def test_summary_has_kind_tools_files_and_final_text(home, monkeypatch, tmp_path):
    assert _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", EDIT_STEP)) == 0
    [row] = _rows(home)
    assert row["step_kind"] == "edit"
    summary = row["summary"]
    assert "Read" in summary and "Edit" in summary and "Write" in summary
    assert "/work/src/parser.py" in summary and "/work/tests/test_parser.py" in summary
    assert "The parser now accepts empty lines." in summary
    assert "an old answer" not in summary and "an old prompt" not in summary
    assert row["summary_hash"] == hashlib.sha256(summary.encode()).hexdigest()
    features = json.loads(row["features"])
    assert features["files"] == ["/work/src/parser.py", "/work/tests/test_parser.py"]
    assert features["tools"] == {"Read": 1, "Edit": 1, "Write": 1}
    assert row["label_command"] == 0 and row["mode"] == "shadow"
    assert all(row[c] is None for c in ("evaluator", "evaluator_version", "score", "verdict", "decided_by"))


def test_summary_is_redacted(home, monkeypatch, tmp_path):
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    lines = [_user("show it"), _say(f"The token is {secret}. " + "x" * 1000)]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines))
    [row] = _rows(home)
    assert secret not in row["summary"] and "[REDACTED]" in row["summary"]
    assert len(row["summary"]) == 400  # w0.summary_chars, after the redaction


@pytest.mark.parametrize(
    "where", ["cwd", "edit", "read", "private_path", "relative_private_path", "grep", "glob", "dotdot"]
)
def test_private_step_keeps_only_the_hash(home, monkeypatch, tmp_path, where):
    secret_file = str(home / "vaults" / "Work" / "Secret" / "plan.md")
    outside = str(home / "private" / "notes.md")
    cwd = {"cwd": home / "vaults" / "Work" / "Secret"}.get(where, home)
    tool = {
        "edit": _tool_use("Edit", file_path=secret_file),
        "read": _tool_use("Read", file_path=secret_file),
        "private_path": _tool_use("Write", file_path=outside),
        "relative_private_path": _tool_use("Edit", file_path=str(home / "vaults" / "Work" / "Diary" / "day.md")),
        "grep": _tool_use("Grep", pattern="plan", path=str(home / "vaults" / "Work" / "Secret"), output_mode="content"),
        "glob": _tool_use("Glob", pattern="*.md", path=str(home / "vaults" / "Work" / "Secret")),
        "dotdot": _tool_use("Edit", file_path=str(home / "vaults" / "Work" / ".." / "Work" / "Secret" / "plan.md")),
    }.get(where, _tool_use("Bash", command="ls"))
    lines = [_user("work"), tool, _tool_result("x"), _say("done with the secret plan")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines), cwd=cwd)
    lines += [*_kb_save("the secret plan matters"), _say("Saved.")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines), cwd=cwd if where == "cwd" else home)
    step, save = _rows(home)
    assert step["summary"] is None and len(step["summary_hash"]) == 64
    features = json.loads(step["features"] or "{}")
    assert features.get("files") is None and features.get("read_files") is None
    assert "secret" not in json.dumps(features)
    if where == "cwd":  # the /kb-save step is private too: its text keeps only its hash
        save_features = json.loads(save["features"])
        assert save["summary"] is None and save_features.get("kb_save_text") is None
        expected = hashlib.sha256(b"the secret plan matters").hexdigest()
        assert save_features["kb_save_hash"] == expected
        assert "secret" not in json.dumps(save_features)


def test_kb_save_labels_the_previous_step(home, monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    lines = list(EDIT_STEP)
    _score(home, monkeypatch, _transcript(path, lines))
    _score(home, monkeypatch, _transcript(tmp_path / "other.jsonl", EDIT_STEP), session="s-2")
    lines += [*_kb_save("keep the parser rule"), _say("Saved.")]
    _score(home, monkeypatch, _transcript(path, lines))
    first, other, save = _rows(home)
    assert first["label_command"] == 1
    assert other["label_command"] == 0  # another session
    assert save["step_kind"] == "kb-save" and save["label_command"] is None
    assert json.loads(save["features"])["kb_save_text"] == "keep the parser rule"


def test_w0_score_does_not_scan_or_embed(home, monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise AssertionError("w0 score must not scan or embed")

    monkeypatch.setattr("ragnarok.index.update", refuse)
    monkeypatch.setattr("ragnarok.index.connect", refuse)
    monkeypatch.setattr("ragnarok.metadata.iter_notes", refuse)
    monkeypatch.setattr("ragnarok.embed.Embedder", refuse, raising=False)
    assert _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", EDIT_STEP)) == 0
    assert len(_rows(home)) == 1
    assert not (home / "data" / "index.db").exists()
    log = home / "logs" / "w0.log"
    assert not log.exists() or "Error" not in log.read_text()


def test_w0_score_works_after_a_derived_ddl_change(home, monkeypatch, tmp_path):
    (home / "data").mkdir()
    core = sqlite3.connect(home / "data" / "core.db")
    core.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "INSERT INTO meta VALUES ('schema_version', '99'), ('derived_ddl', 'a-newer-hash');"
        "CREATE TABLE node (id TEXT PRIMARY KEY, a_newer_column TEXT);"
    )
    core.close()
    index = sqlite3.connect(home / "data" / "index.db")
    index.executescript("CREATE TABLE meta (key TEXT, value TEXT); INSERT INTO meta VALUES ('index_ddl', 'newer');")
    index.close()
    index_bytes = (home / "data" / "index.db").read_bytes()
    assert _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", EDIT_STEP)) == 0
    assert len(_rows(home)) == 1
    core = sqlite3.connect(home / "data" / "core.db")
    assert dict(core.execute("SELECT key, value FROM meta")) == {"schema_version": "99", "derived_ddl": "a-newer-hash"}
    core.close()
    assert (home / "data" / "index.db").read_bytes() == index_bytes


@pytest.mark.parametrize("bad", ["missing", "not-json", "bad-stdin", "bad-config"])
def test_w0_score_exits_zero_on_a_bad_transcript(home, monkeypatch, tmp_path, bad):
    path = tmp_path / "t.jsonl"
    if bad == "not-json":
        path.write_text("{ not json\n" + json.dumps(_user("hi")) + "\n{ also not json\n")
    if bad == "bad-config":
        (home / "config.toml").write_text("not_a_key = 1\n")
        _transcript(path, EDIT_STEP)
    if bad == "bad-stdin":
        monkeypatch.setattr("sys.stdin", io.StringIO("this is not the hook JSON"))
        assert cli.main(["w0", "score"]) == 0
    else:
        assert _score(home, monkeypatch, path) == 0
    if bad in ("missing", "bad-stdin", "bad-config"):
        assert "Error" in (home / "logs" / "w0.log").read_text()


def test_status_gives_the_age_of_the_last_step(vault):
    conn = connect(vault)
    now = time.time()
    rows = [
        (w0.utc(now - 3600), "s", "edit", "h1", 1),
        (w0.utc(now - 120), "s", "chat", "h2", 0),
        (w0.utc(now - 60), "s", "kb-save", "h3", None),
    ]
    conn.executemany(
        "INSERT INTO w0_step (created, session_id, step_kind, summary_hash, label_command) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    from ragnarok.index import status

    result = status(vault, conn)
    assert result["w0_steps"] == 3 and result["w0_labeled_steps"] == 1
    assert 55 <= result["w0_last_step_age_s"] <= 70


def test_status_age_is_none_without_steps(vault):
    from ragnarok.index import status

    result = status(vault, connect(vault))
    assert result["w0_steps"] == 0 and result["w0_last_step_age_s"] is None


def test_w0_import_copies_the_rows_one_time(home, monkeypatch, tmp_path, capsys):
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", EDIT_STEP))
    other = tmp_path / "other" / "core.db"
    other.parent.mkdir()
    src = sqlite3.connect(other)
    from ragnarok.index import KEPT_SCHEMA

    src.executescript(KEPT_SCHEMA.replace("main.", ""))
    [mine] = _rows(home)
    rows = [
        (mine["created"], mine["session_id"], "edit", mine["summary_hash"], 0),  # the same row
        ("2026-11-01T10:00:00Z", "s-9", "chat", "h-new", 1),
        ("2026-11-01T10:00:00Z", "s-9", "chat", "h-other", 0),
    ]
    src.executemany(
        "INSERT INTO w0_step (created, session_id, step_kind, summary_hash, label_command) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    src.commit()
    src.close()
    assert cli.main(["w0", "import", "--from", str(other)]) == 0
    assert capsys.readouterr().out.strip() == "imported 2 w0 steps"
    assert cli.main(["w0", "import", "--from", str(other)]) == 0
    assert capsys.readouterr().out.strip() == "imported 0 w0 steps"
    rows = _rows(home)
    assert len(rows) == 3
    assert {r["summary_hash"] for r in rows} == {mine["summary_hash"], "h-new", "h-other"}
    assert load_config().core_path == home / "data" / "core.db"


@pytest.mark.parametrize(("origin", "kind"), [("peer", "peer"), ("task-notification", "notification")])
def test_a_peer_or_notification_turn_is_a_step_of_its_own(home, monkeypatch, tmp_path, origin, kind):
    path = tmp_path / "t.jsonl"
    lines = list(EDIT_STEP)
    _score(home, monkeypatch, _transcript(path, lines))
    meta = origin == "peer"  # a teammate message is an isMeta line with an origin
    lines += [_user("<teammate-message>check it</teammate-message>", origin=origin, meta=meta)]
    lines += [_tool_use("Bash", command="ls"), _tool_result("Bash"), _say("Checked.")]
    _score(home, monkeypatch, _transcript(path, lines))
    first, second = _rows(home)
    assert second["step_kind"] == kind
    features = json.loads(second["features"])
    assert features["tools"] == {"Bash": 1} and features["files"] == []
    assert features["origin"] == origin
    assert "parser" not in second["summary"]


def test_the_skill_body_after_a_command_is_not_a_step_boundary(home, monkeypatch, tmp_path):
    lines = [*_kb_save("x"), _say("Saved.")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines))
    [row] = _rows(home)
    assert row["step_kind"] == "kb-save"


def test_the_log_holds_no_path_and_no_message(home, monkeypatch, tmp_path):
    marker = "MarkerFolderName"
    missing = tmp_path / f"-x-{marker}" / "a.jsonl"
    assert _score(home, monkeypatch, missing, cwd=tmp_path / marker) == 0
    log = (home / "logs" / "w0.log").read_text()
    assert marker not in log and str(tmp_path) not in log
    assert "Error: FileNotFoundError errno=2" in log
    assert "w0.py:" in log  # the frames: file:line:function


def test_a_repeated_stop_of_one_step_writes_one_row(home, monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    lines = [_user("work", origin="human", uuid="u-1"), _tool_use("Bash", command="ls"), _tool_result("Bash")]
    _score(home, monkeypatch, _transcript(path, lines))
    lines += [_say("Done.")]  # a blocking Stop hook made the turn continue; Stop fires again
    _score(home, monkeypatch, _transcript(path, lines))
    [row] = _rows(home)
    assert "Done." in row["summary"]  # the row holds the complete step, not the part before the block
    assert row["summary_hash"] == hashlib.sha256(row["summary"].encode()).hexdigest()
    lines += [_user("next", origin="human", uuid="u-2"), _say("Next done.")]
    _score(home, monkeypatch, _transcript(path, lines))
    first, second = _rows(home)
    assert json.loads(first["features"])["prompt_uuid"] == "u-1"
    assert json.loads(second["features"])["prompt_uuid"] == "u-2"


def test_the_kb_save_text_is_cut_after_the_redaction(home, monkeypatch, tmp_path):
    secret = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    text = f"{secret} " + "y" * 1000
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", [*_kb_save(text), _say("Saved.")]))
    saved = json.loads(_rows(home)[0]["features"])["kb_save_text"]
    assert saved.startswith("[REDACTED] y") and len(saved) == 400
    lines = [*_kb_save(text), _say("Saved.")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines), cwd=home / "vaults" / "Work" / "Secret")
    private = json.loads(_rows(home)[1]["features"])
    assert private["kb_save_hash"] == hashlib.sha256(saved.encode()).hexdigest()


def test_private_paths_are_resolved(tmp_path):
    from ragnarok.config import Config, W0Config

    real = tmp_path / "real"
    (real / "Work" / "Secret").mkdir(parents=True)
    (tmp_path / "ext" / "Hidden").mkdir(parents=True)
    (tmp_path / "linkroot").symlink_to(real)
    (tmp_path / "pp").mkdir()
    (tmp_path / "pplink").symlink_to(tmp_path / "pp")
    cfg = Config(
        root=tmp_path / "linkroot",
        vaults=["Work", "Ext"],
        vault_paths={"Ext": tmp_path / "ext"},
        private_patterns=["Work/Secret/**", "Ext/Hidden/**"],
        w0=W0Config(private_paths=[str(tmp_path / "pplink")]),
        index_path=tmp_path / "index.db",
    )
    assert w0.is_private_path(cfg, str(real / "Work" / "Secret" / "a.md"))  # the real path of a symlinked root
    assert w0.is_private_path(cfg, str(tmp_path / "linkroot" / "Work" / "Secret"))
    assert w0.is_private_path(cfg, str(tmp_path / "ext" / "Hidden" / "b.md"))  # a vault in vault_paths
    assert w0.is_private_path(cfg, str(tmp_path / "pp" / "c.md"))  # the real path of a symlinked prefix
    assert not w0.is_private_path(cfg, str(real / "Work" / "Open.md"))
    assert not w0.is_private_path(cfg, str(tmp_path / "ext" / "Open.md"))


def test_a_path_that_cannot_be_resolved_is_private(tmp_path, monkeypatch):
    from ragnarok.config import Config

    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db")

    def fail(path, *a, **k):
        raise OSError("loop")

    monkeypatch.setattr("os.path.realpath", fail)
    assert w0.is_private_path(cfg, "/anything/at/all.md")


@pytest.mark.parametrize("case", ["no-index", "mismatch"])
def test_status_prints_the_w0_block_without_a_usable_index(home, monkeypatch, tmp_path, capsys, case):
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", EDIT_STEP))
    if case == "mismatch":
        index = sqlite3.connect(home / "data" / "index.db")
        index.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        index.execute("INSERT INTO meta VALUES ('index_ddl', 'newer')")
        index.commit()
        index.close()
    capsys.readouterr()
    assert cli.main(["status"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["w0_steps"] == 1 and out["w0_labeled_steps"] == 0 and out["w0_last_step_age_s"] < 60


def test_a_repeated_stop_keeps_the_label(home, monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    lines = [_user("work", origin="human", uuid="u-1"), _say("Part one.")]
    _score(home, monkeypatch, _transcript(path, lines))
    lines += [*_kb_save("keep it"), _say("Saved.")]
    _score(home, monkeypatch, _transcript(path, lines))
    assert _rows(home)[0]["label_command"] == 1
    conn = sqlite3.connect(home / "data" / "core.db")
    conn.execute("UPDATE w0_step SET features = json_set(features, '$.prompt_uuid', 'u-1') WHERE id = 1")
    conn.execute("DELETE FROM w0_step WHERE id = 2")
    conn.commit()
    conn.close()
    lines = [_user("work", origin="human", uuid="u-1"), _say("Part one."), _say("Part two.")]
    _score(home, monkeypatch, _transcript(path, lines))
    [row] = _rows(home)
    assert row["label_command"] == 1 and "Part two." in row["summary"]


def test_kb_save_skips_a_peer_or_notification_row(home, monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    lines = list(EDIT_STEP)
    _score(home, monkeypatch, _transcript(path, lines))
    lines += [_user("<teammate-message>x</teammate-message>", origin="peer", meta=True), _say("Noted.")]
    _score(home, monkeypatch, _transcript(path, lines))
    lines += [_user("<task-notification>y</task-notification>", origin="task-notification"), _say("Seen.")]
    _score(home, monkeypatch, _transcript(path, lines))
    lines += [*_kb_save("keep the parser rule"), _say("Saved.")]
    _score(home, monkeypatch, _transcript(path, lines))
    human, peer, notification, save = _rows(home)
    assert human["label_command"] == 1
    assert peer["step_kind"] == "peer" and peer["label_command"] == 0
    assert notification["step_kind"] == "notification" and notification["label_command"] == 0
    assert save["label_command"] is None


def test_a_relative_search_path_resolves_against_the_session_cwd(home, monkeypatch, tmp_path):
    lines = [_user("look", origin="human"), _tool_use("Grep", pattern="plan", path="Secret"), _say("Found.")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines), cwd=home / "vaults" / "Work")
    [row] = _rows(home)
    assert row["summary"] is None and json.loads(row["features"])["private"] is True


# ---- ragnarok w0 backfill ----


def _stamp(lines, session="s-1", cwd="/work", start=1000):
    """Add a uuid, a timestamp, a cwd and the session id to each line, in order."""
    out = []
    for i, line in enumerate(lines):
        line = dict(line)
        line.setdefault("uuid", f"u-{session}-{i}")
        line["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(1_700_000_000 + start + i))
        line.setdefault("cwd", cwd)
        line["sessionId"] = session
        out.append(line)
    return out


@pytest.fixture
def projects(tmp_path):
    folder = tmp_path / "claude" / "proj-a"
    folder.mkdir(parents=True)
    return folder.parent


def _backfill(capsys, projects, *extra):
    assert cli.main(["w0", "backfill", "--projects", str(projects), *extra]) == 0
    return capsys.readouterr().out


TWO_STEPS = [
    _user("first prompt", origin="human"),
    _tool_use("Edit", file_path="/work/a.py"),
    _tool_result("Edit"),
    _say("Edited a."),
    _user("second prompt", origin="human"),
    _say("Just talk."),
]


def test_backfill_divides_a_transcript_into_steps(home, projects, capsys):
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(TWO_STEPS, cwd="/work"))
    out = _backfill(capsys, projects)
    assert "files 1" in out and "steps 2" in out and "inserted 2" in out
    first, second = _rows(home)
    assert first["step_kind"] == "edit" and second["step_kind"] == "chat"
    assert "/work/a.py" in first["summary"] and "Edited a." in first["summary"]
    assert "Edited a." not in second["summary"] and "Just talk." in second["summary"]
    assert first["session_id"] == "s-1"
    assert first["created"] == time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1_700_001_000))
    assert json.loads(first["features"])["prompt_uuid"] == "u-s-1-0"


def test_backfill_two_times_gives_the_same_rows(home, projects, capsys):
    lines = [*TWO_STEPS, *_kb_save("remember"), _say("Saved.")]
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(lines))
    _backfill(capsys, projects)
    before = _rows(home)
    out = _backfill(capsys, projects)
    assert "inserted 0" in out and "updated 0" in out
    assert _rows(home) == before


def test_backfill_completes_a_partial_last_step(home, projects, capsys):
    lines = _stamp(TWO_STEPS)
    path = _transcript(projects / "proj-a" / "s-1.jsonl", lines[:5])  # the second prompt, no answer yet
    _backfill(capsys, projects)
    assert "Just talk." not in _rows(home)[1]["summary"]
    _transcript(path, lines)
    out = _backfill(capsys, projects)
    assert "inserted 0" in out and "updated 1" in out
    rows = _rows(home)
    assert len(rows) == 2 and "Just talk." in rows[1]["summary"]


def test_backfill_skips_subagent_transcripts(home, projects, capsys):
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(TWO_STEPS))
    sub = projects / "proj-a" / "s-1" / "subagents"
    sub.mkdir(parents=True)
    _transcript(sub / "agent-x.jsonl", _stamp(TWO_STEPS, session="agent-x"))
    _transcript(projects / "loose.jsonl", _stamp(TWO_STEPS, session="loose"))
    out = _backfill(capsys, projects)
    assert "files 1" in out
    assert {r["session_id"] for r in _rows(home)} == {"s-1"}


def test_backfill_labels_from_kb_save(home, projects, capsys):
    lines = [
        _user("a"),
        _say("one"),
        _user("human step", origin="human"),
        _say("two"),
        _user("<peer>", origin="peer"),
        _say("three"),
        *_kb_save(),
        _say("Saved."),
    ]
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(lines))
    _backfill(capsys, projects)
    assert [(r["step_kind"], r["label_command"]) for r in _rows(home)] == [
        ("chat", 0),
        ("chat", 1),
        ("peer", 0),
        ("kb-save", None),
    ]


def test_backfill_continues_after_a_bad_file(home, projects, capsys):
    (projects / "proj-a" / "a-bad.jsonl").mkdir()  # reading it as a file raises
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(TWO_STEPS))
    out = _backfill(capsys, projects)
    assert "files 1" in out and "inserted 2" in out
    log = (home / "logs" / "w0.log").read_text()
    assert str(projects) not in log and len(log.splitlines()) >= 1


def test_backfill_skips_sessions_in_a_listed_home(home, projects, capsys, tmp_path):
    base = tmp_path / "baseline"
    base.mkdir()
    cfg = home / "config.toml"
    cfg.write_text(cfg.read_text().replace("[w0]\n", f'[w0]\nskip_homes = ["{base}"]\n'))
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(TWO_STEPS, cwd=str(base / "sub")))
    _transcript(projects / "proj-a" / "s-2.jsonl", _stamp(TWO_STEPS, session="s-2", cwd=str(home / "data")))
    _transcript(projects / "proj-a" / "s-3.jsonl", _stamp(TWO_STEPS, session="s-3", cwd="/work"))
    _backfill(capsys, projects)
    assert {r["session_id"] for r in _rows(home)} == {"s-3"}


def test_backfill_since_drops_earlier_steps(home, projects, capsys):
    _transcript(projects / "proj-a" / "s-1.jsonl", _stamp(TWO_STEPS))
    _backfill(capsys, projects, "--since", "2099-01-01")
    assert _rows(home) == []
