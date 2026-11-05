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


def _user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_use(name, **inputs):
    block = {"type": "tool_use", "id": f"t-{name}", "name": name, "input": inputs}
    return {"type": "assistant", "message": {"role": "assistant", "content": [block]}}


def _tool_result(name):
    block = {"type": "tool_result", "tool_use_id": f"t-{name}", "content": "ok"}
    return {"type": "user", "message": {"role": "user", "content": [block]}}


def _say(text):
    return {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def _kb_save(args=""):
    text = "<command-message>kb-save</command-message>\n<command-name>/kb-save</command-name>"
    return _user(text + (f"\n<command-args>{args}</command-args>" if args else ""))


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
    _user("fix the parser"),
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


@pytest.mark.parametrize("where", ["cwd", "edit", "read", "private_path", "relative_private_path"])
def test_private_step_keeps_only_the_hash(home, monkeypatch, tmp_path, where):
    secret_file = str(home / "vaults" / "Work" / "Secret" / "plan.md")
    outside = str(home / "private" / "notes.md")
    cwd = {"cwd": home / "vaults" / "Work" / "Secret"}.get(where, home)
    tool = {
        "edit": _tool_use("Edit", file_path=secret_file),
        "read": _tool_use("Read", file_path=secret_file),
        "private_path": _tool_use("Write", file_path=outside),
        "relative_private_path": _tool_use("Edit", file_path=str(home / "vaults" / "Work" / "Diary" / "day.md")),
    }.get(where, _tool_use("Bash", command="ls"))
    lines = [_user("work"), tool, _tool_result("x"), _say("done with the secret plan")]
    _score(home, monkeypatch, _transcript(tmp_path / "t.jsonl", lines), cwd=cwd)
    lines += [_kb_save("the secret plan matters"), _say("Saved.")]
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
    lines += [_kb_save("keep the parser rule"), _say("Saved.")]
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
    assert "2" in capsys.readouterr().out
    assert cli.main(["w0", "import", "--from", str(other)]) == 0
    assert "0" in capsys.readouterr().out
    rows = _rows(home)
    assert len(rows) == 3
    assert {r["summary_hash"] for r in rows} == {mine["summary_hash"], "h-new", "h-other"}
    assert load_config().core_path == home / "data" / "core.db"
