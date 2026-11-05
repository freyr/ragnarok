"""W0 step records: the step summary of a Claude Code turn and its explicit-command label, with no evaluator.

`ragnarok w0 score` runs from a Stop hook. It reads the hook JSON on stdin, makes the summary of the last step
of the transcript (the lines after the last user prompt that is not a tool result) and writes one w0_step row to
core.db. It never scans the vaults, never embeds and never calls a model. Each error goes to logs/w0.log and the
command exits with 0, so that it never stops a Claude Code session.

A step is private when its cwd, a changed file, a file that Read opened or a folder that Grep or Glob searched is
private. A Bash command does not make a step private: its paths are not parsed.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path

from .config import Config
from .redact import redact

CHANGE_TOOLS = ("Edit", "Write", "NotebookEdit", "MultiEdit")
SEARCH_TOOLS = ("Grep", "Glob")
# the step kind of a turn that a message, not a person, starts (the origin.kind of its first user line)
ORIGIN_KINDS = {"peer": "peer", "task-notification": "notification"}
COMMAND = re.compile(r"<command-name>\s*/?([^<\s]+)\s*</command-name>")
COMMAND_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.S)
KB_SAVE = "kb-save"


def utc(seconds: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Step:
    prompt: str = ""
    origin: str | None = None
    prompt_uuid: str | None = None
    tools: Counter = field(default_factory=Counter)
    changed: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    final_text: str = ""

    @property
    def command(self) -> str | None:
        m = COMMAND.search(self.prompt)
        return m.group(1) if m else None

    @property
    def command_args(self) -> str:
        m = COMMAND_ARGS.search(self.prompt)
        return m.group(1).strip() if m else ""

    @property
    def kind(self) -> str:
        if self.command == KB_SAVE:
            return KB_SAVE
        if self.command:
            return "command"
        if self.origin in ORIGIN_KINDS:
            return ORIGIN_KINDS[self.origin]
        if self.changed:
            return "edit"
        return "tool" if self.tools else "chat"


def _blocks(entry: dict) -> list:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _origin(entry: dict) -> str | None:
    origin = entry.get("origin")
    kind = origin.get("kind") if isinstance(origin, dict) else None
    return str(kind) if kind else None


def _prompt_text(entry: dict) -> str | None:
    """The text of the user line that starts a step, or None when the line does not start one.

    A line with an origin.kind (a person, a teammate message, a task notification) starts a step, also when it is
    an isMeta line. Without an origin, a non-meta text line starts a step; an isMeta line with no origin (the skill
    body after a slash command, a hook context) does not. A tool result never starts a step.
    """
    if entry.get("type") != "user" or entry.get("isCompactSummary"):
        return None
    blocks = _blocks(entry)
    if any(b.get("type") == "tool_result" for b in blocks):
        return None
    text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if _origin(entry):
        return text
    if entry.get("isMeta") or not text or text.lstrip().startswith("<local-command-"):
        return None
    return text


def last_step(path: Path) -> Step:
    """The last step of the transcript. Only the lines after the last user prompt are parsed (from the end)."""
    entries = []
    prompt: dict | None = None
    for raw in reversed(path.read_bytes().splitlines()):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue  # a damaged or partial line
        if not isinstance(entry, dict):
            continue
        if _prompt_text(entry) is not None:
            prompt = entry
            break
        entries.append(entry)
    step = Step()
    if prompt:
        step.prompt, step.origin = _prompt_text(prompt) or "", _origin(prompt)
        uuid = prompt.get("uuid")
        step.prompt_uuid = str(uuid) if uuid else None
    for entry in reversed(entries):
        if entry.get("type") != "assistant":
            continue
        texts = []
        for block in _blocks(entry):
            if block.get("type") == "tool_use":
                name = str(block.get("name", "?"))
                step.tools[name] += 1
                inputs = block.get("input") or {}
                if name in SEARCH_TOOLS:
                    target = inputs.get("path")
                else:
                    target = inputs.get("file_path") or inputs.get("notebook_path")
                if isinstance(target, str) and target:
                    reads = name == "Read" or name in SEARCH_TOOLS
                    bucket = step.changed if name in CHANGE_TOOLS else step.read if reads else None
                    if bucket is not None and target not in bucket:
                        bucket.append(target)
            elif block.get("type") == "text" and block.get("text", "").strip():
                texts.append(block["text"].strip())
        if texts:
            step.final_text = "\n".join(texts)
    return step


def _real(path: str | Path) -> Path:
    """The absolute path with ~ expanded, .. removed and symlinks resolved. The file need not exist."""
    return Path(os.path.realpath(os.path.expanduser(path)))


def _note_keys(cfg: Config, p: Path) -> list[str]:
    """The keys <vault>/<path in the vault> of the path, one for each vault folder (also from vault_paths) that
    holds it, and the path relative to root."""
    keys = []
    for vault in cfg.vaults:
        base = _real(cfg.vault_dir(vault))
        if p.is_relative_to(base):
            keys.append(f"{vault}/{p.relative_to(base).as_posix()}")
    root = _real(cfg.root)
    if p.is_relative_to(root):
        keys.append(p.relative_to(root).as_posix())
    return keys


def is_private_path(cfg: Config, path: str, cwd: str = "") -> bool:
    """True when the path matches private_patterns or starts with a prefix in w0.private_paths.

    A pattern matches a note key <vault>/<path in the vault>. A prefix is absolute, starts with ~, or is relative
    to root. A relative path (a Grep or Glob path) is relative to the session cwd, not to the cwd of this process.
    Both sides are resolved. A path that cannot be resolved counts as private.
    """
    try:
        given = Path(path).expanduser()
        if not given.is_absolute():
            if not cwd:
                return True  # no base to resolve it against
            given = Path(cwd).expanduser() / given
        p = _real(given)
        root = cfg.root.expanduser()
        if any(p.is_relative_to(_real(root / Path(prefix).expanduser())) for prefix in cfg.w0.private_paths):
            return True
        # the folder itself (a cwd) matches a pattern of the form Folder/**
        return any(
            fnmatchcase(key, pat) or fnmatchcase(key + "/", pat)
            for key in _note_keys(cfg, p)
            for pat in cfg.private_patterns
        )
    except (OSError, ValueError, RuntimeError):
        return True


def summary_text(step: Step) -> str:
    tools = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in step.tools.items()) or "none"
    files = ", ".join(step.changed) or "none"
    return f"kind: {step.kind}\ntools: {tools}\nfiles: {files}\ntext: {step.final_text}"


def make_row(cfg: Config, session_id: str, cwd: str, step: Step) -> dict:
    """The w0_step row of one step. A private step keeps only the hash of the summary and of the /kb-save text."""
    summary = redact(summary_text(step))[0][: cfg.w0.summary_chars]
    private = any(is_private_path(cfg, p, cwd) for p in [cwd, *step.changed, *step.read] if p)
    features: dict = {"tools": dict(step.tools)}
    if step.origin:
        features["origin"] = step.origin
    if step.prompt_uuid:
        features["prompt_uuid"] = step.prompt_uuid
    save_text = None
    if step.kind == KB_SAVE and step.command_args:  # redacted, then cut; the hash is on the cut text
        save_text = redact(step.command_args)[0][: cfg.w0.summary_chars]
    if private:
        features["private"] = True
        if save_text:
            features["kb_save_hash"] = sha256(save_text)
    else:
        features["files"] = [redact(f)[0] for f in step.changed]
        if save_text:
            features["kb_save_text"] = save_text
    return {
        "created": utc(),
        "session_id": session_id,
        "step_kind": step.kind,
        "summary": None if private else summary,
        "summary_hash": sha256(summary),
        "features": json.dumps(features, sort_keys=True),
        "mode": "shadow",
        "label_command": None if step.kind == KB_SAVE else 0,
    }


def write_step(cfg: Config, row: dict) -> None:
    from .index import connect_core

    conn = connect_core(cfg)
    try:
        with conn:
            uuid = json.loads(row["features"]).get("prompt_uuid")
            last = conn.execute(
                "SELECT id, json_extract(features, '$.prompt_uuid') FROM w0_step WHERE session_id = ?"
                " ORDER BY id DESC LIMIT 1",
                (row["session_id"],),
            ).fetchone()
            if uuid and last and last[1] == uuid:
                # a repeated Stop of the same step (a blocking Stop hook): the new parse holds the complete step;
                # the row keeps its id, created and label_command
                conn.execute(
                    "UPDATE w0_step SET step_kind = ?, summary = ?, summary_hash = ?, features = ? WHERE id = ?",
                    (row["step_kind"], row["summary"], row["summary_hash"], row["features"], last[0]),
                )
                return
            if row["step_kind"] == KB_SAVE:
                # the step of the user before the command: not a kb-save, a teammate turn or a task notification
                skipped = (KB_SAVE, *ORIGIN_KINDS.values())
                conn.execute(
                    "UPDATE w0_step SET label_command = 1 WHERE id = (SELECT max(id) FROM w0_step"
                    " WHERE session_id = ? AND step_kind NOT IN (?, ?, ?)"
                    " AND coalesce(json_extract(features, '$.origin'), 'human') = 'human')",
                    (row["session_id"], *skipped),
                )
            columns = ", ".join(row)
            conn.execute(f"INSERT INTO w0_step ({columns}) VALUES ({', '.join('?' * len(row))})", list(row.values()))
    finally:
        conn.close()


def score(cfg: Config, hook_json: str) -> dict:
    """Write the w0_step row of the last step of the transcript in the Stop hook JSON. Return the row."""
    hook = json.loads(hook_json)
    session_id, transcript = str(hook["session_id"]), Path(hook["transcript_path"]).expanduser()
    row = make_row(cfg, session_id, str(hook.get("cwd") or ""), last_step(transcript))
    write_step(cfg, row)
    return row


def log_error(error: BaseException) -> None:
    """Append the error to <home>/logs/w0.log. A failure here is ignored: the hook must never fail."""
    try:
        from .config import home_dir

        logs = home_dir() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        # no message text: the message of an OS error holds the transcript path, which holds the cwd
        errno = getattr(error, "errno", None)
        head = f"{utc()} Error: {type(error).__name__}" + (f" errno={errno}" if errno is not None else "")
        frames = [f"  {Path(f.filename).name}:{f.lineno}:{f.name}" for f in traceback.extract_tb(error.__traceback__)]
        with (logs / "w0.log").open("a") as f:
            f.write("\n".join([head, *frames]) + "\n")
    except Exception:  # noqa: BLE001
        pass


def import_steps(cfg: Config, source: Path) -> int:
    """Copy the w0_step rows of another core file. Skip a row with the same session_id, created and summary_hash."""
    from .index import connect_core

    conn = connect_core(cfg)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (f"{Path(source).resolve().as_uri()}?mode=ro",))
        columns = [r[1] for r in conn.execute("PRAGMA main.table_info(w0_step)") if r[1] != "id"]
        names = ", ".join(columns)
        with conn:
            cursor = conn.execute(
                f"INSERT INTO main.w0_step ({names}) SELECT {names} FROM src.w0_step s WHERE NOT EXISTS"
                " (SELECT 1 FROM main.w0_step m WHERE m.session_id = s.session_id AND m.created = s.created"
                " AND m.summary_hash = s.summary_hash) ORDER BY s.id"
            )
        return cursor.rowcount
    finally:
        conn.close()


def w0_status(conn) -> dict:
    """The W0 block of `ragnarok status`: the count of steps, of labeled steps and the age of the last step."""
    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    last = one("SELECT max(created) FROM w0_step")
    age = int(time.time() - calendar.timegm(time.strptime(last, "%Y-%m-%dT%H:%M:%SZ"))) if last else None
    return {
        "w0_steps": one("SELECT count(*) FROM w0_step"),
        "w0_labeled_steps": one("SELECT count(*) FROM w0_step WHERE label_command = 1"),
        "w0_last_step_age_s": age,
    }
