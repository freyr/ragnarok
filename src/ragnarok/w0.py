"""W0 step records: the step summary of a Claude Code turn and its explicit-command label, with no evaluator.

`ragnarok w0 score` runs from a Stop hook. It reads the hook JSON on stdin, makes the summary of the last step
of the transcript (the lines after the last user prompt that is not a tool result) and writes one w0_step row to
core.db. It never scans the vaults, never embeds and never calls a model. Each error goes to logs/w0.log and the
command exits with 0, so that it never stops a Claude Code session.
"""

from __future__ import annotations

import hashlib
import json
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
        if self.changed:
            return "edit"
        return "tool" if self.tools else "chat"


def _blocks(entry: dict) -> list:
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content or [] if isinstance(b, dict)]


def _prompt_text(entry: dict) -> str | None:
    """The text of a user prompt, or None when the line is not one (a tool result, a meta line, another type)."""
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isCompactSummary"):
        return None
    blocks = _blocks(entry)
    if any(b.get("type") == "tool_result" for b in blocks):
        return None
    text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if not text or text.lstrip().startswith("<local-command-"):
        return None
    return text


def last_step(path: Path) -> Step:
    """The last step of the transcript. Only the lines after the last user prompt are parsed (from the end)."""
    entries = []
    prompt = ""
    for raw in reversed(path.read_bytes().splitlines()):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except ValueError:
            continue  # a damaged or partial line
        if not isinstance(entry, dict):
            continue
        text = _prompt_text(entry)
        if text is not None:
            prompt = text
            break
        entries.append(entry)
    step = Step(prompt=prompt)
    for entry in reversed(entries):
        if entry.get("type") != "assistant":
            continue
        texts = []
        for block in _blocks(entry):
            if block.get("type") == "tool_use":
                name = str(block.get("name", "?"))
                step.tools[name] += 1
                inputs = block.get("input") or {}
                target = inputs.get("file_path") or inputs.get("notebook_path")
                if isinstance(target, str):
                    bucket = step.changed if name in CHANGE_TOOLS else step.read if name == "Read" else None
                    if bucket is not None and target not in bucket:
                        bucket.append(target)
            elif block.get("type") == "text" and block.get("text", "").strip():
                texts.append(block["text"].strip())
        if texts:
            step.final_text = "\n".join(texts)
    return step


def is_private_path(cfg: Config, path: str) -> bool:
    """True when the path matches private_patterns or starts with a prefix in w0.private_paths.

    A pattern is relative to root. A prefix is absolute, starts with ~, or is relative to root.
    """
    p = Path(path).expanduser()
    for prefix in cfg.w0.private_paths:
        if p.is_relative_to(cfg.root / Path(prefix).expanduser()):
            return True
    if p.is_relative_to(cfg.root):
        rel = p.relative_to(cfg.root).as_posix()
        # the folder itself (a cwd) matches a pattern of the form Folder/**
        return any(fnmatchcase(rel, pat) or fnmatchcase(rel + "/", pat) for pat in cfg.private_patterns)
    return False


def summary_text(step: Step) -> str:
    tools = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in step.tools.items()) or "none"
    files = ", ".join(step.changed) or "none"
    return f"kind: {step.kind}\ntools: {tools}\nfiles: {files}\ntext: {step.final_text}"


def make_row(cfg: Config, session_id: str, cwd: str, step: Step) -> dict:
    """The w0_step row of one step. A private step keeps only the hash of the summary and of the /kb-save text."""
    summary = redact(summary_text(step))[0][: cfg.w0.summary_chars]
    private = any(is_private_path(cfg, p) for p in [cwd, *step.changed, *step.read] if p)
    features: dict = {"tools": dict(step.tools)}
    save_text = redact(step.command_args)[0] if step.kind == KB_SAVE and step.command_args else None
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
            if row["step_kind"] == KB_SAVE:
                conn.execute(
                    "UPDATE w0_step SET label_command = 1 WHERE id = (SELECT max(id) FROM w0_step"
                    " WHERE session_id = ? AND step_kind != ?)",
                    (row["session_id"], KB_SAVE),
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
        detail = "".join(traceback.format_exception(error)).rstrip()
        with (logs / "w0.log").open("a") as f:
            f.write(f"{utc()} Error: {type(error).__name__}: {error}\n{detail}\n")
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
