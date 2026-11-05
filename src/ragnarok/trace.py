"""The trace: one JSON line for each read call in `data_dir/trace/<date>.jsonl`.

The names follow the OpenTelemetry GenAI conventions where one exists. The line holds query text and
document paths, so it stays local (git ignores `data/`)."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

from .config import Config

TOP = 50  # documents of each pool that the trace records
GOLD_ENV = "RAGNAROK_TRACE_GOLD"  # a JSON list of gold note paths; the evaluation sets it


def trace_dir(cfg: Config) -> Path:
    return cfg.data_dir / "trace"


@functools.cache
def _commit() -> str:
    checkout = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_hash(cfg: Config) -> str:
    """A hash of every configuration value, so that a changed parameter shows in the trace."""
    return hashlib.sha1(json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()


def index_version(cfg: Config, conn) -> dict:
    hashes = [h for (h,) in conn.execute("SELECT hash FROM doc ORDER BY id")]
    return {
        "commit": _commit(),
        "config_hash": config_hash(cfg),
        "model": cfg.model,
        "min_words": cfg.min_words,
        "max_words": cfg.max_words,
        "corpus_hash": hashlib.sha1("\n".join(hashes).encode()).hexdigest(),
        "counts": {"docs": len(hashes), "units": conn.execute("SELECT count(*) FROM unit").fetchone()[0]},
    }


def gold_paths() -> list[str]:
    raw = os.environ.get(GOLD_ENV)
    return [str(p) for p in json.loads(raw)] if raw else []


def purge(cfg: Config, today: dt.date | None = None) -> list[Path]:
    """Delete the trace files older than trace.retention_days. A file is named by its date."""
    today = today or dt.datetime.now(dt.UTC).date()
    cutoff = today - dt.timedelta(days=cfg.trace.retention_days)
    removed = []
    for path in sorted(trace_dir(cfg).glob("*.jsonl")):
        try:
            day = dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if day < cutoff:
            path.unlink()
            removed.append(path)
    return removed


class Trace:
    """The fields of one call. `finish()` writes the line."""

    def __init__(self, cfg: Config, conn, tool: str, arguments: dict, query: str | None = None, ids=(None, None)):
        self.cfg, self.conn = cfg, conn
        self.id = uuid.uuid4().hex
        self.start = time.perf_counter()
        self.ms: dict[str, int] = {}
        self.fields: dict = {
            "trace_id": self.id,
            "gen_ai.conversation.id": ids[1],
            "mcp.session.id": ids[0],
            "gen_ai.tool.name": tool,
            "gen_ai.tool.call.arguments": arguments,
            "gen_ai.retrieval.query.text": query,
            "ragnarok.index.version": None,
            "ragnarok.plan": None,
            "ragnarok.entities": [],
            "ragnarok.candidates": [],
            "gen_ai.retrieval.documents": [],
            "ragnarok.returned.status": [],
            "ragnarok.ms": self.ms,
            "error.type": None,
        }

    @contextlib.contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0) + round((time.perf_counter() - start) * 1000)

    def set(self, key: str, value) -> None:
        self.fields[key] = value

    def finish(self, error: str | None = None) -> None:
        self.ms["total"] = round((time.perf_counter() - self.start) * 1000)
        self.fields["error.type"] = error
        self.fields["ragnarok.index.version"] = index_version(self.cfg, self.conn)
        folder = trace_dir(self.cfg)
        folder.mkdir(parents=True, exist_ok=True)
        day = dt.datetime.now(dt.UTC).date().isoformat()
        with (folder / f"{day}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.fields, ensure_ascii=False) + "\n")
