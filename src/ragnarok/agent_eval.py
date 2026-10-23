"""Agent evaluation: the same question to a Claude Code agent with Grep, Glob and Read, and to one with the
ragnarok MCP tools and Read, both with the same limit of tool calls.

Both agents work in a mirror of the vaults in <data_dir>/eval/vaults. The mirror holds only the notes that are
not private and not in eval_exclude (for example notes that quote the questions and the expected answers).
The MCP agent gets a server with its own home <data_dir>/eval: a config with the mirror as root, a copy of the
index and its own logs/server.log, from which the report reads the search latency.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import Config, dump_config, home_dir, load_config
from .evaluate import bootstrap, excluded, score_draft, score_hard
from .metadata import is_private, iter_notes

AGENTS = ["grep", "mcp"]
MCP_TOOLS = ["search", "find_projects", "list_project", "open_note", "find_by_jira"]
TOOL_HINT = {
    "grep": "Use the Grep, Glob and Read tools.",
    "mcp": "Use the ragnarok tools (search, find_projects, list_project, open_note, find_by_jira) and Read.",
}
PROMPT = """You search a folder of Obsidian notes: the vaults {vaults} are folders in the current directory. \
{hint}

Request: {question}

Find the notes that answer the request. If the request asks to continue or to start work, find the notes of \
the earlier work that it continues or needs. You can make at most {calls} tool calls. Then end your answer \
with one JSON object and nothing after it:
{{"notes": ["<note path relative to the current directory>", ...]}}
List at most 10 notes, the best first."""

Runner = Callable[[list[str], Path, dict], dict]


def eval_home(cfg: Config) -> Path:
    return cfg.data_dir / "eval"


def mirror_notes(cfg: Config) -> list[str]:
    return [rel for rel in iter_notes(cfg) if not is_private(cfg, rel) and not excluded(cfg, rel)]


def prepare(cfg: Config) -> Config:
    """Make the mirror, the config and the index copy of the evaluation home. Return the evaluation config."""
    home = eval_home(cfg)
    mirror = home / "vaults"
    shutil.rmtree(mirror, ignore_errors=True)
    for rel in mirror_notes(cfg):
        target = mirror / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.root / rel, target)
    (home / "config.toml").write_text(dump_config(cfg, root="vaults", data_dir="data"))
    ecfg = load_config(home / "config.toml")
    if not ecfg.index_path.exists():
        # a copy of the full index: the paths are the same, so the scan only removes the notes outside the mirror
        ecfg.index_path.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(cfg.index_path)
        dst = sqlite3.connect(ecfg.index_path)
        src.backup(dst)
        src.close()
        dst.close()
    from .embed import OllamaEmbedder
    from .index import connect, update

    conn = connect(ecfg)
    try:
        update(ecfg, conn, OllamaEmbedder(ecfg))  # before the run, so no search call pays for this scan
    finally:
        conn.close()
    return ecfg


def mcp_config(cfg: Config) -> Path:
    home = eval_home(cfg)
    path = home / "mcp.json"
    server = {
        "command": "uv",
        "args": ["--directory", str(home_dir()), "run", "ragnarok", "serve"],
        "env": {"RAGNAROK_HOME": str(home)},
    }
    path.write_text(json.dumps({"mcpServers": {"ragnarok": server}}, indent=1))
    return path


def command(agent: str, question: str, calls: int, model: str, mcp_json: Path | None, vaults: list[str]) -> list[str]:
    prompt = PROMPT.format(vaults=", ".join(vaults), hint=TOOL_HINT[agent], question=question, calls=calls)
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model, "--max-turns", str(calls + 2)]
    # restricted: no settings files, no hooks, file tools only inside the working directory
    cmd += ["--restricted", "--strict-mcp-config", "--permission-mode", "dontAsk", "--no-session-persistence"]
    if agent == "grep":
        return cmd + ["--tools", "Grep,Glob,Read", "--allowedTools", "Grep,Glob,Read"]
    tools = ",".join(f"mcp__ragnarok__{t}" for t in MCP_TOOLS)
    return cmd + ["--tools", "Read", "--mcp-config", str(mcp_json), "--allowedTools", f"Read,{tools}"]


def run_claude(cmd: list[str], cwd: Path, env: dict) -> dict:
    out = subprocess.run(cmd, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True, timeout=900)
    try:
        return json.loads(out.stdout)
    except ValueError:
        return {"is_error": True, "result": "", "stderr": out.stderr[-2000:], "stdout": out.stdout[-2000:]}


def parse_notes(text: str, mirror: Path) -> list[str]:
    """The note paths of the last {"notes": [...]} object in the answer, relative to the mirror."""
    found = None
    for m in re.finditer(r'\{\s*"notes"\s*:\s*\[.*?\]\s*\}', text or "", re.S):
        try:
            found = json.loads(m.group(0))["notes"]
        except ValueError:
            continue
    notes = []
    for n in found or []:
        n = str(n).strip()
        if n.startswith(str(mirror)):
            n = n[len(str(mirror)) :]
        n = n.lstrip("./").lstrip("/")
        if n and n not in notes:
            notes.append(n)
    return notes


def run_all(
    cfg: Config,
    sets: dict[str, list[dict]],
    out_dir: Path,
    calls: int,
    model: str,
    workers: int = 4,
    runner: Runner = run_claude,
) -> dict[str, dict[str, dict]]:
    """Run each question with each agent. A result file that exists is used again, so a stopped run continues."""
    ecfg = prepare(cfg)
    mcp_json = mcp_config(cfg)
    mirror = ecfg.root
    jobs = []
    for name, questions in sets.items():
        for q in questions:
            for agent in AGENTS:
                jobs.append((name, q, agent, out_dir / name / f"{q['id']}.{agent}.json"))

    def one(job):
        name, q, agent, path = job
        if not path.exists():
            raw = runner(command(agent, q["question"], calls, model, mcp_json, cfg.vaults), mirror, {})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw, indent=1))
        raw = json.loads(path.read_text())
        return name, q["id"], agent, raw

    results: dict[str, dict[str, dict]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for name, qid, agent, raw in pool.map(one, jobs):
            r = results.setdefault(name, {}).setdefault(qid, {})
            r[agent] = {
                "notes": parse_notes(raw.get("result", ""), mirror),
                "cost": raw.get("total_cost_usd") or 0.0,
                "turns": raw.get("num_turns"),
                "subtype": raw.get("subtype"),
                "error": bool(raw.get("is_error")),
            }
    return results


def search_latency(log: Path) -> list[int]:
    """The time in ms of each search call with no changed note in its scan."""
    if not log.exists():
        return []
    pattern = re.compile(r"call=search ms=(\d+) scan_ms=\S+ scan_changed=0 result=ok")
    return [int(m.group(1)) for m in pattern.finditer(log.read_text())]


def _interval(values: list[float]) -> str:
    mean, lo, hi = bootstrap(values, [0.0] * len(values))
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def _diff(rows: list[dict], measure: str) -> str:
    d, lo, hi = bootstrap([r[f"mcp_{measure}"] for r in rows], [r[f"grep_{measure}"] for r in rows])
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def report(
    cfg: Config, sets: dict[str, list[dict]], results: dict[str, dict[str, dict]], latency: list[int]
) -> list[str]:
    known = set(mirror_notes(cfg))
    draft, hard = [], []
    for name, questions in sets.items():
        for q in questions:
            r = results.get(name, {}).get(q["id"])
            if not r:
                continue
            ranked = {a: r[a]["notes"] for a in AGENTS}
            if "subtype" in q:
                hard.append(score_hard(q, ranked, AGENTS))
            elif s := score_draft(q, ranked, known, AGENTS):
                draft.append(s)
    cont = [r for r in hard if r["type"] == "continuation"]
    cat = [r for r in hard if r["type"] == "category"]
    synth = [r for r in draft if str(r["type"]).startswith("synthesis")]

    def mean(rows, key):
        return f"{np.mean([r[key] for r in rows]):.2f}" if rows else "-"

    lines = ["| Quality attribute | Target | Measured | 95% interval |", "|---|---|---|---|"]
    if draft:
        lines.append(
            f"| Retrieval, draft set (n={len(draft)}) | MCP - grep recall@K >= +0.10, interval above 0 | "
            f"grep {mean(draft, 'grep_recall')}, MCP {mean(draft, 'mcp_recall')} | {_diff(draft, 'recall')} |"
        )
    if cont:
        lines.append(
            f"| Retrieval, continuation (n={len(cont)}) | MCP - grep project hit@5 >= +0.10 | "
            f"grep {mean(cont, 'grep_proj@5')}, MCP {mean(cont, 'mcp_proj@5')} | {_diff(cont, 'proj@5')} |"
        )
    if cat:
        lines.append(
            f"| Retrieval, category (n={len(cat)}) | MCP - grep area precision@5 >= +0.10 | "
            f"grep {mean(cat, 'grep_prec@5')}, MCP {mean(cat, 'mcp_prec@5')} | {_diff(cat, 'prec@5')} |"
        )
    if cont:
        lines.append(
            f"| Project continuation (n={len(cont)}) | MCP primary hit@5 >= 0.80 | "
            f"MCP {mean(cont, 'mcp_prim@5')} (grep {mean(cont, 'grep_prim@5')}) | "
            f"{_interval([r['mcp_prim@5'] for r in cont])} |"
        )
    if synth:
        lines.append(
            f"| Cross-vault synthesis (n={len(synth)}) | MCP recall@10 >= 0.80 | "
            f"MCP {mean(synth, 'mcp_recall')} (grep {mean(synth, 'grep_recall')}) | "
            f"{_interval([r['mcp_recall'] for r in synth])} |"
        )
    if latency:
        p50, p95 = np.percentile(latency, [50, 95])
        under = sum(ms < 500 for ms in latency) / len(latency)
        lines.append(
            f"| Query latency (n={len(latency)} search calls) | < 500 ms for each call | "
            f"p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {max(latency)} ms, {under:.0%} under 500 ms | - |"
        )
    runs = [r[a] for s in results.values() for r in s.values() for a in AGENTS if a in r]
    cost = {a: sum(r[a]["cost"] for s in results.values() for r in s.values() if a in r) for a in AGENTS}
    failed = sum(1 for r in runs if r["error"] or not r["notes"])
    lines += [
        "",
        f"runs {len(runs)}, without a note list {failed}",
        f"API-equivalent cost (no bill on a subscription): grep ${cost['grep']:.2f}, MCP ${cost['mcp']:.2f}",
    ]
    return lines


def evaluate(cfg: Config, sets_dir: Path, names: list[str], out_dir: Path, calls: int, model: str, limit: int | None):
    sets = {}
    for name in names:
        questions = json.loads((sets_dir / f"{name}.json").read_text())
        sets[name] = questions[:limit] if limit else questions
    results = run_all(cfg, sets, out_dir, calls, model)
    latency = search_latency(eval_home(cfg) / "logs" / "server.log")
    return report(cfg, sets, results, latency)
