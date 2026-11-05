"""Agent evaluation: the same question to Claude Code agents with the same limit of tool calls. The agents are
mcp-kb (the ragnarok MCP tools of this checkout, and Read), mcp-base (the MCP tools of the baseline checkout on
the baseline home, and Read) and, as an option, grep (Grep, Glob and Read). Each question runs several times.
Each run keeps the raw stream-json lines of `claude -p`; the tool sequence and the trace IDs come from them.

All agents work in a mirror of the vaults in <data_dir>/eval/vaults. The mirror holds only the notes that are
not private and not in eval_exclude (for example notes that quote the questions and the expected answers),
and the generated notes of <data_dir>/eval/distractors.
The mcp-kb agent gets a server with its own home <data_dir>/eval: a config with the mirror as root, copies of
core.db and index.db (the update drops the notes that are not in the mirror) and its own logs/server.log, from
which the report reads the search latency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import Config, dump_config, load_config
from .evaluate import bootstrap, excluded, score_draft, score_hard
from .metadata import is_private, iter_notes

AGENTS = ["mcp-kb", "mcp-base", "grep"]
DEFAULT_AGENTS = ["mcp-kb", "mcp-base"]
BASE_TOOLS = ["search", "find_projects", "list_project", "open_note", "find_by_jira"]
MCP_TOOLS = {
    "mcp-base": BASE_TOOLS,
    "mcp-kb": [*BASE_TOOLS, "open_projects", "kb_check", "find_entity", "get_entity", "related"],
}
TOOL_HINT = {
    "grep": "Use the Grep, Glob and Read tools.",
    "mcp-kb": f"Use the ragnarok tools ({', '.join(MCP_TOOLS['mcp-kb'])}) and Read.",
    "mcp-base": f"Use the ragnarok tools ({', '.join(MCP_TOOLS['mcp-base'])}) and Read.",
}
PROMPT = """You search a folder of Obsidian notes: the vaults {vaults} are folders in the current directory. \
{hint}

Request: {question}

Find the notes that answer the request. If the request asks to continue or to start work, find the notes of \
the earlier work that it continues or needs. You can make at most {calls} tool calls. Then end your answer \
with one JSON object and nothing after it:
{{"notes": ["<note path relative to the current directory>", ...]}}
List at most 10 notes, the best first."""
# the prompt of the questions that have a `class` field: one final JSON object with four keys
PROMPT_NEW = """You search a folder of Obsidian notes: the vaults {vaults} are folders in the current directory. \
{hint}

Request: {question}

Find the notes that answer the request. You can make at most {calls} tool calls. Then end your answer with one \
JSON object and nothing after it:
{{"notes": ["<note path relative to the current directory>", ...], "entities": ["<entity id>", ...], \
"route": "kb|skill|none", "answer": "<your answer>"}}
Use the route "kb" when the notes hold the answer, "skill" when the answer needs a live value from a system \
of record, and "none" when no note answers. List at most 10 notes, the best first."""

# a runner gives the raw stdout of `claude -p ... --output-format stream-json --verbose`, its stderr and its exit code
Runner = Callable[[list[str], Path, dict], tuple[str, str, int]]


def eval_home(cfg: Config) -> Path:
    return cfg.data_dir / "eval"


def mirror_notes(cfg: Config) -> list[str]:
    return [rel for rel in iter_notes(cfg) if not is_private(cfg, rel) and not excluded(cfg, rel)]


def prepare(cfg: Config, reuse_mirror: bool = False) -> Config:
    """Make the mirror, the config and the copies of core.db and index.db of the evaluation home.

    Return the evaluation config.

    After the vault notes, copy the files of <home>/distractors into the mirror, with the same relative paths.
    With reuse_mirror and a mirror that exists, keep the mirror and copy nothing into it.
    """
    home = eval_home(cfg)
    mirror = home / "vaults"
    if not (reuse_mirror and mirror.is_dir()):
        shutil.rmtree(mirror, ignore_errors=True)
        for rel in mirror_notes(cfg):
            target = mirror / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg.note_path(rel), target)
        distractors = home / "distractors"
        for path in sorted(p for p in distractors.rglob("*") if p.is_file()):
            target = mirror / path.relative_to(distractors)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (home / "config.toml").write_text(dump_config(replace(cfg, vault_paths={}), root="vaults", data_dir="data"))
    ecfg = load_config(home / "config.toml")
    # the files of the full home: the mirror index finds its notes and the cache in them and embeds nothing
    copied = False
    for src_path, dst_path in ((cfg.core_path, ecfg.core_path), (cfg.index_path, ecfg.index_path)):
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            src = sqlite3.connect(src_path)
            dst = sqlite3.connect(dst_path)
            src.backup(dst)
            src.close()
            dst.close()
            copied = True
    from .embed import OllamaEmbedder
    from .index import connect, update

    conn = connect(ecfg)
    try:
        update(ecfg, conn, OllamaEmbedder(ecfg))  # before the run, so no search call pays for this scan
        if copied:
            # the copies held the private and excluded notes of the full home: merge the FTS5 segments, so that
            # their postings go, then free their pages
            conn.execute("INSERT INTO unit_fts (unit_fts) VALUES ('optimize')")
            conn.commit()
            conn.execute("VACUUM main")
            conn.execute("VACUUM idx")
    finally:
        conn.close()
    return ecfg


def mirror_hash(mirror: Path) -> str:
    """One SHA-256 over the sorted relative paths and the contents of the mirror files."""
    if not mirror.is_dir():
        raise FileNotFoundError(f"no mirror at {mirror}; run `ragnarok eval prepare`")
    digest = hashlib.sha256()
    for path in sorted(p for p in mirror.rglob("*") if p.is_file()):
        digest.update(path.relative_to(mirror).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


# the keys of the version 1 Config; the baseline load_config() refuses any other key
V1_KEYS = (
    "vaults", "skip_dirs", "private_patterns", "allow_private", "project_markers", "archive_folder",
    "jira_prefixes", "eval_exclude", "ollama_url", "model", "query_prefix", "doc_prefix", "dimension",
    "min_words", "max_words",
)  # fmt: skip


def write_baseline_home(cfg: Config, home: Path, mirror: Path) -> Path:
    """Write config.toml of the baseline home: the v1 keys only, with the mirror of the bet home as root."""
    lines = [f"root = {json.dumps(str(mirror.resolve()))}", 'data_dir = "data"']
    for key in V1_KEYS:
        value = getattr(cfg, key)
        lines.append(f"{key} = {json.dumps(sorted(value) if isinstance(value, set) else value)}")
    for vault, rules in cfg.vault_rules.items():
        lines.append(f"\n[vault_rules.{json.dumps(vault)}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in asdict(rules).items()]
    home.mkdir(parents=True, exist_ok=True)
    path = home / "config.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def mcp_config(
    cfg: Config, agent: str, name: str, qid: str, gold: list[str], baseline: tuple[Path, Path] | None
) -> Path:
    """The MCP configuration of one question: its own server environment, with the gold paths for the trace.

    mcp-base starts the server of the baseline checkout on the baseline home; mcp-kb starts this checkout on
    the evaluation home. The baseline code ignores RAGNAROK_TRACE_GOLD.
    """
    if agent == "mcp-base":
        if baseline is None:
            raise ValueError("the agent mcp-base needs baseline_checkout and baseline_home in config.toml")
        checkout, home = baseline
    else:
        checkout, home = Path(__file__).resolve().parents[2], eval_home(cfg)
    path = eval_home(cfg) / "mcp" / f"{name}.{qid}.{agent}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    server = {
        "command": "uv",
        "args": ["--directory", str(checkout), "run", "ragnarok", "serve"],
        "env": {"RAGNAROK_HOME": str(home), "RAGNAROK_TRACE_GOLD": json.dumps(gold)},
    }
    path.write_text(json.dumps({"mcpServers": {"ragnarok": server}}, indent=1))
    return path


def gold_paths(q: dict) -> list[str]:
    """The gold note paths of a question: `gold` of the new sets, else the `expected` entries that are notes."""
    if "gold" in q:
        return [g["path"] for g in q["gold"]]
    return [e for e in q.get("expected", []) if e.endswith(".md")]


def command(
    agent: str,
    question: str,
    calls: int,
    model: str,
    mcp_json: Path | None,
    vaults: list[str],
    new_prompt: bool = False,
) -> list[str]:
    template = PROMPT_NEW if new_prompt else PROMPT
    prompt = template.format(vaults=", ".join(vaults), hint=TOOL_HINT[agent], question=question, calls=calls)
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", model]
    cmd += ["--max-turns", str(calls + 2)]
    # restricted: no settings files, no hooks, file tools only inside the working directory
    cmd += ["--restricted", "--strict-mcp-config", "--permission-mode", "dontAsk", "--no-session-persistence"]
    if agent == "grep":
        return cmd + ["--tools", "Grep,Glob,Read", "--allowedTools", "Grep,Glob,Read"]
    tools = ",".join(f"mcp__ragnarok__{t}" for t in MCP_TOOLS[agent])
    return cmd + ["--tools", "Read", "--mcp-config", str(mcp_json), "--allowedTools", f"Read,{tools}"]


def run_claude(cmd: list[str], cwd: Path, env: dict) -> tuple[str, str, int]:
    out = subprocess.run(cmd, cwd=cwd, env={**os.environ, **env}, capture_output=True, text=True, timeout=900)
    return out.stdout, out.stderr, out.returncode


TRACE_ID = re.compile(r'"trace_id"\s*:\s*"([^"]+)"')


def _result_text(content) -> str:
    if isinstance(content, list):
        return "".join(str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content)
    return str(content or "")


def parse_stream(text: str) -> dict:
    """The tool sequence and the final result of the raw stream-json lines of one run.

    tools has one item for each tool_use block, in order: name, input and the trace_id values of its result.
    """
    tools: list[dict] = []
    by_id: dict[str, dict] = {}
    final: dict = {}
    for line in (text or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            final = event
            continue
        content = (event.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                item = {"name": block.get("name"), "input": block.get("input"), "trace_ids": []}
                tools.append(item)
                by_id[block.get("id")] = item
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in by_id:
                by_id[block["tool_use_id"]]["trace_ids"] += TRACE_ID.findall(_result_text(block.get("content")))
    return {
        "tools": tools,
        "trace_ids": [t for item in tools for t in item["trace_ids"]],
        "turns": final.get("num_turns"),
        "cost": final.get("total_cost_usd") or 0.0,
        "subtype": final.get("subtype"),
        "error": bool(final.get("is_error")) or not final,
        "answer_text": final.get("result") or "",
    }


def parse_answer(text: str, mirror: Path | None = None) -> dict:
    """The last JSON object with a `notes` list in the answer: notes relative to the mirror, entities, route, answer."""
    found: dict = {}
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("notes"), list):
            found = obj
    notes = []
    for n in found.get("notes", []):
        n = str(n).strip()
        if mirror is not None and n.startswith(str(mirror)):
            n = n[len(str(mirror)) :]
        n = n.lstrip("./").lstrip("/")
        if n and n not in notes:
            notes.append(n)
    return {
        "notes": notes,
        "entities": [str(e) for e in found.get("entities") or []],
        "route": found.get("route"),
        "answer": found.get("answer"),
    }


def parse_notes(text: str, mirror: Path) -> list[str]:
    """The note paths of the last {"notes": [...]} object in the answer, relative to the mirror."""
    return parse_answer(text, mirror)["notes"]


def run_all(
    cfg: Config,
    sets: dict[str, list[dict]],
    out_dir: Path,
    calls: int,
    model: str,
    agents: list[str] = DEFAULT_AGENTS,
    runs: int = 2,
    workers: int = 4,
    runner: Runner = run_claude,
    reuse_mirror: bool = False,
    baseline: tuple[Path, Path] | None = None,
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """Run each question runs times with each agent. Give results[set][question id][agent] = one dict for each run.

    The raw lines of each run are in <out_dir>/<set>/<id>.<agent>.<run>.jsonl. A run with a stderr or a
    non-zero exit code also has <id>.<agent>.<run>.stderr: the exit code on the first line, then the stderr. A
    file that exists is used again, so a stopped run continues. Run number n of one agent pairs with run
    number n of the other.
    """
    if "mcp-base" in agents and baseline is None:
        raise ValueError("the agent mcp-base needs baseline_checkout and baseline_home in config.toml")
    ecfg = prepare(cfg, reuse_mirror=reuse_mirror)
    mirror = ecfg.root
    jobs = []
    for name, questions in sets.items():
        for q in questions:
            for agent in agents:
                # one file for each question and agent, written before the pool starts: the runs only read it
                mcp_json = None if agent == "grep" else mcp_config(cfg, agent, name, q["id"], gold_paths(q), baseline)
                for n in range(1, runs + 1):
                    jobs.append((name, q, agent, n, out_dir / name / f"{q['id']}.{agent}.{n}.jsonl", mcp_json))

    def one(job):
        name, q, agent, n, path, mcp_json = job
        if not path.exists():
            cmd = command(agent, q["question"], calls, model, mcp_json, cfg.vaults, "class" in q)
            path.with_suffix(".stderr").unlink(missing_ok=True)
            raw, stderr, code = runner(cmd, mirror, {})
            path.parent.mkdir(parents=True, exist_ok=True)
            if stderr or code:
                path.with_suffix(".stderr").write_text(f"{code}\n{stderr}")
            path.write_text(raw)
        run = parse_stream(path.read_text())
        run.update(parse_answer(run.pop("answer_text"), mirror))
        return name, q["id"], agent, n, run

    results: dict[str, dict[str, dict[str, list[dict]]]] = {}
    with ThreadPoolExecutor(workers) as pool:
        for name, qid, agent, _, run in pool.map(one, jobs):  # map keeps the job order, so runs stay in order
            results.setdefault(name, {}).setdefault(qid, {}).setdefault(agent, []).append(run)
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


def _mean_rows(rows: list[dict]) -> dict:
    """One row from the rows of the runs of a question: the mean of each number, the first row for the rest."""
    out = dict(rows[0])
    for key, value in rows[0].items():
        if isinstance(value, float):
            out[key] = float(np.mean([row[key] for row in rows]))
    return out


def _diff(rows: list[dict], measure: str, test: str, ref: str) -> str:
    d, lo, hi = bootstrap([r[f"{test}_{measure}"] for r in rows], [r[f"{ref}_{measure}"] for r in rows])
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def report(
    cfg: Config,
    sets: dict[str, list[dict]],
    results: dict[str, dict[str, dict[str, list[dict]]]],
    latency: list[int],
    agents: list[str] = DEFAULT_AGENTS,
) -> list[str]:
    """The quality table. The first agent is the tested one, the last is the reference.

    Run n of one agent pairs with run n of the other.
    """
    test, ref = agents[0], agents[-1]
    known = set(mirror_notes(cfg))
    draft, hard = [], []
    for name, questions in sets.items():
        for q in questions:
            r = results.get(name, {}).get(q["id"])
            if not r:
                continue
            # the unit is the question: score each paired run, then take the mean of each measure over the runs
            scores = []
            for n in range(min(len(r[a]) for a in agents)):
                ranked = {a: r[a][n]["notes"] for a in agents}
                if "subtype" in q:
                    scores.append(score_hard(q, ranked, agents))
                elif s := score_draft(q, ranked, known, agents):
                    scores.append(s)
            if scores:
                (hard if "subtype" in q else draft).append(_mean_rows(scores))
    cont = [r for r in hard if r["type"] == "continuation"]
    cat = [r for r in hard if r["type"] == "category"]
    synth = [r for r in draft if str(r["type"]).startswith("synthesis")]

    def mean(rows, key):
        return f"{np.mean([r[key] for r in rows]):.2f}" if rows else "-"

    def both(rows, measure):
        return f"{ref} {mean(rows, f'{ref}_{measure}')}, {test} {mean(rows, f'{test}_{measure}')}"

    lines = ["| Quality attribute | Target | Measured | 95% interval |", "|---|---|---|---|"]
    if draft:
        lines.append(
            f"| Retrieval, draft set (n={len(draft)}) | {test} - {ref} recall@K >= +0.10, interval above 0 | "
            f"{both(draft, 'recall')} | {_diff(draft, 'recall', test, ref)} |"
        )
    if cont:
        lines.append(
            f"| Retrieval, continuation (n={len(cont)}) | {test} - {ref} project hit@5 >= +0.10 | "
            f"{both(cont, 'proj@5')} | {_diff(cont, 'proj@5', test, ref)} |"
        )
    if cat:
        lines.append(
            f"| Retrieval, category (n={len(cat)}) | {test} - {ref} area precision@5 >= +0.10 | "
            f"{both(cat, 'prec@5')} | {_diff(cat, 'prec@5', test, ref)} |"
        )
    if cont:
        lines.append(
            f"| Project continuation (n={len(cont)}) | {test} primary hit@5 >= 0.80 | "
            f"{test} {mean(cont, f'{test}_prim@5')} ({ref} {mean(cont, f'{ref}_prim@5')}) | "
            f"{_interval([r[f'{test}_prim@5'] for r in cont])} |"
        )
    if synth:
        lines.append(
            f"| Cross-vault synthesis (n={len(synth)}) | {test} recall@10 >= 0.80 | "
            f"{test} {mean(synth, f'{test}_recall')} ({ref} {mean(synth, f'{ref}_recall')}) | "
            f"{_interval([r[f'{test}_recall'] for r in synth])} |"
        )
    if latency:
        p50, p95 = np.percentile(latency, [50, 95])
        under = sum(ms < 500 for ms in latency) / len(latency)
        lines.append(
            f"| Query latency (n={len(latency)} search calls) | < 500 ms for each call | "
            f"p50 {p50:.0f} ms, p95 {p95:.0f} ms, max {max(latency)} ms, {under:.0%} under 500 ms | - |"
        )
    runs = [run for s in results.values() for r in s.values() for a in agents if a in r for run in r[a]]
    cost = {a: sum(run["cost"] for s in results.values() for r in s.values() for run in r.get(a, [])) for a in agents}
    turns = {
        a: [run["turns"] for s in results.values() for r in s.values() for run in r.get(a, []) if run["turns"]]
        for a in agents
    }
    failed = sum(1 for run in runs if run["error"] or not run["notes"])
    lines += [
        "",
        f"runs {len(runs)}, without a note list {failed}",
        "mean turns: " + ", ".join(f"{a} {np.mean(turns[a]):.1f}" for a in agents if turns[a]),
        "API-equivalent cost (no bill on a subscription): " + ", ".join(f"{a} ${cost[a]:.2f}" for a in agents),
    ]
    return lines


def evaluate(
    cfg: Config,
    sets_dir: Path,
    names: list[str],
    out_dir: Path,
    calls: int,
    model: str,
    limit: int | None,
    reuse_mirror: bool = False,
    agents: list[str] = DEFAULT_AGENTS,
    runs: int = 2,
):
    sets = {}
    for name in names:
        questions = json.loads((sets_dir / f"{name}.json").read_text())
        sets[name] = questions[:limit] if limit else questions
    baseline = (
        (Path(cfg.baseline_checkout), Path(cfg.baseline_home)) if cfg.baseline_checkout and cfg.baseline_home else None
    )
    results = run_all(cfg, sets, out_dir, calls, model, agents, runs, reuse_mirror=reuse_mirror, baseline=baseline)
    latency = search_latency(eval_home(cfg) / "logs" / "server.log")
    return report(cfg, sets, results, latency, agents)
