"""`ragnarok replay run`: the kb-write skill on each case of a replay set, in fixture mode, and its score (R2).

For each case the runner makes a case home under `<set>/runs/<run id>/<case id>/`:
- `mirror/`: a real copy of the source mirror (no symbolic link) in the start state of the case: the newer note is
  left out, and the `remove_keys` of the old notes are removed from their frontmatter;
- `data/index.db`: a copy of the source index, brought in line with the case mirror from the embedding cache. An
  embedder that refuses every call proves that no chunk gets a new embedding;
- `config.toml` (the source config with `root` set to the case mirror), the marker `.ragnarok-bench`, `draft.md`
  (the draft form of the case) and `mcp.json` (one MCP server that runs this ragnarok with the case home).

The agent is `claude -p` with only that MCP server, the skill file to read (the plugin is not loaded: its own MCP
config points to the real vaults), Write and Edit allowed only under the case mirror, a cost limit and a time limit.
The runner snapshots the case mirror before and after the agent; the snapshot diff, not the agent's record, gives
the changed files. The source mirror and the source home are snapshotted before and after the whole run.

The scorer compares the action and the targets of the agent's final record (`KB-WRITE-RECORD`) with the expected
values of the case, and checks that `written` agrees with the action and the snapshot. The skill can rewrite the
draft text, so the text is not compared.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from .config import Config, dump_config, load_config
from .embed import EmbedError
from .metadata import parse_note
from .replay import MARKER, code_commit, wilson

PREFIX = "KB-WRITE-RECORD"
SERVER = "ragnarok-bench"
WRITE_ACTIONS = ("new", "supersede", "merge")
AS_OF_KEYS = ("as_of", "updated", "created", "date")
EMPTY_DIFF = {"created": [], "changed": [], "deleted": []}
TOOLS = (
    "Read",
    "Glob",
    "Grep",
    "ToolSearch",
    f"mcp__{SERVER}",
    "Bash(bash:*)",
    "Bash(KB_WRITE_WITHIN=*)",
    "Bash(test:*)",
    "Bash(diff:*)",
    "Bash(mktemp:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(date:*)",
)


class SourceChanged(RuntimeError):
    """A file of the source mirror or the source home changed during the run."""


@dataclass
class AgentOutput:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class CaseHome:
    home: Path
    mirror: Path
    draft: Path
    mcp_config: Path
    as_of: str | None
    index: dict


Runner = Callable[[list[str], str, Path, dict, float], AgentOutput]


# --- the final record --------------------------------------------------------------------------------------------


def parse_record(text: str | None) -> tuple[dict | None, str | None]:
    """The JSON object of the last `KB-WRITE-RECORD` line of text, or None and the reason."""
    lines = [line.strip() for line in (text or "").splitlines()]
    found = [line for line in lines if line.startswith(PREFIX + " ") or line == PREFIX]
    if not found:
        return None, f"no {PREFIX} line in the answer"
    try:
        record = json.loads(found[-1][len(PREFIX) :])
    except json.JSONDecodeError as e:
        return None, f"the {PREFIX} line is not valid JSON: {e}"
    if not isinstance(record, dict):
        return None, f"the {PREFIX} line is not a JSON object"
    return record, None


def parse_agent_output(stdout: str) -> dict:
    """The answer, cost, turns and permission denials of the JSON output of `claude -p --output-format json`."""
    try:
        data = json.loads(stdout)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except ValueError as e:
        return {
            "result": None,
            "cost_usd": None,
            "turns": None,
            "is_error": True,
            "permission_denials": None,
            "error": f"the agent output is not JSON: {e}",
        }
    return {
        "result": data.get("result"),
        "cost_usd": data.get("total_cost_usd"),
        "turns": data.get("num_turns"),
        "is_error": bool(data.get("is_error")),
        "permission_denials": len(data.get("permission_denials") or []),
        "error": None,
    }


# --- snapshots ---------------------------------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def snapshot(folder: Path, exclude: tuple[Path, ...] = ()) -> dict[str, str]:
    """{path relative to folder: sha256} of all files under folder; a symbolic link gives its target text."""
    out = {}
    skip = {p.resolve() for p in exclude}
    for top, dirs, files in os.walk(folder):
        base = Path(top)
        dirs[:] = [d for d in dirs if (base / d).resolve() not in skip]
        for name in files:
            path = base / name
            rel = path.relative_to(folder).as_posix()
            out[rel] = "link:" + os.readlink(path) if path.is_symlink() else _file_hash(path)
    return out


def diff_snapshots(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "created": sorted(after.keys() - before.keys()),
        "changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
        "deleted": sorted(before.keys() - after.keys()),
    }


# --- the case home -----------------------------------------------------------------------------------------------


def strip_keys(text: str, keys: list[str]) -> str:
    """text with the named top-level frontmatter keys (and their indented or list lines) removed."""
    if not keys or not text.startswith("---\n"):
        return text
    end = text.find("\n---", 3)
    if end < 0:
        return text
    head, rest = text[4:end].split("\n"), text[end:]
    out, skip = [], False
    for line in head:
        key = line.split(":", 1)[0] if ":" in line and not line[:1].isspace() and not line.startswith("-") else None
        if key is not None:
            skip = key.strip() in keys
        elif not (line.startswith((" ", "\t", "-"))):
            skip = False
        if not skip:
            out.append(line)
    return "---\n" + "\n".join(out) + rest


def note_as_of(fm: dict) -> str | None:
    """The date of the claims of a note: as_of, else updated, else created, else date."""
    for key in AS_OF_KEYS:
        value = fm.get(key)
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if value:
            return str(value)[:10]
    return None


class CacheOnlyEmbedder:
    """An embedder for a case index: every chunk must be in the embedding cache already."""

    def embed_docs(self, items):
        raise EmbedError(f"{len(items)} chunks are not in the embedding cache; a case home makes no new embedding")

    def embed_query(self, query):
        raise EmbedError("a case home makes no query embedding")


def mcp_command() -> list[str]:
    """The MCP server of this ragnarok install: the same Python as the runner."""
    return [sys.executable, "-m", "ragnarok.cli", "serve"]


def build_case_home(case: dict, cfg: Config, run_dir: Path, form: str) -> CaseHome:
    """The home of one case under run_dir: the start mirror, its index, config, marker, draft and MCP config."""
    from .index import connect, update

    if cfg.vault_paths:
        raise ValueError("a replay run needs all vaults under root (no vault_paths)")
    home = run_dir / case["id"]
    mirror = home / "mirror"
    home.mkdir(parents=True)
    shutil.copytree(cfg.root, mirror, symlinks=False)
    state = case.get("start_state") or {}
    if state.get("exclude"):
        (mirror / state["exclude"]).unlink()
    for key, keys in (state.get("remove_keys") or {}).items():
        if keys:
            path = mirror / key
            path.write_text(strip_keys(path.read_text(), keys))
    (home / "data").mkdir()
    for suffix in ("", "-wal"):
        src = cfg.index_path.with_name(cfg.index_path.name + suffix)
        if src.exists():
            shutil.copy2(src, home / "data" / ("index.db" + suffix))
    (home / "config.toml").write_text(dump_config(cfg, root=mirror, data_dir="data"))
    (home / MARKER).write_text(f"case home of a replay run: {case['id']}\n")
    draft = home / "draft.md"
    draft.write_text(case["drafts"][form])
    mcp = home / "mcp.json"
    server = {"command": mcp_command()[0], "args": mcp_command()[1:], "env": {"RAGNAROK_HOME": str(home)}}
    mcp.write_text(json.dumps({"mcpServers": {SERVER: server}}, indent=2) + "\n")

    newer = cfg.note_path(case["newer"]) if case.get("newer") else None
    as_of = note_as_of(parse_note(newer.read_text(errors="replace"))[0]) if newer and newer.is_file() else None

    case_cfg = load_config(home / "config.toml")
    conn = connect(case_cfg)
    try:
        result = update(case_cfg, conn, CacheOnlyEmbedder())
    finally:
        conn.close()
    index = {k: v for k, v in result.__dict__.items() if isinstance(v, int | float)}
    return CaseHome(home, mirror, draft, mcp, as_of, index)


# --- the agent ---------------------------------------------------------------------------------------------------


def agent_command(claude: str, case_home: CaseHome, plugin_root: Path, *, budget: float | None, model: str | None):
    mirror = str(case_home.mirror)
    if any(ch.isspace() or ch == "," for ch in mirror):
        raise ValueError(f"the case mirror path must have no space or comma: {mirror}")
    argv = [
        claude, "-p", "--output-format", "json", "--strict-mcp-config", "--mcp-config", str(case_home.mcp_config),
        "--permission-mode", "dontAsk", "--no-session-persistence",
        "--add-dir", str(case_home.home), "--add-dir", str(plugin_root),
    ]  # fmt: skip
    if model:
        argv += ["--model", model]
    if budget is not None:
        argv += ["--max-budget-usd", str(budget)]
    # a path rule with a leading // is an absolute path
    return argv + ["--allowedTools", *TOOLS, f"Write(/{mirror}/**)", f"Edit(/{mirror}/**)"]


def agent_prompt(case_home: CaseHome, skill: Path, plugin_root: Path) -> str:
    date_line = (
        f"- The date of the claims of the candidate text (its `as_of`) is {case_home.as_of}."
        if case_home.as_of
        else "- The candidate text has no known date."
    )
    return f"""Run the kb-write skill in fixture mode, as a replay runner calls it.

- The skill is the file {skill}. Read it in full first and follow it exactly.
- CLAUDE_PLUGIN_ROOT is {plugin_root}. The helper script is {plugin_root}/scripts/kb-write.sh; run it with `bash`.
- mirror={case_home.mirror}
- The fixture ragnarok MCP server is `{SERVER}`. Its tools have the prefix `mcp__{SERVER}__`
  (for example `mcp__{SERVER}__find_collisions`). Use them where the skill names
  `mcp__plugin_ragnarok_ragnarok__*`. Load them with one ToolSearch `select:` call.
- The candidate text is the content of the file {case_home.draft}. No target path is given.
{date_line}
- Write and edit files only under the mirror.
- Never ask a question. End with the KB-WRITE-RECORD line as the last line of your answer.
"""


def subprocess_runner(argv: list[str], prompt: str, cwd: Path, env: dict, timeout: float) -> AgentOutput:
    """Run the agent with the prompt on stdin; on the time limit, kill its process group."""
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, env=env,
            start_new_session=True,
        )  # fmt: skip
    except OSError as e:
        return AgentOutput(None, "", str(e), time.monotonic() - start)
    try:
        out, err = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return AgentOutput(None, out or "", err or "", time.monotonic() - start, timed_out=True)
    return AgentOutput(proc.returncode, out, err, time.monotonic() - start)


# --- the scorer --------------------------------------------------------------------------------------------------


def expected_of(case: dict) -> tuple[str, list[str]]:
    """The expected action and targets: positive supersede or merge (merge when an old note expects it), duplicate
    same, negative new with no targets."""
    olds = [o["path"] for o in case.get("old", [])]
    if case["kind"] == "negative":
        return "new", []
    if case["kind"] == "duplicate":
        return "same", olds
    if any(o["expected"] == "merge" for o in case["old"]):
        return "merge", olds
    return "supersede", olds


def _consistent(record: dict, diff: dict) -> bool:
    """written agrees with the action, and the snapshot shows the change of a writing action (nothing else)."""
    if diff["deleted"]:
        return False
    action, written = record.get("action"), record.get("written") is True
    created, changed = diff["created"], set(diff["changed"])
    targets = set(record.get("targets") or [])
    if action not in WRITE_ACTIONS:
        return not written and not created and not changed
    if not written:
        return False
    if action == "new":
        return bool(created)
    if action == "supersede":
        return bool(created) and bool(targets) and targets <= changed
    return bool(targets) and targets <= changed  # merge: the target and each superseded note change


def score(case: dict, record: dict | None, diff: dict) -> dict:
    expected, expected_targets = expected_of(case)
    action = record.get("action") if record else None
    targets = set(record.get("targets") or []) if record else set()
    action_ok = record is not None and action == expected
    target_ok = record is not None and (not targets if expected == "new" else set(expected_targets) <= targets)
    written_ok = record is not None and _consistent(record, diff)
    correct = action_ok and target_ok and written_ok
    merge_for_supersede = expected == "supersede" and action == "merge"
    if record is None:
        category = "failed"
    elif action == "refused" or record.get("outcome") == "refused":
        category = "refused"
    elif action == "ask":
        category = "ask"
    elif correct:
        category = "correct"
    elif merge_for_supersede:
        category = "merge-for-supersede"
    else:
        category = "wrong"
    return {
        "expected_action": expected,
        "expected_targets": expected_targets,
        "action": action,
        "targets": sorted(targets),
        "action_ok": action_ok,
        "target_ok": target_ok,
        "written_ok": written_ok,
        "correct": correct,
        "merge_for_supersede": merge_for_supersede,
        "category": category,
    }


def _rate(rows: list[dict]) -> dict:
    hits, n = sum(r["score"]["correct"] for r in rows), len(rows)
    low, high = wilson(hits, n)
    return {"correct": hits, "cases": n, "rate": hits / n if n else None, "low": low, "high": high}


def summarize(rows: list[dict]) -> dict:
    """correct / n with the 95% Wilson interval: all, without ask, by kind and by positive kind; the categories and
    the total cost."""
    kinds = sorted({r["kind"] for r in rows})
    positive_kinds = sorted({r["positive_kind"] for r in rows if r["positive_kind"]})
    return {
        "all": _rate(rows),
        "without_ask": _rate([r for r in rows if r["score"]["category"] != "ask"]),
        "by_kind": {k: _rate([r for r in rows if r["kind"] == k]) for k in kinds},
        "by_positive_kind": {k: _rate([r for r in rows if r["positive_kind"] == k]) for k in positive_kinds},
        "categories": dict(Counter(r["score"]["category"] for r in rows)),
        "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
    }


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_summary(summary: dict) -> str:
    lines = [f"{'subset':24} {'correct':>9} {'rate':>6}  95% interval"]
    groups = [("all", summary["all"]), ("without ask", summary["without_ask"])]
    groups += [(f"kind {k}", v) for k, v in summary["by_kind"].items()]
    groups += [(f"positive {k}", v) for k, v in summary["by_positive_kind"].items()]
    for name, r in groups:
        lines.append(
            f"{name:24} {r['correct']:>4}/{r['cases']:<4} {_num(r['rate']):>6}  {_num(r['low'])} - {_num(r['high'])}"
        )
    lines.append("categories: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["categories"].items())))
    lines.append(f"cost: ${summary['cost_usd']:.2f}")
    return "\n".join(lines)


# --- the run -----------------------------------------------------------------------------------------------------


def check_bench(home: Path) -> None:
    if not (home / MARKER).is_file():
        raise ValueError(f"{home} holds no {MARKER}: a replay run needs a bench home as RAGNAROK_HOME")


def run_case(
    case: dict,
    cfg: Config,
    run_dir: Path,
    *,
    form: str,
    skill: Path,
    claude: str,
    model: str | None,
    budget: float | None,
    timeout: float,
    runner: Runner,
) -> dict:
    row = {
        "case": case["id"], "kind": case["kind"], "positive_kind": case.get("positive_kind"), "form": form,
        "as_of": None, "home": None, "index": None, "record": None, "record_error": None, "diff": EMPTY_DIFF,
        "exit_code": None, "timed_out": False, "cost_usd": None, "turns": None, "duration_s": None,
        "permission_denials": None, "error": None,
    }  # fmt: skip
    plugin_root = skill.parent.parent.parent
    try:
        case_home = build_case_home(case, cfg, run_dir, form)
        argv = agent_command(claude, case_home, plugin_root, budget=budget, model=model)
    except (OSError, ValueError, EmbedError) as e:
        row["error"] = f"case home: {e}"
        row["score"] = score(case, None, EMPTY_DIFF)
        return row
    row.update(as_of=case_home.as_of, home=str(case_home.home), index=case_home.index)
    prompt = agent_prompt(case_home, skill, plugin_root)
    (case_home.home / "prompt.txt").write_text(prompt)
    env = {**os.environ, "RAGNAROK_HOME": str(case_home.home)}
    before = snapshot(case_home.mirror)
    out = runner(argv, prompt, case_home.home, env, timeout)
    row["diff"] = diff_snapshots(before, snapshot(case_home.mirror))
    (case_home.home / "agent.json").write_text(out.stdout)
    (case_home.home / "agent.stderr").write_text(out.stderr)
    agent = parse_agent_output(out.stdout)
    row.update(
        exit_code=out.exit_code,
        timed_out=out.timed_out,
        duration_s=round(out.duration_s, 1),
        cost_usd=agent["cost_usd"],
        turns=agent["turns"],
        permission_denials=agent["permission_denials"],
    )
    record, error = parse_record(agent["result"])
    row["record"], row["record_error"] = record, agent["error"] or error
    if out.timed_out:
        row["error"] = f"the agent passed the time limit of {timeout:g} s"
    row["score"] = score(case, record, row["diff"])
    return row


def run_cases(
    cfg: Config,
    home: Path,
    set_dir: Path,
    cases: list[dict],
    *,
    skill: Path,
    form: str = "bare",
    parallel: int = 1,
    budget: float | None = None,
    claude: str = "claude",
    model: str | None = None,
    timeout: float = 1800,
    out: Path | None = None,
    runner: Runner | None = None,
) -> tuple[list[dict], dict, Path]:
    """Run each case; write records.jsonl and summary.json into the run folder. Raise SourceChanged (after the
    files are written) when a file of the source mirror or the source home changed."""
    runner = runner or subprocess_runner
    check_bench(home)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    run_dir = (out or set_dir / "runs" / run_id).resolve()
    if run_dir.is_relative_to(cfg.root.resolve()):
        raise ValueError(f"the run folder {run_dir} is inside the source mirror")
    run_dir.mkdir(parents=True)
    skip = (set_dir / "runs", run_dir)
    sources = {"mirror": (cfg.root, ()), "home": (home, skip)}
    before = {name: snapshot(folder, ex) for name, (folder, ex) in sources.items()}

    lock = threading.Lock()
    records = run_dir / "records.jsonl"

    def one(case: dict) -> dict:
        row = run_case(case, cfg, run_dir, form=form, skill=skill, claude=claude, model=model, budget=budget,
                       timeout=timeout, runner=runner)  # fmt: skip
        with lock, records.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        rows = list(pool.map(one, cases))

    changes = {}
    for name, (folder, ex) in sources.items():
        d = diff_snapshots(before[name], snapshot(folder, ex))
        if any(d.values()):
            changes[name] = d
    summary = {
        **summarize(rows),
        "source_unchanged": not changes,
        "source_changes": changes,
        "run_id": run_dir.name,
        "set": str(set_dir),
        "form": form,
        "cases": [c["id"] for c in cases],
        "skill": str(skill),
        "model": model,
        "max_budget_usd": budget,
        "timeout_s": timeout,
        "commit": code_commit(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if changes:
        raise SourceChanged(f"files outside the case mirrors changed: {json.dumps(changes)}")
    return rows, summary, run_dir
