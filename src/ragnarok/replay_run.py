"""`ragnarok replay run`: the kb-write skill on each case of a replay set, in fixture mode, and its score (R2).

For each case the runner makes a case home under `<set>/runs/<run id>/<case id>/`:
- `mirror/`: a real copy of the source mirror (no symbolic link) in the start state of the case: the newer note is
  left out, and the `remove_keys` of the old notes are removed from their frontmatter;
- `data/index.db`: a copy of the source index, brought in line with the case mirror from the embedding cache. An
  embedder that refuses every call proves that no chunk gets a new embedding. It is deleted after the case unless
  the run keeps the homes;
- `config.toml` (the source config with `root` set to the case mirror), the marker `.ragnarok-bench`, `draft.md`
  (the draft form of the case), `mcp.json` (one MCP server that runs this ragnarok with the case home),
  `settings.json` (the only settings of the agent) and `tmp/` (the TMPDIR of the agent).

The agent is `claude -p` with no user, project or local settings (so no user hook, plugin or permission rule), only
the case MCP server, the built-in tools that the skill needs, permission rules for the exact helper commands, Write
and Edit only under the case mirror, and the Bash sandbox with writes only to the case mirror and the case TMPDIR.
It runs in the case home with a minimal environment, a cost limit and a time limit. The plugin is not loaded: its
own MCP config points to the real vaults; the agent reads the skill file.

The runner snapshots the case mirror before and after the agent; the snapshot diff, not the agent's record, gives
the changed files. The source mirror is compared by sha256 before and after the run. Each file with a modification
time after the start of the run under the source home (outside the case mirrors, the case TMPDIRs, the server data
and the runner's own files), the plugin folder and each guard folder is an outside change, and fails the run.

The scorer compares the action and the targets of the agent's final record (`KB-WRITE-RECORD`) with the expected
values of the case, and checks that the snapshot diff holds exactly the files of that action. The skill can rewrite
the draft text, so the text is not compared.
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
BUILTIN_TOOLS = "Bash,Read,Write,Edit,Glob,Grep,ToolSearch"
SHELL_COMMANDS = ("test", "diff", "mktemp", "ls", "cat", "date")
ENV_KEYS = ("HOME", "PATH", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL", "TERM")
# the folders of a case home that the agent (mirror, tmp) or the MCP server (data, logs) writes
CASE_WRITABLE = ("mirror", "tmp", "data", "logs")
# the files of a case home and of a run folder that the runner writes
RUNNER_FILES = {
    MARKER, "config.toml", "mcp.json", "settings.json", "draft.md", "prompt.txt", "agent.json", "agent.stderr",
    "records.jsonl", "summary.json",
}  # fmt: skip


class SourceChanged(RuntimeError):
    """A file of the source mirror, or a file outside the case mirrors, changed during the run."""


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
    tmp: Path
    draft: Path
    mcp_config: Path
    settings: Path
    as_of: str | None
    index: dict


Runner = Callable[[list[str], str, Path, dict, float], AgentOutput]


# --- the final record --------------------------------------------------------------------------------------------


def parse_record(text: str | None) -> tuple[dict | None, str | None]:
    """The JSON object of the last `KB-WRITE-RECORD` line of text, or None and the reason. The object needs a string
    `action`, a list of strings `targets` and a boolean `written`."""
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
    if not isinstance(record.get("action"), str):
        return None, f"the {PREFIX} has no string action"
    targets = record.get("targets", [])
    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        return None, f"the targets of the {PREFIX} are not a list of note keys"
    if not isinstance(record.get("written"), bool):
        return None, f"the {PREFIX} has no boolean written"
    return record, None


def parse_agent_output(stdout: str) -> dict:
    """The answer, cost, turns, permission denials and session facts of the output of `claude -p`: one JSON object
    (`--output-format json`) or one event per line (`stream-json`), with the result event last."""
    events = []
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    if not events:
        try:
            whole = json.loads(stdout)
            events = [whole] if isinstance(whole, dict) else []
        except ValueError:
            pass
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    result = next((e for e in reversed(events) if e.get("type", "result") == "result"), None)
    session = None
    if init is not None:
        session = {
            "plugins": [p.get("name") for p in init.get("plugins") or [] if isinstance(p, dict)],
            "mcp_servers": [s.get("name") for s in init.get("mcp_servers") or [] if isinstance(s, dict)],
            "tools": init.get("tools"),
            "permission_mode": init.get("permissionMode"),
        }
    if result is None:
        return {"result": None, "cost_usd": None, "turns": None, "is_error": True, "permission_denials": None,
                "session": session, "error": "the agent output has no result"}  # fmt: skip
    return {
        "result": result.get("result"),
        "cost_usd": result.get("total_cost_usd"),
        "turns": result.get("num_turns"),
        "is_error": bool(result.get("is_error")),
        "permission_denials": len(result.get("permission_denials") or []),
        "session": session,
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
    """{path relative to folder: sha256} of all files under folder. A symbolic link (to a file or a folder) gives
    `link:<target>` and is not followed."""
    out = {}
    skip = {p.resolve() for p in exclude}
    for top, dirs, files in os.walk(folder):
        base = Path(top)
        for name in list(dirs):
            if (base / name).is_symlink():
                out[(base / name).relative_to(folder).as_posix()] = "link:" + os.readlink(base / name)
        dirs[:] = [d for d in dirs if not (base / d).is_symlink() and (base / d).resolve() not in skip]
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


def changed_since(folder: Path, start_ns: int, skip_dirs: set[Path], skip_files: set[Path]) -> list[str]:
    """The files under folder (absolute paths) with a modification time at or after start_ns. A skipped folder is
    not entered. A symbolic link counts by its own time."""
    found = []
    if not folder.exists():
        return found
    for top, dirs, files in os.walk(folder):
        base = Path(top)
        dirs[:] = [d for d in dirs if base / d not in skip_dirs]
        for name in [*files, *(d for d in dirs if (base / d).is_symlink())]:
            path = base / name
            if path in skip_files:
                continue
            try:
                if path.lstat().st_mtime_ns >= start_ns:
                    found.append(str(path))
            except FileNotFoundError:
                continue
    return sorted(found)


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


def helper_script(plugin_root: Path) -> Path:
    return plugin_root / "scripts" / "kb-write.sh"


def case_settings(mirror: Path, tmp: Path, plugin_root: Path, guards: tuple[Path, ...] = ()) -> dict:
    """The only settings of a case agent: permission rules for the exact helper and shell commands, Write and Edit
    under the mirror, and the Bash sandbox with writes only to the mirror and the case TMPDIR. A guard folder that
    holds neither the case home nor the plugin folder is also closed to reads."""
    script = helper_script(plugin_root)
    bash = [
        f"Bash(bash {script} bench:*)",
        f'Bash(bash "{script}" bench:*)',
        f"Bash(KB_WRITE_WITHIN={mirror} bash {script}:*)",
        f'Bash(KB_WRITE_WITHIN="{mirror}" bash "{script}":*)',
        *(f"Bash({c}:*)" for c in SHELL_COMMANDS),
    ]
    closed = [g for g in guards if not mirror.is_relative_to(g) and not plugin_root.is_relative_to(g)]
    # a path rule with a leading // is an absolute path
    return {
        "permissions": {
            "allow": [
                "Read",
                "Glob",
                "Grep",
                "ToolSearch",
                f"mcp__{SERVER}",
                *bash,
                f"Write(/{mirror}/**)",
                f"Edit(/{mirror}/**)",
            ],  # fmt: skip
            "deny": [f"Read(/{g}/**)" for g in closed],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": False,
            "filesystem": {"allowWrite": [str(mirror), str(tmp)], "denyRead": [str(g) for g in closed]},
        },
    }


def build_case_home(
    case: dict, cfg: Config, run_dir: Path, form: str, plugin_root: Path, guards: tuple[Path, ...] = ()
) -> CaseHome:
    """The home of one case under run_dir: the start mirror, its index, config, marker, draft, MCP config, agent
    settings and TMPDIR."""
    from .index import connect, update

    if cfg.vault_paths:
        raise ValueError("a replay run needs all vaults under root (no vault_paths)")
    home = run_dir / case["id"]
    mirror = home / "mirror"
    draft_text = case["drafts"][form]
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
    tmp = home / "tmp"
    tmp.mkdir()
    draft = home / "draft.md"
    draft.write_text(draft_text)
    mcp = home / "mcp.json"
    server = {"command": mcp_command()[0], "args": mcp_command()[1:], "env": {"RAGNAROK_HOME": str(home)}}
    mcp.write_text(json.dumps({"mcpServers": {SERVER: server}}, indent=2) + "\n")
    settings = home / "settings.json"
    settings.write_text(json.dumps(case_settings(mirror, tmp, plugin_root, guards), indent=2) + "\n")

    newer = cfg.note_path(case["newer"]) if case.get("newer") else None
    as_of = note_as_of(parse_note(newer.read_text(errors="replace"))[0]) if newer and newer.is_file() else None

    case_cfg = load_config(home / "config.toml")
    conn = connect(case_cfg)
    try:
        result = update(case_cfg, conn, CacheOnlyEmbedder())
    finally:
        conn.close()
    index = {k: v for k, v in result.__dict__.items() if isinstance(v, int | float)}
    return CaseHome(home, mirror, tmp, draft, mcp, settings, as_of, index)


# --- the agent ---------------------------------------------------------------------------------------------------


def agent_command(claude: str, case_home: CaseHome, *, budget: float | None, model: str | None) -> list[str]:
    """`claude -p` with no setting source but the case settings, only the case MCP server and the built-in tools
    that the skill needs."""
    for path in (case_home.mirror, case_home.tmp):
        if any(ch.isspace() or ch in ',"' for ch in str(path)):
            raise ValueError(f"a case path must have no space, comma or quote: {path}")
    argv = [
        claude, "-p", "--output-format", "stream-json", "--verbose",
        "--setting-sources", "", "--settings", str(case_home.settings),
        "--strict-mcp-config", "--mcp-config", str(case_home.mcp_config),
        "--tools", BUILTIN_TOOLS, "--permission-mode", "dontAsk", "--no-session-persistence",
    ]  # fmt: skip
    if model:
        argv += ["--model", model]
    if budget is not None:
        argv += ["--max-budget-usd", str(budget)]
    return argv


def agent_env(case_home: CaseHome) -> dict:
    """A minimal environment: the login and locale variables that the CLI and its auth need, TMPDIR in the case."""
    env = {k: os.environ[k] for k in ENV_KEYS if k in os.environ}
    env.update(TMPDIR=str(case_home.tmp), RAGNAROK_HOME=str(case_home.home))
    return env


def agent_prompt(case_home: CaseHome, skill: Path, plugin_root: Path) -> str:
    date_line = (
        f"- The date of the claims of the candidate text (its `as_of`) is {case_home.as_of}."
        if case_home.as_of
        else "- The candidate text has no known date."
    )
    script = helper_script(plugin_root)
    return f"""Run the kb-write skill in fixture mode, as a replay runner calls it.

- The skill is the file {skill}. Read it in full first and follow it exactly.
- CLAUDE_PLUGIN_ROOT is {plugin_root}. The helper script is {script}.
- mirror={case_home.mirror}
- TMPDIR is {case_home.tmp}.
- The fixture ragnarok MCP server is `{SERVER}`. Its tools have the prefix `mcp__{SERVER}__`
  (for example `mcp__{SERVER}__find_collisions`). Use them where the skill names
  `mcp__plugin_ragnarok_ragnarok__*`. Load them with one ToolSearch `select:` call.
- The candidate text is the content of the file {case_home.draft}. No target path is given.
{date_line}
- Shell commands: run one command in each Bash call, with no `;`, `&&`, `|` or redirection. Write each path in
  full, with no shell variable: `mktemp -d {case_home.tmp}/kb-write.XXXXXX`, `bash {script} bench {case_home.mirror}`,
  `KB_WRITE_WITHIN={case_home.mirror} bash {script} save <backup folder> <note file> ...` and
  `KB_WRITE_WITHIN={case_home.mirror} bash {script} restore <backup folder>`. Only these commands and `test`, `diff`,
  `ls`, `cat` and `date` are allowed.
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


def is_note(path: str) -> bool:
    """A Markdown file in a vault folder, not in a dot folder such as .obsidian."""
    parts = path.split("/")
    return len(parts) >= 2 and path.endswith(".md") and not any(p.startswith(".") for p in parts)


def check_writes(action: str | None, targets: set[str], diff: dict) -> tuple[list[str], list[str]]:
    """The extra and the missing writes of the snapshot diff for an action and its targets.
    new: one created note; supersede: one created note and each target changed; merge: each target changed; any
    other action: no change. Nothing else may change, and nothing may be deleted."""
    created, changed = list(diff["created"]), set(diff["changed"])
    notes = 1 if action in ("new", "supersede") else 0
    must_change = targets if action in ("supersede", "merge") else set()
    extra = [f"deleted {p}" for p in diff["deleted"]]
    extra += [f"changed {p}" for p in sorted(changed - must_change)]
    extra += [f"created {p}" for p in created if not is_note(p)]
    created_notes = [p for p in created if is_note(p)]
    extra += [f"created {p}" for p in created_notes[notes:]]
    missing = ["no created note"] * (notes - len(created_notes[:notes]))
    missing += [f"unchanged {p}" for p in sorted(must_change - changed)]
    if action in ("supersede", "merge") and not targets:
        missing.append("no target")
    return extra, missing


def score(case: dict, record: dict | None, diff: dict) -> dict:
    expected, expected_targets = expected_of(case)
    action = record.get("action") if record else None
    targets = set(record.get("targets") or []) if record else set()
    action_ok = record is not None and action == expected
    target_ok = record is not None and targets == set(expected_targets)
    extra, missing = check_writes(action, targets, diff) if record else ([], [])
    flag_ok = record is not None and record.get("written") is (action in WRITE_ACTIONS)
    written_ok = record is not None and not extra and not missing and flag_ok
    correct = action_ok and target_ok and written_ok
    merge_for_supersede = expected == "supersede" and action == "merge"
    reasons = []
    if record is not None:
        reasons += [] if action_ok else ["action"]
        reasons += ["missing-targets"] if set(expected_targets) - targets else []
        reasons += ["extra-targets"] if targets - set(expected_targets) else []
        reasons += ["extra-writes"] if extra else []
        reasons += ["missing-writes"] if missing else []
        reasons += [] if flag_ok else ["written-flag"]
    if record is None:
        category = "failed"
    elif action == "refused" or record.get("outcome") == "refused":
        category = "refused"
    elif action == "ask":
        category = "ask"
    elif correct:
        category = "correct"
    elif merge_for_supersede and reasons == ["action"]:
        category = "merge-for-supersede"
    else:
        category = "wrong"
    return {
        "expected_action": expected,
        "expected_targets": expected_targets,
        "action": action,
        "targets": sorted(targets),
        "extra_targets": sorted(targets - set(expected_targets)),
        "extra_writes": extra,
        "missing_writes": missing,
        "action_ok": action_ok,
        "target_ok": target_ok,
        "written_ok": written_ok,
        "correct": correct,
        "merge_for_supersede": merge_for_supersede,
        "reasons": reasons,
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
    if "clean" in summary:
        lines.append(
            "source mirror unchanged, no outside change"
            if summary["clean"]
            else f"OUTSIDE CHANGES: mirror {summary['source_mirror_changes']}, files {summary['outside_changes']}"
        )
    return "\n".join(lines)


# --- the run -----------------------------------------------------------------------------------------------------


def check_bench(home: Path) -> None:
    if not (home / MARKER).is_file():
        raise ValueError(f"{home} holds no {MARKER}: a replay run needs a bench home as RAGNAROK_HOME")


def plugin_root_of(skill: Path) -> Path:
    """The plugin folder of skills/<name>/SKILL.md; it must hold scripts/kb-write.sh."""
    root = skill.parent.parent.parent
    if not helper_script(root).is_file():
        raise ValueError(f"no {helper_script(root)}: --skill must be skills/kb-write/SKILL.md in its plugin folder")
    return root


def _new_row(case: dict, form: str) -> dict:
    return {
        "case": case.get("id"), "kind": case.get("kind"), "positive_kind": case.get("positive_kind"), "form": form,
        "as_of": None, "home": None, "index": None, "record": None, "record_error": None, "diff": EMPTY_DIFF,
        "home_changes": None, "session": None, "exit_code": None, "timed_out": False, "cost_usd": None,
        "turns": None, "duration_s": None, "permission_denials": None, "error": None,
    }  # fmt: skip


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
    guards: tuple[Path, ...] = (),
    keep_home: bool = False,
) -> dict:
    """One case: build its home, run the agent, diff the mirror, score. Any error gives a failed row."""
    row = _new_row(case, form)
    case_home = None
    try:
        plugin_root = plugin_root_of(skill)
        case_home = build_case_home(case, cfg, run_dir, form, plugin_root, guards)
        row.update(as_of=case_home.as_of, home=str(case_home.home), index=case_home.index)
        argv = agent_command(claude, case_home, budget=budget, model=model)
        prompt = agent_prompt(case_home, skill, plugin_root)
        (case_home.home / "prompt.txt").write_text(prompt)
        own = tuple(case_home.home / d for d in CASE_WRITABLE)
        home_before = snapshot(case_home.home, own)
        before = snapshot(case_home.mirror)
        out = runner(argv, prompt, case_home.home, agent_env(case_home), timeout)
        row["diff"] = diff_snapshots(before, snapshot(case_home.mirror))
        home_diff = diff_snapshots(home_before, snapshot(case_home.home, own))
        row["home_changes"] = home_diff if any(home_diff.values()) else None
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
            session=agent["session"],
        )
        record, error = parse_record(agent["result"])
        row["record"], row["record_error"] = record, agent["error"] or error
        if out.timed_out:
            row["error"] = f"the agent passed the time limit of {timeout:g} s"
        row["score"] = score(case, record, row["diff"])
    except Exception as e:  # one case must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
        row["score"] = score(case, None, EMPTY_DIFF) if "kind" in case else {"correct": False, "category": "failed"}
    finally:
        if case_home is not None and not keep_home:
            shutil.rmtree(case_home.home / "data", ignore_errors=True)
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
    guards: tuple[Path, ...] = (),
    keep_homes: bool = False,
    runner: Runner | None = None,
) -> tuple[list[dict], dict, Path]:
    """Run each case; write records.jsonl and summary.json into the run folder. Raise SourceChanged (after the
    files are written) when the source mirror changed or a file changed outside the case mirrors."""
    runner = runner or subprocess_runner
    check_bench(home)
    if parallel < 1:
        raise ValueError("--parallel must be 1 or more")
    if timeout <= 0:
        raise ValueError("--timeout must be more than 0")
    if budget is not None and budget <= 0:
        raise ValueError("--max-budget-usd must be more than 0")
    ids = [c["id"] for c in cases]
    if repeated := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"a case id is repeated: {', '.join(repeated)}")
    plugin_root = plugin_root_of(skill)
    guards = tuple(g.resolve() for g in guards)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    run_dir = (out or set_dir / "runs" / run_id).resolve()
    if run_dir.is_relative_to(cfg.root.resolve()):
        raise ValueError(f"the run folder {run_dir} is inside the source mirror")

    start_ns = time.time_ns() - 1_000_000_000  # a margin for the time resolution of the file system
    mirror_before = snapshot(cfg.root)
    run_dir.mkdir(parents=True)
    lock = threading.Lock()
    records = run_dir / "records.jsonl"
    rows: list[dict] = []

    def one(case: dict) -> dict:
        row = run_case(case, cfg, run_dir, form=form, skill=skill, claude=claude, model=model, budget=budget,
                       timeout=timeout, runner=runner, guards=guards, keep_home=keep_homes)  # fmt: skip
        with lock, records.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            rows = list(pool.map(one, cases))
    finally:
        mirror_changes = diff_snapshots(mirror_before, snapshot(cfg.root))
        mirror_changes = mirror_changes if any(mirror_changes.values()) else {}
        homes = [run_dir / i for i in ids]
        skip_dirs = {h / d for h in homes for d in CASE_WRITABLE}
        skip_files = {h / f for h in homes for f in RUNNER_FILES} | {run_dir / f for f in RUNNER_FILES}
        outside = []
        for root in (home.resolve(), plugin_root.resolve(), *guards):
            outside += changed_since(root, start_ns, skip_dirs, skip_files)
        outside += [f"{r['home']}: {r['home_changes']}" for r in rows if r.get("home_changes")]
        outside = sorted(set(outside))
        summary = {
            **summarize(rows),
            "clean": not mirror_changes and not outside,
            "source_mirror_changes": mirror_changes,
            "outside_changes": outside,
            "guards": [str(g) for g in guards],
            "failed_cases": [r["case"] for r in rows if r["error"]],
            "run_id": run_dir.name,
            "set": str(set_dir),
            "form": form,
            "cases": ids,
            "skill": str(skill),
            "model": model,
            "max_budget_usd": budget,
            "timeout_s": timeout,
            "commit": code_commit(),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not summary["clean"]:
        raise SourceChanged(
            f"files outside the case mirrors changed: mirror {json.dumps(mirror_changes)}, files {outside}"
            f" (see {run_dir / 'summary.json'})"
        )
    return rows, summary, run_dir
