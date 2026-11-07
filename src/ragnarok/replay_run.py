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
the case MCP server, the built-in tools that the skill needs, permission rules for the exact helper commands, Read
only in the case home and the plugin folder, Write and Edit only under the case mirror, and the Bash sandbox with
writes only to the case mirror and the case TMPDIR. The Bash sandbox cannot read the siblings of the case home, of
the plugin folder and of their parent folders under the user's home. The source mirror and index, the answer key of
the set (`cases.jsonl`, `manifest.json`, score files), the other runs, sets and case homes, and the guard folders are
also closed by rule. It runs in the case home with a minimal environment (plus the auth variables that are set), a
cost limit and a time limit. The plugin is not loaded: its own MCP config points to the real vaults; the agent reads
the skill file.

The fatal checks of a run: the sha256 snapshot of the source mirror before and after the run, the snapshot of the
runner's files of each case home before and after its agent, and the audit of each agent's transcript (the init
event shows only the given session; each Write and Edit is under the case mirror; each Read, Glob and Grep is under
the case home or the plugin folder; each Bash command has an allowed shape and names only paths under the case home
or the plugin folder). A file with a new time under the bench home, a guard folder, or a change in the plugin folder
is only a warning, because another session can write them.

The scorer compares the action and the targets of the agent's final record (`KB-WRITE-RECORD`) with the expected
values of the case, checks that the snapshot diff of the case mirror holds exactly the files of that action, and
checks the content of a supersede or a merge: the `supersedes`, `superseded_by` and `validity` keys of each note in
the direction that the skill defines, the expected merge target, and the unchanged body of each superseded note. The
skill can rewrite the draft text, so the text is not compared. A created note with `type: entity` (the skill makes
one when no entity exists for the main topic of the draft) is no extra write of any action; the score lists it in
`entity_notes`. The `open_conflicts` of the record are reported in the score and do not change it. The summary
counts both.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
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
# the auth and provider variables of Claude Code, passed through when they are set
AUTH_ENV_KEYS = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_BEDROCK_BASE_URL", "AWS_REGION", "AWS_PROFILE", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX", "ANTHROPIC_VERTEX_BASE_URL", "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION",
    "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT",
)  # fmt: skip
# the folders of a case home that the agent (mirror, tmp) or the MCP server (data, logs) writes
CASE_WRITABLE = ("mirror", "tmp", "data", "logs")
# the files of a case home and of a run folder that the runner writes
RUNNER_FILES = {
    MARKER, "config.toml", "mcp.json", "settings.json", "draft.md", "prompt.txt", "agent.json", "agent.stderr",
    "records.jsonl", "summary.json",
}  # fmt: skip


class SourceChanged(RuntimeError):
    """The run is not clean: a file of the source mirror or a runner file of a case home changed, or the audit of an
    agent found a violation."""


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
    closed: tuple[Path, ...] = ()


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


def agent_events(stdout: str) -> list[dict]:
    """The JSON objects of the output of `claude -p`: one per line (`stream-json`), or one object (`json`)."""
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
    return events


def parse_agent_output(stdout: str) -> dict:
    """The answer, cost, turns, permission denials and session facts of the output of `claude -p`: one JSON object
    (`--output-format json`) or one event per line (`stream-json`), with the result event last."""
    events = agent_events(stdout)
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    result = next((e for e in reversed(events) if e.get("type", "result") == "result"), None)
    session = None
    if init is not None:
        session = {
            "plugins": [p.get("source") or p.get("name") for p in init.get("plugins") or [] if isinstance(p, dict)],
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


def closed_reads(cfg: Config, home: Path, case_dir: Path, case_ids: list[str]) -> list[Path]:
    """The paths that a case agent must not read: the source mirror and its index, every sibling of the case home
    and of each of its parent folders up to the bench home (the other cases and runs, `cases.jsonl`,
    `manifest.json`, the score files, the other sets, the source config and data), and the files of the run folder
    that the runner writes later."""
    case_dir, top = case_dir.resolve(), home.resolve()
    run_dir = case_dir.parent
    closed = {cfg.root.resolve()}
    data = cfg.index_path.parent.resolve()
    if case_dir.is_relative_to(data):
        closed |= {cfg.index_path.resolve(), cfg.index_path.resolve().with_name(cfg.index_path.name + "-wal")}
    else:
        closed.add(data)
    closed |= {run_dir / i for i in case_ids if i != case_dir.name}
    closed |= {run_dir / "records.jsonl", run_dir / "summary.json"}
    node = case_dir
    while node != top and node.is_relative_to(top) and node.parent != node:
        closed |= {c for c in node.parent.iterdir() if c != node}
        node = node.parent
    return sorted(p for p in closed if not case_dir.is_relative_to(p))


def closed_around(keep_open: list[Path], top: Path, extra_open: list[Path] | tuple = ()) -> list[Path]:
    """The paths under top that a Bash command of the agent must not read: each sibling of a keep_open path and of
    each of its parent folders up to top. A path inside an extra_open path (for example the Claude Code config
    folder, which holds the shell snapshot of the Bash tool) also stays open. Empty when no keep_open path is under
    top."""
    top = top.resolve()
    keep = [k.resolve() for k in keep_open if k.resolve().is_relative_to(top)]
    if not keep:
        return []
    keep += [e.resolve() for e in extra_open if e.resolve().is_relative_to(top)]
    chain = {a for k in keep for a in (k, *k.parents) if a.is_relative_to(top)}
    closed = []
    for folder in sorted(chain):
        if any(folder.is_relative_to(k) for k in keep):
            continue
        try:
            children = list(folder.iterdir())
        except OSError:
            continue
        closed += [c for c in children if c not in chain and not any(c.is_relative_to(k) for k in keep)]
    return sorted(closed)


def case_settings(
    mirror: Path,
    tmp: Path,
    plugin_root: Path,
    guards: tuple[Path, ...] = (),
    closed: list[Path] | tuple = (),
    outside: list[Path] | tuple = (),
) -> dict:
    """The only settings of a case agent: permission rules for the exact helper and shell commands, Read only in
    the case home and the plugin folder (in dontAsk mode any other read is denied; a Read rule also covers Glob and
    Grep), Write and Edit under the mirror, and the Bash sandbox with writes only to the mirror and the case TMPDIR.
    The closed paths, and each guard folder that holds neither the case home nor the plugin folder, are also denied
    by rule. The Bash sandbox cannot read the closed paths, the guards and the outside paths (see closed_around)."""
    home = mirror.parent
    script = helper_script(plugin_root)
    bash = [
        f"Bash(bash {script} bench:*)",
        f'Bash(bash "{script}" bench:*)',
        f"Bash(KB_WRITE_WITHIN={mirror} bash {script}:*)",
        f'Bash(KB_WRITE_WITHIN="{mirror}" bash "{script}":*)',
        *(f"Bash({c}:*)" for c in SHELL_COMMANDS),
    ]
    guards_closed = [g for g in guards if not mirror.is_relative_to(g) and not plugin_root.is_relative_to(g)]
    no_read = [*closed, *guards_closed]
    # a path rule with a leading // is an absolute path; a closed path can be a file or a folder
    return {
        "permissions": {
            "allow": [
                f"Read(/{home}/**)",
                f"Read(/{plugin_root}/**)",
                "ToolSearch",
                f"mcp__{SERVER}",
                *bash,
                f"Write(/{mirror}/**)",
                f"Edit(/{mirror}/**)",
            ],  # fmt: skip
            "deny": [rule for p in no_read for rule in (f"Read(/{p})", f"Read(/{p}/**)")],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": False,
            "allowUnsandboxedCommands": False,
            "filesystem": {"allowWrite": [str(mirror), str(tmp)], "denyRead": [str(p) for p in [*no_read, *outside]]},
        },
    }


def build_case_home(
    case: dict,
    cfg: Config,
    run_dir: Path,
    form: str,
    plugin_root: Path,
    guards: tuple[Path, ...] = (),
    *,
    bench_home: Path | None = None,
    case_ids: list[str] | None = None,
    read_root: Path | None = None,
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
    closed = closed_reads(cfg, bench_home or run_dir, home, case_ids or [case["id"]])
    config_dir = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    outside = closed_around([home, plugin_root], read_root, [config_dir]) if read_root else []
    settings.write_text(json.dumps(case_settings(mirror, tmp, plugin_root, guards, closed, outside), indent=2) + "\n")

    newer = cfg.note_path(case["newer"]) if case.get("newer") else None
    as_of = note_as_of(parse_note(newer.read_text(errors="replace"))[0]) if newer and newer.is_file() else None

    case_cfg = load_config(home / "config.toml")
    conn = connect(case_cfg)
    try:
        result = update(case_cfg, conn, CacheOnlyEmbedder())
    finally:
        conn.close()
    index = {k: v for k, v in result.__dict__.items() if isinstance(v, int | float)}
    return CaseHome(home, mirror, tmp, draft, mcp, settings, as_of, index, tuple(closed))


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
    """A minimal environment: the login, locale and auth variables that the CLI needs, and TMPDIR in the case."""
    env = {k: os.environ[k] for k in (*ENV_KEYS, *AUTH_ENV_KEYS) if k in os.environ}
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


# --- the audit of the agent ---------------------------------------------------------------------------------------

WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
READ_TOOLS = ("Read", "Glob", "Grep")
SHELL_OPERATORS = set("();<>|&")


def _norm(path: str, cwd: Path) -> Path:
    return Path(os.path.normpath(path if path.startswith("/") else cwd / path))


def session_errors(session: dict | None) -> list[str]:
    """The differences of the init event of the agent from the session that the runner gave: only the built-in
    `cc-plugin-*` plugins, only the case MCP server and its tools, the built-in tools of the case, `dontAsk`."""
    if session is None:
        return ["the agent output has no init event"]
    errors = []
    for plugin in session.get("plugins") or []:
        if not (str(plugin).startswith("cc-plugin-") and str(plugin).endswith("@builtin")):
            errors.append(f"the plugin {plugin} is not a built-in plugin")
    if session.get("mcp_servers") != [SERVER]:
        errors.append(f"the MCP servers are {session.get('mcp_servers')}, not [{SERVER}]")
    if session.get("permission_mode") != "dontAsk":
        errors.append(f"the permission mode is {session.get('permission_mode')}, not dontAsk")
    tools = [str(t) for t in session.get("tools") or []]
    builtin = sorted(t for t in tools if not t.startswith("mcp__"))
    if builtin != sorted(BUILTIN_TOOLS.split(",")):
        errors.append(f"the built-in tools are {builtin}, not {BUILTIN_TOOLS}")
    other = sorted(t for t in tools if t.startswith("mcp__") and not t.startswith(f"mcp__{SERVER}__"))
    if other:
        errors.append(f"tools of another MCP server: {other}")
    return errors


def bash_errors(command: str, case_home: CaseHome, plugin_root: Path) -> list[str]:
    """The differences of a Bash command from the allowed shapes: `bash <script> bench ...`,
    `KB_WRITE_WITHIN=<mirror> bash <script> ...` or one of the shell commands, with no operator or expansion, and
    each path under the case home or the plugin folder (or /dev/null)."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return [f"the command cannot be parsed: {command}"]
    if not words:
        return ["an empty command"]
    if "\n" in command or any(set(w) <= SHELL_OPERATORS or "$" in w or "`" in w for w in words):
        return [f"a compound command or an expansion: {command}"]
    script = str(helper_script(plugin_root))
    if words[0].startswith("KB_WRITE_WITHIN="):
        if _norm(words[0].split("=", 1)[1], case_home.home) != case_home.mirror:
            return [f"KB_WRITE_WITHIN is not the case mirror: {command}"]
        words = words[1:]
        if words[:2] != ["bash", script] or len(words) < 3:
            return [f"not an allowed command: {command}"]
    elif words[:2] == ["bash", script]:
        if words[2:3] != ["bench"]:
            return [f"the helper runs without KB_WRITE_WITHIN: {command}"]
    elif words[0] not in SHELL_COMMANDS:
        return [f"not an allowed command: {command}"]
    errors = []
    for word in words[1:]:
        value = word.split("=", 1)[1] if word.startswith("-") and "=" in word else word
        if value.startswith("~"):
            errors.append(f"a path with ~: {command}")
        elif ("/" in value or value in (".", "..")) and value != "/dev/null":
            path = _norm(value, case_home.home)
            if not (path.is_relative_to(case_home.home) or path.is_relative_to(plugin_root)):
                errors.append(f"the path {value} is outside the case home: {command}")
    return errors


def audit_agent(stdout: str, case_home: CaseHome, plugin_root: Path) -> dict:
    """The check of the transcript of the agent: the session of the init event, and each tool call that was not
    denied. A Write or Edit outside the case mirror, a Bash command of another shape or with a path outside the case
    home, a read outside the case home and the plugin folder and a tool that the runner did not give are
    violations."""
    events = agent_events(stdout)
    if not events:
        return {"tool_calls": 0, "denied": 0, "violations": []}
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None)
    violations = session_errors(parse_agent_output(stdout)["session"] if init else None)
    calls = [
        block
        for e in events
        if e.get("type") == "assistant"
        for block in (e.get("message") or {}).get("content") or []
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    denied = {
        d.get("tool_use_id")
        for e in events
        if e.get("type") == "result"
        for d in e.get("permission_denials") or []
        if isinstance(d, dict)
    }
    allowed_tools = set(BUILTIN_TOOLS.split(","))
    home = case_home.home
    for call in calls:
        if call.get("id") in denied:
            continue
        name, args = str(call.get("name")), call.get("input") or {}
        if name in WRITE_TOOLS:
            path = args.get("file_path") or args.get("notebook_path") or ""
            if not path or not _norm(str(path), home).is_relative_to(case_home.mirror):
                violations.append(f"{name} outside the case mirror: {path}")
        elif name == "Bash":
            violations += bash_errors(str(args.get("command") or ""), case_home, plugin_root)
        elif name in READ_TOOLS:
            values = [args.get("file_path"), args.get("path")]
            if name == "Glob" and str(args.get("pattern") or "").startswith("/"):
                values.append(args.get("pattern"))
            for value in values:
                if not isinstance(value, str) or not value:
                    continue
                path = _norm(re.split(r"[*?\[{]", value, maxsplit=1)[0] or "/", home)
                if any(path.is_relative_to(c) for c in case_home.closed):
                    violations.append(f"{name} of a closed path: {value}")
                elif not (path.is_relative_to(home) or path.is_relative_to(plugin_root)):
                    violations.append(f"{name} outside the case home and the plugin folder: {value}")
        elif name not in allowed_tools and not name.startswith(f"mcp__{SERVER}__"):
            violations.append(f"a tool that the runner did not give: {name}")
    return {"tool_calls": len(calls), "denied": len(denied & {c.get("id") for c in calls}), "violations": violations}


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


@dataclass
class NoteTexts:
    """The text of a note key before the agent (the start state of the case) and after it (the case mirror), or None
    when the note does not exist; and the date of the claims of the candidate text."""

    before: Callable[[str], str | None]
    after: Callable[[str], str | None]
    as_of: str | None = None


def mirror_texts(cfg: Config, case: dict, mirror: Path, as_of: str | None) -> NoteTexts:
    """The note texts of a case: before from the source mirror in the start state, after from the case mirror."""
    state = case.get("start_state") or {}

    def before(key: str) -> str | None:
        path = cfg.root / key
        if key == state.get("exclude") or not path.is_file():
            return None
        return strip_keys(path.read_text(errors="replace"), (state.get("remove_keys") or {}).get(key) or [])

    def after(key: str) -> str | None:
        path = mirror / key
        return path.read_text(errors="replace") if path.is_file() else None

    return NoteTexts(before, after, as_of)


def link_names(value) -> list[str]:
    """The targets of the wiki-links of a frontmatter value (a string or a list); a plain string is a path."""
    items = value if isinstance(value, list) else [value] if value else []
    names = []
    for item in items:
        found = re.findall(r"\[\[([^\]|#]+)", str(item))
        names += [f.strip() for f in found] if found else [str(item).strip()]
    return [n for n in names if n]


def links_to(names: list[str], key: str) -> bool:
    """One of the link targets resolves to the note key: the same path, or a path suffix (a basename link)."""
    stem = key.removesuffix(".md")
    for name in names:
        name = name.removesuffix(".md").strip("/")
        if stem == name or stem.endswith("/" + name):
            return True
    return False


def _plain(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _replaced_errors(key: str, fm: dict, by: str, by_fm: dict, notes: NoteTexts) -> list[tuple[str, str]]:
    """A note that the run replaces by the note `by`: `superseded_by` to it, `validity: superseded` when the note
    has a validity or is a state or a proposal, and the same body (apart from white space at line ends)."""
    errors = []
    old, new = notes.before(key), notes.after(key)
    if old is not None and new is not None and _plain(parse_note(old)[1]) != _plain(parse_note(new)[1]):
        errors.append(("old-body-changed", f"the body of {key} changed"))
    if not links_to(link_names(fm.get("superseded_by")), by):
        errors.append(("missing-superseded-by", f"{key} has no superseded_by {by}"))
    if links_to(link_names(fm.get("supersedes")), by) or links_to(link_names(by_fm.get("superseded_by")), key):
        errors.append(("supersede-direction", f"{key} supersedes {by}, not the reverse"))
    if ("validity" in fm or fm.get("class") in ("state", "proposal")) and fm.get("validity") != "superseded":
        errors.append(("validity", f"{key} has validity {fm.get('validity')}, not superseded"))
    return errors


def check_effect(
    action: str | None, targets: set[str], diff: dict, notes: NoteTexts, merge_targets: tuple[str, ...] = ()
) -> list[tuple[str, str]]:
    """The (reason, detail) errors of the content of the writes of a supersede or a merge (step 5 of the skill).
    supersede: the created note lists each target in `supersedes` and has no `superseded_by`; each target has
    `superseded_by` the created note. merge: one target (the merge target) has no `superseded_by` and lists each
    other target in `supersedes`; each other target has `superseded_by` the merge target; a state merge target has
    the `as_of` of the candidate text, and is the expected merge target (one of merge_targets, the old notes that
    the case expects to absorb the draft; when there is none, any target). The body of a superseded note does not
    change. A changed note that differs only in white space is a no-op edit."""
    errors: list[tuple[str, str]] = []
    for key in diff["changed"]:
        old, new = notes.before(key), notes.after(key)
        if old is not None and new is not None and _plain(old) == _plain(new):
            errors.append(("no-op-edit", f"{key} differs only in white space"))

    def fm(key: str) -> dict:
        text = notes.after(key)
        return parse_note(text)[0] if text is not None else {}

    if action == "supersede":
        created = [p for p in diff["created"] if is_note(p)]
        if len(created) != 1:
            return errors  # check_writes reports it
        new = created[0]
        new_fm = fm(new)
        if new_fm.get("superseded_by"):
            errors.append(("supersede-direction", f"the created note {new} has superseded_by"))
        listed = link_names(new_fm.get("supersedes"))
        for key in sorted(targets):
            if not links_to(listed, key):
                errors.append(("missing-supersedes", f"{new} does not list {key} in supersedes"))
            errors += _replaced_errors(key, fm(key), new, new_fm, notes)
    elif action == "merge":
        heads = sorted(t for t in targets if not fm(t).get("superseded_by"))
        if len(heads) != 1:
            return [*errors, ("merge-target", f"{len(heads)} targets have no superseded_by; a merge has one target")]
        head = heads[0]
        if merge_targets and head not in merge_targets:
            errors.append(("merge-target", f"the merge target is {head}, not {' or '.join(merge_targets)}"))
        head_fm = fm(head)
        listed = link_names(head_fm.get("supersedes"))
        for key in sorted(targets - {head}):
            if not links_to(listed, key):
                errors.append(("missing-supersedes", f"{head} does not list {key} in supersedes"))
            errors += _replaced_errors(key, fm(key), head, head_fm, notes)
        head_as_of = note_as_of({"as_of": head_fm.get("as_of")})
        if head_fm.get("class") == "state" and notes.as_of and head_as_of != notes.as_of:
            errors.append(("as-of", f"the merge target {head} has as_of {head_fm.get('as_of')}, not {notes.as_of}"))
    return errors


def merge_target_of(action: str | None, targets: set[str], notes: NoteTexts | None) -> str | None:
    """The merge target of a merge: the one target with no `superseded_by` after the run, or None."""
    if action != "merge" or notes is None:
        return None
    heads = [t for t in sorted(targets) if (text := notes.after(t)) is not None
             and not parse_note(text)[0].get("superseded_by")]  # fmt: skip
    return heads[0] if len(heads) == 1 else None


def entity_notes_of(action: str | None, diff: dict, notes: NoteTexts | None) -> list[str]:
    """The created notes with `type: entity` that the skill adds for the main topic of the draft (at most one in a
    run, but all are listed); they are no extra write of any action. The created note of a new or a supersede comes
    first from the other created notes: when no other note is created, the first entity note is that note. Empty
    without the note texts."""
    if notes is None:
        return []
    created = [p for p in diff["created"] if is_note(p)]
    entities = []
    for path in created:
        text = notes.after(path)
        kind = parse_note(text)[0].get("type") if text is not None else None
        if str(kind or "").strip().lower() == "entity":
            entities.append(path)
    need = 1 if action in ("new", "supersede") else 0
    return sorted(entities[max(0, need - (len(created) - len(entities))) :])


def open_conflicts_of(record: dict | None) -> list[str]:
    """The paths of the `open_conflicts` of a record (objects with a `path`, or paths); a value of another shape
    gives none. They do not change the score."""
    value = (record or {}).get("open_conflicts")
    if not isinstance(value, list):
        return []
    paths = [item.get("path") if isinstance(item, dict) else item for item in value]
    return sorted(str(p) for p in paths if isinstance(p, str) and p)


def score(case: dict, record: dict | None, diff: dict, notes: NoteTexts | None = None) -> dict:
    """The score of one case. A supersede or a merge needs the note texts (notes) for the check of its content;
    without them it is not correct (`effect-unchecked`). A created entity note (`entity_notes_of`) is left out of
    the check of the writes; with one, the written flag must be true. The `open_conflicts` of the record are only
    reported."""
    expected, expected_targets = expected_of(case)
    action = record.get("action") if record else None
    targets = set(record.get("targets") or []) if record else set()
    action_ok = record is not None and action == expected
    target_ok = record is not None and targets == set(expected_targets)
    entities = entity_notes_of(action, diff, notes) if record else []
    diff = {**diff, "created": [p for p in diff["created"] if p not in entities]}
    extra, missing = check_writes(action, targets, diff) if record else ([], [])
    flag_ok = record is not None and record.get("written") is (action in WRITE_ACTIONS or bool(entities))
    written_ok = record is not None and not extra and not missing and flag_ok
    effect: list[tuple[str, str]] = []
    if record is not None and action in ("supersede", "merge"):
        merge_targets = tuple(o["path"] for o in case.get("old", []) if o.get("expected") == "merge")
        effect = (
            check_effect(action, targets, diff, notes, merge_targets)
            if notes
            else [("effect-unchecked", "no note texts")]
        )
    correct = action_ok and target_ok and written_ok and not effect
    merge_for_supersede = expected == "supersede" and action == "merge"
    reasons = []
    if record is not None:
        reasons += [] if action_ok else ["action"]
        reasons += ["missing-targets"] if set(expected_targets) - targets else []
        reasons += ["extra-targets"] if targets - set(expected_targets) else []
        reasons += ["extra-writes"] if extra else []
        reasons += ["missing-writes"] if missing else []
        reasons += [] if flag_ok else ["written-flag"]
        reasons += list(dict.fromkeys(reason for reason, _ in effect))
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
        "entity_notes": entities,
        "open_conflicts": open_conflicts_of(record),
        "effect_errors": [detail for _, detail in effect],
        "merge_target": merge_target_of(action, targets, notes),
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
    """correct / n with the 95% Wilson interval: all, without ask, by kind and by positive kind; the categories, the
    total of created entity notes and of open conflicts, and the total cost."""
    kinds = sorted({r["kind"] for r in rows})
    positive_kinds = sorted({r["positive_kind"] for r in rows if r["positive_kind"]})
    return {
        "all": _rate(rows),
        "without_ask": _rate([r for r in rows if r["score"]["category"] != "ask"]),
        "by_kind": {k: _rate([r for r in rows if r["kind"] == k]) for k in kinds},
        "by_positive_kind": {k: _rate([r for r in rows if r["positive_kind"] == k]) for k in positive_kinds},
        "categories": dict(Counter(r["score"]["category"] for r in rows)),
        "entity_notes": sum(len(r["score"].get("entity_notes") or []) for r in rows),
        "open_conflicts": sum(len(r["score"].get("open_conflicts") or []) for r in rows),
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
    lines.append(
        f"created entity notes {summary.get('entity_notes', 0)}, open conflicts {summary.get('open_conflicts', 0)}"
    )
    lines.append(f"cost: ${summary['cost_usd']:.2f}")
    if "clean" in summary:
        lines.append(
            "clean: source mirror and case homes unchanged, audit passed"
            if summary["clean"]
            else f"NOT CLEAN: mirror {summary['source_mirror_changes']}, case homes {summary['home_changes']},"
            f" audit {summary['audit_failures']}"
        )
        if summary.get("outside_warnings"):
            lines.append(f"warning: {len(summary['outside_warnings'])} files changed outside the run, see summary.json")
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
        "home_changes": None, "session": None, "audit": None, "exit_code": None, "timed_out": False, "cost_usd": None,
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
    bench_home: Path | None = None,
    case_ids: list[str] | None = None,
    read_root: Path | None = None,
) -> dict:
    """One case: build its home, run the agent, diff the mirror, audit the transcript, score. Any error gives a
    failed row; an audit violation fails the case."""
    row = _new_row(case, form)
    case_home = None
    try:
        plugin_root = plugin_root_of(skill)
        case_home = build_case_home(
            case,
            cfg,
            run_dir,
            form,
            plugin_root,
            guards,
            bench_home=bench_home,
            case_ids=case_ids,
            read_root=read_root,
        )
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
        row["audit"] = audit_agent(out.stdout, case_home, plugin_root)
        notes = mirror_texts(cfg, case, case_home.mirror, case_home.as_of)
        row["score"] = score(case, record, row["diff"], notes)
        if row["audit"]["violations"]:
            reasons = [*row["score"]["reasons"], "audit"]
            row["score"] = {**row["score"], "correct": False, "category": "failed", "reasons": reasons}
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
    read_root: Path | None = None,
) -> tuple[list[dict], dict, Path]:
    """Run each case; write records.jsonl and summary.json into the run folder. Raise SourceChanged (after the
    files are written) when the source mirror changed, a runner file of a case home changed, or the audit of an
    agent found a violation. A new file time under the bench home, the plugin folder or a guard is a warning."""
    runner = runner or subprocess_runner
    read_root = read_root or Path.home()  # the Bash sandbox closes the paths around the case home under it
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
    plugin_before = snapshot(plugin_root)
    run_dir.mkdir(parents=True)
    lock = threading.Lock()
    records = run_dir / "records.jsonl"
    rows: list[dict] = []

    def one(case: dict) -> dict:
        row = run_case(case, cfg, run_dir, form=form, skill=skill, claude=claude, model=model, budget=budget,
                       timeout=timeout, runner=runner, guards=guards, keep_home=keep_homes, bench_home=home,
                       read_root=read_root, case_ids=ids)  # fmt: skip
        with lock, records.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    try:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            rows = list(pool.map(one, cases))
    finally:
        mirror_changes = diff_snapshots(mirror_before, snapshot(cfg.root))
        mirror_changes = mirror_changes if any(mirror_changes.values()) else {}
        home_changes = {r["case"]: r["home_changes"] for r in rows if r.get("home_changes")}
        audit_failures = {r["case"]: r["audit"]["violations"] for r in rows if (r.get("audit") or {}).get("violations")}
        # warnings: a file with a new time under the bench home, the plugin folder and the guards (another session
        # can write them), and the snapshot diff of the plugin folder (it also shows a deleted file)
        homes = [run_dir / i for i in ids]
        skip_dirs = {h / d for h in homes for d in CASE_WRITABLE}
        skip_dirs |= {r.resolve() for r in home.glob("replay/*/runs/*") if r.resolve() != run_dir}
        skip_files = {h / f for h in homes for f in RUNNER_FILES} | {run_dir / f for f in RUNNER_FILES}
        warnings = []
        for root in (home.resolve(), *guards):
            warnings += changed_since(root, start_ns, skip_dirs, skip_files)
        plugin_changes = diff_snapshots(plugin_before, snapshot(plugin_root))
        warnings += [f"{kind} {plugin_root / p}" for kind, paths in plugin_changes.items() for p in paths]
        summary = {
            **summarize(rows),
            "clean": not mirror_changes and not home_changes and not audit_failures,
            "source_mirror_changes": mirror_changes,
            "home_changes": home_changes,
            "audit_failures": audit_failures,
            "outside_warnings": sorted(set(warnings)),
            "guards": [str(g) for g in guards],
            "failed_cases": [r["case"] for r in rows if r["error"] or r["case"] in audit_failures],
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
            f"the run is not clean: source mirror {json.dumps(mirror_changes)}, case homes {json.dumps(home_changes)},"
            f" audit {json.dumps(audit_failures)} (see {run_dir / 'summary.json'})"
        )
    return rows, summary, run_dir
