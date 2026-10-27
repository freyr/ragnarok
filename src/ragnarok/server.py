"""The MCP server on stdio. It logs only to stderr, with the time of each call."""

from __future__ import annotations

import functools
import hashlib
import logging
import threading
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import Config, config_path, load_config
from .embed import EmbedError, OllamaEmbedder
from .index import IndexMismatch, connect
from .search import PrivateNotAllowed
from .tools import Tools

log = logging.getLogger("ragnarok.server")

EXPECTED = (EmbedError, IndexMismatch, PrivateNotAllowed, FileNotFoundError, LookupError)

INSTRUCTIONS = (
    "Search the user's Obsidian vaults ({vaults}). "
    "When a prompt continues or resembles earlier work but does not name the project, call find_projects first, "
    "then list_project on the best match. Use search for facts, open_note to read a note, and find_by_jira for "
    "Jira keys. Results hold note paths; cite them. A result can be unrelated: read the text before you use it."
)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


def _digest(path: Path) -> str | None:
    return hashlib.sha1(path.read_bytes()).hexdigest() if path.exists() else None


SETTLE_NS = 10**9  # a config file changed less than one second ago can still be in the middle of a write


def build(cfg: Config | None = None, tools: Tools | None = None, config_file: Path | None = None) -> MCPServer:
    """The server. With tools or cfg, the configuration is fixed. Else each call first reads the config file
    again when it changed, so that a new private pattern or rule applies to that call without a restart.
    A missing, recent, invalid or vault-less file does not replace the configuration in use. A valid file that
    the index refuses (a new model or dimension) stops the old configuration: each call fails until the rebuild."""
    state: dict = {}
    lock = threading.Lock()  # one call at a time: the scan writes to the index
    watch = None if tools is not None or cfg is not None else (config_file or config_path())

    def open_tools(c: Config) -> Tools:
        return Tools(c, connect(c), OllamaEmbedder(c))

    def keep(reason: str) -> None:
        """Reject the file: keep the configuration in use, or fail again while the old one is stopped."""
        log.warning("config file %s %s; the server keeps its configuration", watch, reason)
        if "stopped" in state:
            raise IndexMismatch(state["stopped"])

    def reload(stamp: tuple[int, int] | None) -> None:
        stopped = "stopped" in state
        if stamp is None:
            return keep("is missing")
        if time.time_ns() - stamp[0] < SETTLE_NS:
            return keep("changed less than a second ago")
        digest = _digest(watch)
        if not stopped:
            state["stamp"] = stamp
            if digest == state["digest"]:
                return
        try:
            new = load_config(watch)
            if not new.vaults:
                raise ValueError("no vaults")
        except Exception as e:
            return keep(f"is not usable ({e})")
        try:
            new_tools = open_tools(new)
        except IndexMismatch as e:
            # a valid config that the index refuses: stop the old config; each call tries again until the rebuild
            log.warning("config file %s needs a new index; the calls fail until `ragnarok index --rebuild`", watch)
            if "tools" in state:
                state.pop("tools").conn.close()
            state["stopped"] = str(e)
            raise
        if "tools" in state:
            state.pop("tools").conn.close()
        state.pop("stopped", None)
        state["tools"], state["stamp"], state["digest"] = new_tools, stamp, digest
        log.info("config changed: %s", watch)

    def get() -> Tools:
        if tools is not None:
            state["tools"] = tools
        elif watch is None:
            if "tools" not in state:
                state["tools"] = open_tools(cfg)
        elif "tools" not in state and "stopped" not in state:
            stamp, digest = _stamp(watch), _digest(watch)
            state["tools"] = open_tools(load_config(watch))
            state["stamp"], state["digest"] = stamp, digest
        else:
            # while stopped, each call tries again: the rebuild changes the index, not the file
            stamp = _stamp(watch)
            if "stopped" in state or stamp != state["stamp"]:
                reload(stamp)
        return state["tools"]

    def timed(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            ok = "ok"
            try:
                with lock:
                    return fn(*args, **kwargs)
            except EXPECTED as e:
                # the client sees the text of a ToolError only; other exceptions show as a bare failure
                ok = type(e).__name__
                raise ToolError(str(e)) from e
            except Exception as e:
                ok = type(e).__name__
                raise
            finally:
                t = state.get("tools")
                scan = t.last_scan if t else {}
                log.info(
                    "call=%s ms=%d scan_ms=%s scan_changed=%s result=%s",
                    fn.__name__,
                    (time.perf_counter() - start) * 1000,
                    scan.get("ms"),
                    scan.get("changed"),
                    ok,
                )

        return wrapper

    server = MCPServer(
        name="ragnarok", instructions=INSTRUCTIONS.format(vaults=", ".join((cfg or load_config(watch)).vaults))
    )

    @server.tool()
    @timed
    def search(
        query: str,
        vault: str | None = None,
        project: str | None = None,
        area: str | None = None,
        type: str | None = None,
        include_archived: bool = True,
        include_private: bool = False,
        limit: int = 8,
    ) -> dict:
        """Hybrid search (meaning and keywords) over the vault notes. Returns notes, each with its best section.

        Filters: vault (the name of a vault folder), project (folder slug), area, type (research,
        spec, plan, summary, drivers, guide, note, ...). include_private works only when the user allows it.
        """
        return get().search(
            query,
            vault=vault,
            project=project,
            area=area,
            type=type,
            include_archived=include_archived,
            include_private=include_private,
            limit=limit,
        )

    @server.tool()
    @timed
    def find_projects(query: str, vault: str | None = None, limit: int = 5) -> dict:
        """Find the earlier projects that a prompt continues or resembles, ranked by their matching notes.

        Each project has its slug, vault, folder, description, stage, next action and matching notes.
        """
        return get().find_projects(query, vault=vault, limit=limit)

    @server.tool()
    @timed
    def list_project(project: str, vault: str | None = None) -> dict:
        """All notes of a project in any folder, with type, status and description, plus its stage and next action."""
        return get().list_project(project, vault=vault)

    @server.tool()
    @timed
    def open_note(path: str, heading: str | None = None, outline: bool | None = None) -> dict:
        """The text of a note (or of one section, by heading), its frontmatter, links and backlinks.

        outline=true gives the sections (heading path, words, first sentence) instead of the text. A note of more
        than 4000 words gives the outline unless outline=false; then open the section you need by heading, as the
        last heading or the full path such as "Results > Inferences".
        Credentials in the text are replaced with [REDACTED]. Private notes are refused.
        """
        return get().open_note(path, heading=heading, outline=outline)

    @server.tool()
    @timed
    def find_by_jira(key: str) -> dict:
        """The notes that contain a Jira key, for example ABC-123."""
        return get().find_by_jira(key)

    return server
