"""The MCP server on stdio. It logs only to stderr, with the time of each call."""

from __future__ import annotations

import functools
import logging
import threading
import time

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .config import Config, load_config
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


def build(cfg: Config | None = None, tools: Tools | None = None) -> MCPServer:
    state: dict[str, Tools] = {}
    lock = threading.Lock()  # one call at a time: the scan writes to the index

    def get() -> Tools:
        if "tools" not in state:
            if tools is not None:
                state["tools"] = tools
            else:
                c = cfg or load_config()
                state["tools"] = Tools(c, connect(c), OllamaEmbedder(c))
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
        name="ragnarok", instructions=INSTRUCTIONS.format(vaults=", ".join((cfg or load_config()).vaults))
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
    def open_note(path: str, heading: str | None = None) -> dict:
        """The text of a note (or of one section, by heading), its frontmatter, links and backlinks.

        Credentials in the text are replaced with [REDACTED]. Private notes are refused.
        """
        return get().open_note(path, heading=heading)

    @server.tool()
    @timed
    def find_by_jira(key: str) -> dict:
        """The notes that contain a Jira key, for example ABC-123."""
        return get().find_by_jira(key)

    return server
