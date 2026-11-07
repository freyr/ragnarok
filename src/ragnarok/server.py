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

EXPECTED = (EmbedError, IndexMismatch, PrivateNotAllowed, FileNotFoundError, LookupError, ValueError)

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
        include_superseded: bool = False,
        subject: str | None = None,
        limit: int = 8,
    ) -> dict:
        """Hybrid search (meaning and keywords) over the vault notes. Returns notes, each with its best section.

        Filters: vault (the name of a vault folder), project (folder slug), area, type (research,
        spec, plan, summary, drivers, guide, note, ...). include_private works only when the user allows it.
        A note that has them also shows class (state, proposal, record), validity (current, superseded, open,
        accepted, rejected), as_of, superseded_by (the note that replaces it) and subject (the notes it is about).
        A note with no class gets one from its type. Notes with validity superseded or rejected are left out;
        include_superseded=true shows them, for a question about the past.
        subject (an entity name, alias or path, see find_entities) keeps only the notes about that entity; an
        unknown entity gives no results.
        """
        return get().search(
            query,
            vault=vault,
            project=project,
            area=area,
            type=type,
            include_archived=include_archived,
            include_private=include_private,
            include_superseded=include_superseded,
            subject=subject,
            limit=limit,
        )

    @server.tool()
    @timed
    def find_projects(query: str, vault: str | None = None, limit: int = 5) -> dict:
        """Find the earlier projects that a prompt continues or resembles, ranked by their matching notes.

        Each project has its slug, vault, folder, description, stage, state, next action and matching notes.
        """
        return get().find_projects(query, vault=vault, limit=limit)

    @server.tool()
    @timed
    def list_project(project: str, vault: str | None = None) -> dict:
        """All notes of a project, any folder: type, status, description, class, validity; stage, state, next action."""
        return get().list_project(project, vault=vault)

    @server.tool()
    @timed
    def open_projects(states: list[str] | None = None, vault: str | None = None) -> dict:
        """The projects that are open, grouped by the state in their SUMMARY.md, newest update first.

        states defaults to in-progress and paused; the other values are done, superseded and abandoned. Each project has
        its slug, vault, folder, description, stage, state, gate, updated date and next action.
        """
        return get().open_projects(states=states, vault=vault)

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

    @server.tool()
    @timed
    def find_entities(kind: str | None = None, name: str | None = None) -> dict:
        """The entity notes (type: entity): path, title, kind and aliases.

        kind keeps one kind (device, person, team, ...); name keeps the entities whose title or alias contains it,
        in any case. Use a title or an alias as the subject filter of search.
        """
        return get().find_entities(kind=kind, name=name)

    @server.tool()
    @timed
    def validate(paths: list[str] | None = None, draft: str | None = None, path: str | None = None) -> dict:
        """Check notes, or a draft before a write, against the knowledge-base frontmatter model.

        paths: note paths (as in search results). draft: the frontmatter and the body as one Markdown text; path is
        its intended note path. Errors: wrong values of class, validity, as_of, subject, task, supersedes,
        superseded_by and owner; a missing class, or a missing validity for a state or a proposal; a wiki-link that
        resolves to no note (outside code); a subject or owner that is not an entity note, or a reference target that
        does not exist; a superseded_by target that does not list the note in supersedes; two current state notes with
        one identity key (type, primary subject, task). Warnings: unknown frontmatter keys.
        Each issue has path, check (1 to 7), severity, kind and message.
        """
        return get().validate(paths=paths, draft=draft, path=path)

    @server.tool()
    @timed
    def find_collisions(draft: str | None = None, path: str | None = None, allow_private: bool = False) -> dict:
        """The notes that a draft can collide with, before a write: at most 10 candidates.

        draft: the frontmatter and the body as one Markdown text; path is its intended note path. With no draft, path
        is a note path (as in search results) and that note is not a candidate. The sources (reasons): identity (the
        same type, primary subject and task, any class or validity), search (the hybrid search for the title, the
        description and the start of the body, superseded notes included), links_out (the notes the draft links to),
        links_to_supersedes (each supersedes target and the notes that link to it). Each candidate has path, title,
        type, status, class, validity, as_of, subject, superseded_by, reasons, rank (the search rank or null) and
        distance (the cosine distance of its best chunk to the query, or null). The order: identity candidates, then
        the search rank, then the link candidates by distance. allow_private works only when the user allows it.
        """
        return get().find_collisions(draft=draft, path=path, allow_private=allow_private)

    @server.tool()
    @timed
    def vault_rules() -> dict:
        """The path rules of each vault and the local write rules, before a note is classified and placed.

        Each vault: name, folder, the area and project rules, project markers, archive folder and the private
        patterns that apply to it. write: known_keys (frontmatter keys outside the model that are allowed) and
        systems_of_record (each with data, the kind of data; system, where it lives; skill, the skill that reads it).
        """
        return get().vault_rules()

    return server
