"""The MCP server on stdio. It logs only to stderr, with the time of each call."""

from __future__ import annotations

import functools
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from .config import Config, config_path, load_config
from .embed import EmbedError, OllamaEmbedder
from .index import IndexMismatch, connect
from .search import PrivateNotAllowed
from .tools import Tools
from .trace import purge, safely

log = logging.getLogger("ragnarok.server")

EXPECTED = (EmbedError, IndexMismatch, PrivateNotAllowed, FileNotFoundError, LookupError)

INSTRUCTIONS = (
    "Search the user's Obsidian vaults ({vaults}). "
    "When a prompt continues or resembles earlier work but does not name the project, call find_projects first, "
    "then list_project on the best match. Use search for facts, open_note to read a note, and find_by_jira for "
    "Jira keys. Results hold note paths; cite them. A result can be unrelated: read the text before you use it. "
    "For the knowledge base: resolve the entities first (find_entity, then get_entity and related); a live value "
    "(a cost, a count, a current state) comes from the skill that get_entity names, never from a note; for a "
    "question with one true answer, call search with subject and type. kb_check counts the integrity violations."
)


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


def _ids(ctx: Context | None) -> tuple[str | None, str | None]:
    """The mcp.session.id (the header of an HTTP request) and the gen_ai.conversation.id (a value of the request
    metadata) that the MCP context gives; None for a value it does not give."""
    try:
        request_context = ctx.request_context if ctx is not None else None
    except ValueError:  # a call outside a request, as in a test
        request_context = None
    request = getattr(request_context, "request", None)
    headers = getattr(request, "headers", None)
    session = headers.get("mcp-session-id") if headers is not None else None
    meta = getattr(request_context, "meta", None)
    conversation = (getattr(meta, "model_extra", None) or {}).get("gen_ai.conversation.id")
    return session, conversation


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
        state["tools"].ids = state.get("ids", (None, None))
        return state["tools"]

    def timed(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            ok = "ok"
            try:
                with lock:
                    state["ids"] = _ids(kwargs.pop("ctx", None))
                    if "class" in kwargs:  # the MCP input name of the parameter class_ (a Python keyword)
                        kwargs["class_"] = kwargs.pop("class")
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
    safely("purge", purge, tools.cfg if tools is not None else cfg or load_config(watch))

    @server.tool()
    @timed
    def search(
        query: str,
        vault: str | None = None,
        project: str | None = None,
        area: str | None = None,
        type: str | list[str] | None = None,
        include_archived: bool = True,
        include_private: bool = False,
        collection: str | None = None,
        class_: Annotated[str | None, Field(alias="class")] = None,
        subject: str | list[str] | None = None,
        status: str | list[str] | None = None,
        limit: int = 8,
        ctx: Context | None = None,
    ) -> dict:
        """Hybrid search (meaning and keywords) over the vault notes. Returns notes, each with its best section.

        Filters: vault (the name of a vault folder), project (folder slug), area, type (research, spec, plan,
        how-to, decision, ..., one or a list), collection, class (state, record or none), status (active, archived,
        superseded; one or a list) and subject (an entity ID, name or alias, or a list: any of them matches). A
        redirect is never a result. include_private works only when the user allows it.

        Resolve the entities first (find_entity); for a question with one true answer use subject and type; a live
        value comes from the skill that get_entity names, not from a note. An unknown or ambiguous subject is an
        error with the candidate IDs. Each result has id, class, subjects, canonical, as_of, review_warning,
        successor and dispute; "subjects" gives the entity of each subject name.
        """
        return get().search(
            query,
            vault=vault,
            project=project,
            area=area,
            type=type,
            include_archived=include_archived,
            include_private=include_private,
            collection=collection,
            class_=class_,
            subject=subject,
            status=status,
            limit=limit,
        )

    @server.tool()
    @timed
    def find_projects(query: str, vault: str | None = None, limit: int = 5, ctx: Context | None = None) -> dict:
        """Find the earlier projects that a prompt continues or resembles, ranked by their matching notes.

        Each project has its slug, vault, folder, description, stage, state, next action and matching notes.
        """
        return get().find_projects(query, vault=vault, limit=limit)

    @server.tool()
    @timed
    def list_project(project: str, vault: str | None = None, ctx: Context | None = None) -> dict:
        """All notes of a project in any folder, with type, status and description, with stage, state, next action."""
        return get().list_project(project, vault=vault)

    @server.tool()
    @timed
    def open_projects(states: list[str] | None = None, vault: str | None = None, ctx: Context | None = None) -> dict:
        """The projects that are open, grouped by the state in their SUMMARY.md, newest update first.

        states defaults to in-progress and paused; the other values are done, superseded and abandoned. Each project has
        its slug, vault, folder, description, stage, state, gate, updated date and next action.
        """
        return get().open_projects(states=states, vault=vault)

    @server.tool()
    @timed
    def open_note(
        path: str, heading: str | None = None, outline: bool | None = None, ctx: Context | None = None
    ) -> dict:
        """The text of a note (or of one section, by heading), its frontmatter, links and backlinks.

        outline=true gives the sections (heading path, words, first sentence) instead of the text. A note of more
        than 4000 words gives the outline unless outline=false; then open the section you need by heading, as the
        last heading or the full path such as "Results > Inferences".
        Credentials in the text are replaced with [REDACTED]. Private notes are refused.
        """
        return get().open_note(path, heading=heading, outline=outline)

    @server.tool()
    @timed
    def find_by_jira(key: str, ctx: Context | None = None) -> dict:
        """The notes that contain a Jira key, for example ABC-123."""
        return get().find_by_jira(key)

    @server.tool()
    @timed
    def kb_check(collection: str | None = None, subject: str | None = None, ctx: Context | None = None) -> dict:
        """The count of each integrity check of the knowledge base, with the violations of M1 and M3.

        M1 is a broken link outside code. M3 is two active state documents with one identity key. The other checks
        are ref, vocab, relation_kinds, live_value, canonical_twice and subjects. Filters: collection, and subject
        (an entity ID, name or alias).
        """
        return get().kb_check(collection=collection, subject=subject)

    @server.tool()
    @timed
    def find_entity(name: str, kind: str | None = None, limit: int = 5, ctx: Context | None = None) -> dict:
        """Resolve a name to entity IDs: an exact alias first, then the words of the names, then the entity text.

        Cite the path of the entity that answers the request. Each entity has its id, kind, name, path (the entity
        document), aliases, owner, canonical documents, how it matched (alias, name or text)
        and a score. An ambiguous alias gives all its entities. kind narrows the result. Only an alias match is
        exact: a name or text result is the nearest entity and does not prove that the named entity exists.
        """
        return get().find_entity(name, kind=kind, limit=limit)

    @server.tool()
    @timed
    def get_entity(id: str, as_of: str | None = None, ctx: Context | None = None) -> dict:
        """An entity by its ID: its path (the entity document), the facts valid at as_of (default today),
        canonical documents and relation counts.

        A live attribute is a pointer {attribute, system, skill, key}, never a value: call the skill for the value.
        """
        return get().get_entity(id, as_of=as_of)

    @server.tool()
    @timed
    def related(
        entity: str,
        relation: str | None = None,
        direction: str = "out",
        kind: str | None = None,
        depth: int = 1,
        as_of: str | None = None,
        include_extracted: bool = False,
        ctx: Context | None = None,
    ) -> dict:
        """The exact join through the declared relations of an entity, to depth 3 (no cycle).

        entity is an ID or a name; an ambiguous name is an error with the candidate IDs. direction is out (the
        entity is the source), in (the entity is the target: the clients of a feature) or both (each hop follows
        a relation out or in). relation names one
        relation type. kind keeps the edges whose far end has the kind. as_of (default today) drops an edge that is
        not valid then, with each edge behind it. Each edge has its source and target (id, kind, name, path), relation,
        origin, confidence, validity, source_doc and depth. Only declared edges unless include_extracted is true.
        """
        return get().related(
            entity,
            relation=relation,
            direction=direction,
            kind=kind,
            depth=depth,
            as_of=as_of,
            include_extracted=include_extracted,
        )

    return server
