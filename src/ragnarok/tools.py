"""The logic of the MCP tools. Each call scans the vaults for changes first, then reads the index."""

from __future__ import annotations

import datetime as dt
import posixpath
import re
import sqlite3
from dataclasses import dataclass, field

from .config import Config
from .index import SCAN_LIMIT, update
from .metadata import read_note
from .redact import redact
from .search import PrivateNotAllowed, search
from .splitter import sections

PROJECT_POOL = 50  # notes that find_projects groups into projects
DESCRIPTION_WORDS = 60
LONG_NOTE_WORDS = 4000  # open_note gives the outline of a longer note when no outline value is given
FIRST_WORDS = 25
OPEN_STATES = ("in-progress", "paused")


def _plain(value):
    """A frontmatter value that JSON can carry."""
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return redact(value)[0]
    return value


def _first_sentence(lines: list[str]) -> str:
    text = " ".join(" ".join(lines).split())
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    words = sentence.split()
    return " ".join(words[:FIRST_WORDS]) + (" …" if len(words) > FIRST_WORDS else "")


def _outline(secs) -> list[dict]:
    return [
        {
            "heading": " > ".join(s.path),
            "level": len(s.path),
            "words": s.words(),
            "first": _first_sentence(s.lines),
        }
        for s in secs
    ]


def _first_paragraph(body: str) -> str | None:
    for block in body.split("\n\n"):
        text = " ".join(line.strip() for line in block.strip().splitlines())
        if text and not text.startswith(("#", "---", "|", "```", "-", ">")):
            words = text.split()
            return " ".join(words[:DESCRIPTION_WORDS]) + (" …" if len(words) > DESCRIPTION_WORDS else "")
    return None


def _state(fm: dict, archived: bool) -> str:
    """The project state; a SUMMARY.md without one is done when it is archived or its stage is DONE."""
    state = str(fm.get("state") or "").strip().lower()
    if state:
        return state
    return "done" if archived or str(fm.get("stage", "")).strip().upper() == "DONE" else "in-progress"


@dataclass
class Tools:
    cfg: Config
    conn: sqlite3.Connection
    embedder: object
    scan_limit: int | None = SCAN_LIMIT
    last_scan: dict = field(default_factory=dict)

    def scan(self) -> list[str]:
        result = update(self.cfg, self.conn, self.embedder, limit=self.scan_limit)
        self.last_scan = {
            "ms": round(result.seconds * 1000),
            "changed": result.added + result.changed + result.moved + result.removed,
        }
        return result.warnings

    def _visible(self) -> str:
        return "" if self.cfg.allow_private else " AND n.sensitivity = 'normal'"

    def _note_row(self, path: str):
        return self.conn.execute(
            "SELECT id, vault, title, project, area, type, status, sensitivity, description FROM notes n"
            " WHERE path = ?",
            (path,),
        ).fetchone()

    def _check_private(self, sensitivity: str, path: str) -> None:
        if sensitivity == "private" and not self.cfg.allow_private:
            raise PrivateNotAllowed(f"{path} is a private note. Only the user can set allow_private.")

    # --- tools -------------------------------------------------------------------------------------------------

    def search(self, query: str, **filters) -> dict:
        warnings = self.scan()
        hits = search(self.cfg, self.conn, self.embedder, query, **filters)
        return {"results": [h.as_dict() for h in hits], "warnings": warnings}

    def find_projects(self, query: str, vault: str | None = None, limit: int = 5) -> dict:
        warnings = self.scan()
        hits = search(self.cfg, self.conn, self.embedder, query, vault=vault, limit=PROJECT_POOL)
        groups: dict[tuple[str, str], dict] = {}
        for h in hits:
            if h.project == "none":
                continue
            # the rank of a project is the rank of its best note; a sum favors large folders with weak matches
            g = groups.setdefault((h.vault, h.project), {"score": h.score, "notes": []})
            g["notes"].append(h.path)
        ranked = sorted(groups.items(), key=lambda item: (-item[1]["score"], -len(item[1]["notes"])))[:limit]
        projects = []
        for (v, project), g in ranked:
            info = self._project_info(v, project)
            projects.append({**info, "score": round(g["score"], 5), "matching_notes": g["notes"]})
        return {"results": projects, "warnings": warnings}

    def list_project(self, project: str, vault: str | None = None) -> dict:
        warnings = self.scan()
        where, params = "n.project = ?", [project]
        if vault:
            where += " AND n.vault = ?"
            params.append(vault)
        rows = self.conn.execute(
            f"SELECT vault, path, title, type, status, description FROM notes n WHERE {where}{self._visible()}"
            " ORDER BY vault, path",
            params,
        ).fetchall()
        by_vault: dict[str, list[dict]] = {}
        for v, path, title, type_, status, description in rows:
            by_vault.setdefault(v, []).append(
                {"path": path, "title": title, "type": type_, "status": status, "description": description}
            )
        results = [{**self._project_info(v, project), "notes": notes} for v, notes in by_vault.items()]
        return {"results": results, "warnings": warnings}

    def open_projects(self, states: list[str] | None = None, vault: str | None = None) -> dict:
        warnings = self.scan()
        wanted = {st.strip().lower() for st in (states or OPEN_STATES)}
        where, params = "(n.path LIKE '%/SUMMARY.md' OR n.path = 'SUMMARY.md')", []
        if vault:
            where += " AND n.vault = ?"
            params.append(vault)
        rows = self.conn.execute(
            f"SELECT n.vault, n.project, n.path, n.status FROM notes n WHERE {where}{self._visible()} ORDER BY n.path",
            params,
        ).fetchall()
        projects = []
        for v, project, path, status in rows:
            fm, _ = read_note(self.cfg.root / path)
            state = _state(fm, status == "archived")
            if state not in wanted:
                continue
            info = self._project_info(v, project)
            projects.append({**info, "state": state, "updated": _plain(fm.get("updated")), "gate": fm.get("gate")})
        by_state: dict[str, list[dict]] = {}
        for p in projects:
            by_state.setdefault(p["state"], []).append(p)
        for group in by_state.values():
            group.sort(key=lambda p: str(p["updated"] or ""), reverse=True)
        return {"results": by_state, "count": len(projects), "warnings": warnings}

    def open_note(self, path: str, heading: str | None = None, outline: bool | None = None) -> dict:
        warnings = self.scan()
        row = self._note_row(path)
        if row is None:
            raise FileNotFoundError(f"{path} is not a note in the index. Use a path from a search result.")
        note_id, vault, title, project, area, type_, status, sensitivity, description = row
        self._check_private(sensitivity, path)
        fm, body = read_note(self.cfg.root / path)
        secs = sections(body)
        if heading is not None:
            wanted = heading.strip().lstrip("#").strip().lower()
            found = [s for s in secs if s.path and (" > " + " > ".join(s.path).lower()).endswith(" > " + wanted)]
            if not found:
                names = [" > ".join(s.path) for s in secs if s.path]
                raise LookupError(f"No heading {heading!r} in {path}. Headings: {names}")
            body = "\n".join(found[0].lines)
            outline = False
        elif outline is None:
            outline = len(body.split()) > LONG_NOTE_WORDS
        if outline:
            content = {"outline": _outline(sections(redact(body)[0]))}
        else:
            text, redactions = redact(body)
            content = {"text": text, "redactions": redactions}
        hide = "" if self.cfg.allow_private else " AND (n.id IS NULL OR n.sensitivity = 'normal')"
        links = self.conn.execute(
            "SELECT l.target, n.path FROM links l LEFT JOIN notes n ON n.id = l.to_note"
            f" WHERE l.from_note = ?{hide} ORDER BY l.target",
            (note_id,),
        ).fetchall()
        backlinks = self.conn.execute(
            "SELECT DISTINCT n.path FROM links l JOIN notes n ON n.id = l.from_note"
            f" WHERE l.to_note = ?{self._visible()} ORDER BY n.path",
            (note_id,),
        ).fetchall()
        return {
            "path": path,
            "vault": vault,
            "title": title,
            "project": project,
            "area": area,
            "type": type_,
            "status": status,
            "description": description,
            "frontmatter": _plain(fm),
            **content,
            "links": [{"target": t, "path": p} for t, p in links],
            "backlinks": [p for (p,) in backlinks],
            "warnings": warnings,
        }

    def find_by_jira(self, key: str) -> dict:
        warnings = self.scan()
        rows = self.conn.execute(
            "SELECT DISTINCT n.path, n.title, n.project, n.type, n.status FROM jira j JOIN notes n ON n.id = j.note_id"
            f" WHERE j.key = ?{self._visible()} ORDER BY n.path",
            (key.strip().upper(),),
        ).fetchall()
        keys = ("path", "title", "project", "type", "status")
        return {"results": [dict(zip(keys, r, strict=True)) for r in rows], "warnings": warnings}

    # --- project summary ---------------------------------------------------------------------------------------

    def _project_info(self, vault: str, project: str) -> dict:
        rows = self.conn.execute(
            f"SELECT path, sensitivity FROM notes n WHERE n.vault = ? AND n.project = ?{self._visible()}",
            (vault, project),
        ).fetchall()
        paths = [p for p, _ in rows]
        markers = sorted(
            (p for p in paths if posixpath.basename(p) in ("SUMMARY.md", "DRIVERS.md")), key=lambda p: (len(p), p)
        )
        if markers:
            folder = posixpath.dirname(markers[0])
        elif paths:
            folder = posixpath.commonpath([posixpath.dirname(p) for p in paths])
        else:
            folder = None
        info = {
            "project": project,
            "vault": vault,
            "folder": folder,
            "description": None,
            "stage": None,
            "state": None,
            "next": None,
        }
        if not folder:
            return info
        summary, drivers = f"{folder}/SUMMARY.md", f"{folder}/DRIVERS.md"
        s_fm, s_body = read_note(self.cfg.root / summary) if summary in paths else ({}, "")
        d_fm = read_note(self.cfg.root / drivers)[0] if drivers in paths else {}
        description = s_fm.get("description") or d_fm.get("description") or _first_paragraph(s_body)
        info["description"] = redact(str(description))[0] if description else None
        info["stage"] = s_fm.get("stage")
        info["state"] = s_fm.get("state")
        info["next"] = redact(str(s_fm["next"]))[0] if s_fm.get("next") else None
        return info
