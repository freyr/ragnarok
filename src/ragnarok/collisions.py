"""Find the notes that a draft (or an indexed note) can collide with, before a write.

The candidate sources:
- identity: the notes with the same identity key (type, primary subject, task), in any class or validity;
- search: the hybrid search for the title, the description and the start of the body, superseded notes included;
- links_out: the notes that the draft links to (body and frontmatter, as check 4 of `validate` reads them);
- links_to_supersedes: each target of `supersedes`, and the notes that link to it.

Each candidate has its labels, its reasons, its search rank and the cosine distance of its best chunk to the query.
"""

from __future__ import annotations

import re
import sqlite3

import sqlite_vec

from .config import Config
from .metadata import parse_note
from .search import PrivateNotAllowed, search
from .validate import _Context, body_links, frontmatter_links

LIMIT = 10
QUERY_CHARS = 1000
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


class _OneQuery:
    """An embedder that embeds each query once, so that the search and the distances use one vector."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.vectors: dict[str, list[float]] = {}

    def embed_query(self, query: str) -> list[float]:
        if query not in self.vectors:
            self.vectors[query] = self.embedder.embed_query(query)
        return self.vectors[query]


def query_text(title: str, description: str | None, body: str) -> str:
    """The search text of a draft: the title, the description and the body, cut at QUERY_CHARS characters."""
    return "\n".join(part for part in (title, description, body.strip()) if part)[:QUERY_CHARS]


def _first_heading(body: str) -> str:
    m = HEADING.search(body)
    return m.group(1) if m else ""


def _distances(conn: sqlite3.Connection, vector: list[float], ids: list[int]) -> dict[int, float]:
    """The cosine distance of the best chunk of each note to the vector; a note with no vector is not in it."""
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT c.note_id, min(vec_distance_cosine(v.embedding, ?)) FROM chunks c"
        f" JOIN chunks_vec v ON v.chunk_id = c.id WHERE c.note_id IN ({marks}) GROUP BY c.note_id",
        [sqlite_vec.serialize_float32(vector), *ids],
    ).fetchall()
    return dict(rows)


def find_collisions(
    cfg: Config,
    conn: sqlite3.Connection,
    embedder,
    *,
    draft: str | None = None,
    draft_path: str | None = None,
    path: str | None = None,
    allow_private: bool = False,
) -> list[dict]:
    """The collision candidates of a draft (with its intended key draft_path) or of the indexed note at path.

    The order: identity candidates (by search rank, then distance), search candidates (by rank), then the
    candidates of the links only (by distance); at most LIMIT. The note at path is not a candidate. A private
    note is a candidate only with allow_private, which needs allow_private = true in the configuration."""
    if allow_private and not cfg.allow_private:
        raise PrivateNotAllowed("allow_private needs allow_private = true in the configuration")
    if draft is not None:
        text, key, exclude = draft, draft_path, None
    elif path is not None:
        row = conn.execute("SELECT id FROM notes WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"{path} is not a note in the index. Use a path from a search result.")
        text, key, exclude = cfg.note_path(path).read_text(errors="replace"), path, row[0]
    else:
        raise ValueError("Give a draft or a path.")

    ctx = _Context(cfg, conn)
    note = ctx.note(text, key, strict=True)
    fm, body = parse_note(text)
    query = query_text(note.meta.title if key else _first_heading(body), note.meta.description, body)
    one = _OneQuery(embedder)
    reasons: dict[int, list[str]] = {}

    def add(nid: int | None, reason: str) -> None:
        if nid is not None and nid != exclude and reason not in reasons.setdefault(nid, []):
            reasons[nid].append(reason)

    if note.identity is not None:
        for nid, _, _ in ctx.identity_notes(note.identity, current_only=False):
            add(nid, "identity")
    hits = search(
        cfg,
        conn,
        one,
        query,
        include_superseded=True,
        include_private=allow_private,
        limit=LIMIT + (exclude is not None),
    )
    ranks: dict[int, int] = {}
    for rank, hit in enumerate(hits, start=1):
        nid = ctx.ids.get(hit.path)
        if nid is not None and nid != exclude:
            ranks[nid] = rank
            add(nid, "search")
    for target in [*frontmatter_links(fm), *body_links(body)]:
        if target:
            add(ctx.resolver.paths.find(target, note.note_id, None, note.path), "links_out")
    for target in note.meta.supersedes:
        to = ctx.resolver.note(target, note.note_id, note.path)
        if to is None:
            continue
        add(to, "links_to_supersedes")
        for (nid,) in conn.execute("SELECT DISTINCT from_note FROM links WHERE to_note = ?", (to,)):
            add(nid, "links_to_supersedes")

    if not reasons:
        return []
    ids = list(reasons)
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT id, path, title, type, status, sensitivity, note_class, validity, as_of, superseded_by, subject"
        f" FROM notes WHERE id IN ({marks})",
        ids,
    ).fetchall()
    distances = _distances(conn, one.embed_query(query), ids)
    candidates = []
    for nid, npath, title, type_, status, sensitivity, cls, validity, as_of, successor, subject in rows:
        if sensitivity == "private" and not allow_private:
            continue
        candidates.append(
            {
                "path": npath,
                "title": title,
                "type": type_,
                "status": status,
                "class": cls,
                "validity": validity,
                "as_of": as_of,
                "subject": subject.split("\n") if subject else None,
                "superseded_by": successor,
                "reasons": reasons[nid],
                "rank": ranks.get(nid),
                "distance": distances.get(nid),
            }
        )

    def distance(c: dict) -> tuple:
        return (c["distance"] is None, c["distance"] or 0.0, c["path"])

    def order(c: dict) -> tuple:
        if "identity" in c["reasons"]:
            return (0, c["rank"] is None, c["rank"] or 0, *distance(c))
        if c["rank"] is not None:
            return (1, False, c["rank"], *distance(c))
        return (2, True, 0, *distance(c))

    return sorted(candidates, key=order)[:LIMIT]


def format_text(candidates: list[dict]) -> str:
    """One line for each candidate: the position, the path, the reasons, the rank and the distance."""
    lines = []
    for i, c in enumerate(candidates, start=1):
        distance = "-" if c["distance"] is None else f"{c['distance']:.4f}"
        rank = "-" if c["rank"] is None else c["rank"]
        lines.append(f"{i}. {c['path']}  {', '.join(c['reasons'])}  rank {rank}  distance {distance}")
    lines.append(f"{len(candidates)} candidates")
    return "\n".join(lines)
