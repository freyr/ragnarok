"""Hybrid search: dense (sqlite-vec) and BM25 (FTS5) chunk lists, ranked by note, joined with RRF."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass

import sqlite_vec

from .config import Config

RRF_K = 60
CANDIDATES = 200  # chunks from each list
FILTERED_CANDIDATES = 1000  # when a filter applies after the vector search
SNIPPET_WORDS = 120

STOP = set(
    "a an the and or of to in on for with is are was were be been do does did i my me we our you your it its this "
    "that these those what which who where when how why can could should would will there their from at by as about "
    "into than then also any all some find check show list tell give please".split()
)


class PrivateNotAllowed(PermissionError):
    pass


@dataclass
class Hit:
    path: str
    vault: str
    title: str
    project: str
    area: str
    type: str
    status: str
    description: str | None
    heading: str
    text: str
    score: float
    dense_rank: int | None
    fts_rank: int | None

    def as_dict(self) -> dict:
        return asdict(self)


def fts_query(query: str) -> str:
    """An OR of the quoted query words, without stop words."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


def _note_filters(vault, project, area, type_, include_archived, include_private) -> tuple[str, list]:
    where, params = [], []
    for column, value in (("vault", vault), ("project", project), ("area", area), ("type", type_)):
        if value is not None:
            where.append(f"n.{column} = ?")
            params.append(value)
    if not include_archived:
        where.append("n.status = 'active'")
    if not include_private:
        where.append("n.sensitivity = 'normal'")
    return (" AND " + " AND ".join(where)) if where else "", params


def _dense(conn, vector, k, vault, include_archived, include_private) -> list[int]:
    where, params = ["embedding MATCH ?", "k = ?"], [sqlite_vec.serialize_float32(vector), k]
    if vault is not None:
        where.append("vault = ?")
        params.append(vault)
    if not include_archived:
        where.append("status = 'active'")
    if not include_private:
        where.append("sensitivity = 'normal'")
    sql = f"SELECT chunk_id FROM chunks_vec WHERE {' AND '.join(where)} ORDER BY distance"
    return [r[0] for r in conn.execute(sql, params)]


def _fts(conn, query, k, note_where, note_params) -> list[int]:
    match = fts_query(query)
    if not match:
        return []
    sql = (
        "SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid JOIN notes n ON n.id = c.note_id"
        f" WHERE chunks_fts MATCH ?{note_where} ORDER BY bm25(chunks_fts, 2.0, 1.0) LIMIT ?"
    )
    return [r[0] for r in conn.execute(sql, [match, *note_params, k])]


def _by_note(conn, chunk_ids: list[int], note_where: str, note_params: list) -> list[tuple[int, int]]:
    """Change a ranked chunk list into a ranked list of (note_id, best chunk_id), with the note filters."""
    if not chunk_ids:
        return []
    marks = ",".join("?" * len(chunk_ids))
    sql = f"SELECT c.id, c.note_id FROM chunks c JOIN notes n ON n.id = c.note_id WHERE c.id IN ({marks}){note_where}"
    note_of = dict(conn.execute(sql, [*chunk_ids, *note_params]).fetchall())
    seen: dict[int, int] = {}
    for cid in chunk_ids:
        nid = note_of.get(cid)
        if nid is not None and nid not in seen:
            seen[nid] = cid
    return list(seen.items())


def search(
    cfg: Config,
    conn: sqlite3.Connection,
    embedder,
    query: str,
    *,
    vault: str | None = None,
    project: str | None = None,
    area: str | None = None,
    type: str | None = None,
    include_archived: bool = True,
    include_private: bool = False,
    limit: int = 8,
    mode: str = "hybrid",
) -> list[Hit]:
    if include_private and not cfg.allow_private:
        raise PrivateNotAllowed("include_private needs allow_private = true in the configuration")
    note_where, note_params = _note_filters(vault, project, area, type, include_archived, include_private)
    lists: list[list[tuple[int, int]]] = []
    if mode in ("hybrid", "dense"):
        k = FILTERED_CANDIDATES if (project or area or type) else CANDIDATES
        ids = _dense(conn, embedder.embed_query(query), k, vault, include_archived, include_private)
        lists.append(_by_note(conn, ids, note_where, note_params))
    else:
        lists.append([])
    if mode in ("hybrid", "fts"):
        lists.append(_by_note(conn, _fts(conn, query, CANDIDATES, note_where, note_params), note_where, note_params))
    else:
        lists.append([])

    scores: dict[int, float] = {}
    ranks: dict[int, list[int | None]] = {}
    best: dict[int, tuple[int, int]] = {}  # note -> (rank, chunk) of its best chunk in any list
    for li, ranked in enumerate(lists):
        for rank, (nid, cid) in enumerate(ranked, start=1):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank)
            ranks.setdefault(nid, [None, None])[li] = rank
            if nid not in best or rank < best[nid][0]:
                best[nid] = (rank, cid)
    top = sorted(scores, key=lambda n: -scores[n])[:limit]
    hits = []
    for nid in top:
        n = conn.execute(
            "SELECT path, vault, title, project, area, type, status, description FROM notes WHERE id = ?", (nid,)
        ).fetchone()
        heading, text = conn.execute("SELECT heading, text FROM chunks WHERE id = ?", (best[nid][1],)).fetchone()
        words = text.split()
        snippet = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
        hits.append(Hit(*n, heading, snippet, round(scores[nid], 5), *ranks[nid]))
    return hits
