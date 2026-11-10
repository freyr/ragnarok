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
TOPK = 3  # chunks of a note that note_rank = topk_sum adds

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


def _vec_where(vector, k, vault, include_archived, include_private) -> tuple[str, list]:
    where, params = ["embedding MATCH ?", "k = ?"], [sqlite_vec.serialize_float32(vector), k]
    if vault is not None:
        where.append("vault = ?")
        params.append(vault)
    if not include_archived:
        where.append("status = 'active'")
    if not include_private:
        where.append("sensitivity = 'normal'")
    return " AND ".join(where), params


def _dense(conn, vector, k, vault, include_archived, include_private) -> list[tuple[int, float]]:
    """Ranked (chunk_id, cosine similarity)."""
    where, params = _vec_where(vector, k, vault, include_archived, include_private)
    sql = f"SELECT chunk_id, distance FROM chunks_vec WHERE {where} ORDER BY distance"
    return [(cid, 1.0 - distance) for cid, distance in conn.execute(sql, params)]


def _fts(conn, query, k, note_where, note_params) -> list[tuple[int, float]]:
    """Ranked (chunk_id, BM25 score); SQLite gives a lower bm25() value for a better match."""
    match = fts_query(query)
    if not match:
        return []
    sql = (
        "SELECT f.rowid, -bm25(chunks_fts, 2.0, 1.0) AS score FROM chunks_fts f JOIN chunks c ON c.id = f.rowid"
        f" JOIN notes n ON n.id = c.note_id WHERE chunks_fts MATCH ?{note_where} ORDER BY score DESC LIMIT ?"
    )
    return [(cid, score) for cid, score in conn.execute(sql, [match, *note_params, k])]


def _by_note(conn, scored: list[tuple[int, float]], note_where: str, note_params: list, note_rank: str):
    """Change a ranked (chunk_id, score) list into a ranked list of (note_id, best chunk_id), with the note filters.

    maxp: the rank of a note is the rank of its best chunk. topk_sum: the score of a note is the sum of the scores of
    its best TOPK chunks in the list."""
    if not scored:
        return []
    chunk_ids = [cid for cid, _ in scored]
    marks = ",".join("?" * len(chunk_ids))
    sql = f"SELECT c.id, c.note_id FROM chunks c JOIN notes n ON n.id = c.note_id WHERE c.id IN ({marks}){note_where}"
    note_of = dict(conn.execute(sql, [*chunk_ids, *note_params]).fetchall())
    best: dict[int, int] = {}
    scores: dict[int, list[float]] = {}
    for cid, score in scored:
        nid = note_of.get(cid)
        if nid is None:
            continue
        best.setdefault(nid, cid)
        scores.setdefault(nid, []).append(score)
    order = list(best)
    if note_rank == "topk_sum":
        # the list is in score order, so the first TOPK scores of a note are its best; the sort keeps ties in order
        order.sort(key=lambda nid: -sum(scores[nid][:TOPK]))
    return [(nid, best[nid]) for nid in order]


def _note_vectors(conn, vector, k, vault, include_archived, include_private, note_where, note_params):
    """Ranked (note_id, None) by the mean chunk vector of each note, with the note filters."""
    where, params = _vec_where(vector, k, vault, include_archived, include_private)
    ranked = [nid for (nid,) in conn.execute(f"SELECT note_id FROM notes_vec WHERE {where} ORDER BY distance", params)]
    if not ranked or not note_where:
        return [(nid, None) for nid in ranked]
    marks = ",".join("?" * len(ranked))
    sql = f"SELECT n.id FROM notes n WHERE n.id IN ({marks}){note_where}"
    keep = {nid for (nid,) in conn.execute(sql, [*ranked, *note_params])}
    return [(nid, None) for nid in ranked if nid in keep]


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
    rank = cfg.note_rank
    # the dense list, the BM25 list and, for note_rank = note_vector, the list of the mean chunk vectors of the notes
    lists: list[list[tuple[int, int | None]]] = [[], []]
    if mode in ("hybrid", "dense"):
        k = FILTERED_CANDIDATES if (project or area or type) else CANDIDATES
        vector = embedder.embed_query(query)
        scored = _dense(conn, vector, k, vault, include_archived, include_private)
        lists[0] = _by_note(conn, scored, note_where, note_params, rank)
        if rank == "note_vector":
            filters = (vault, include_archived, include_private, note_where, note_params)
            lists.append(_note_vectors(conn, vector, k, *filters))
    if mode in ("hybrid", "fts"):
        scored = _fts(conn, query, CANDIDATES, note_where, note_params)
        lists[1] = _by_note(conn, scored, note_where, note_params, rank)

    scores: dict[int, float] = {}
    ranks: dict[int, list[int | None]] = {}
    best: dict[int, tuple[int, int]] = {}  # note -> (rank, chunk) of its best chunk in a chunk list
    for li, ranked in enumerate(lists):
        for pos, (nid, cid) in enumerate(ranked, start=1):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + pos)
            ranks.setdefault(nid, [None, None])
            if cid is None:  # the note list has no chunk and no rank in the hit
                continue
            ranks[nid][li] = pos
            if nid not in best or pos < best[nid][0]:
                best[nid] = (pos, cid)
    top = sorted(scores, key=lambda n: -scores[n])[:limit]
    hits = []
    for nid in top:
        n = conn.execute(
            "SELECT path, vault, title, project, area, type, status, description FROM notes WHERE id = ?", (nid,)
        ).fetchone()
        if nid in best:
            row = conn.execute("SELECT heading, text FROM chunks WHERE id = ?", (best[nid][1],)).fetchone()
        else:  # found only by its note vector: the first chunk
            row = conn.execute(
                "SELECT heading, text FROM chunks WHERE note_id = ? ORDER BY ord LIMIT 1", (nid,)
            ).fetchone()
        heading, text = row
        words = text.split()
        snippet = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
        hits.append(Hit(*n, heading, snippet, round(scores[nid], 5), *ranks[nid]))
    return hits
