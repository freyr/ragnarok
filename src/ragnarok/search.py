"""Hybrid search: dense (exact numpy scan) and BM25 (FTS5) chunk lists, ranked by note, joined with RRF."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass

import numpy as np

from .config import Config

RRF_K = 60
CANDIDATES = 200  # units from each list
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


@dataclass(frozen=True)
class Filters:
    vault: str | None = None
    project: str | None = None
    area: str | None = None
    type: str | None = None
    include_archived: bool = True
    include_private: bool = False


def eligibility(filters: Filters) -> tuple[str, list]:
    """SQL condition on the alias d (a doc row) and its parameters."""
    where, params = [], []
    for column in ("vault", "project", "area", "type"):
        value = getattr(filters, column)
        if value is not None:
            where.append(f"d.{column} = ?")
            params.append(value)
    if not filters.include_archived:
        where.append("d.status = 'active'")
    if not filters.include_private:
        where.append("d.sensitivity = 'normal'")
    return (" AND ".join(where) or "1"), params


def _dense(conn, vector, filters: Filters) -> tuple[list[int], int]:
    """The CANDIDATES eligible units nearest to the vector (equal scores sort by unit_id), and the eligible count."""
    predicate, params = eligibility(filters)
    rows = conn.execute(
        f"SELECT e.unit_id, e.embedding FROM unit_emb e JOIN doc d ON d.id = e.doc_id WHERE {predicate}", params
    ).fetchall()
    if not rows:
        return [], 0
    ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(len(rows), -1)
    query = np.asarray(vector, dtype=np.float32)
    scores = matrix @ (query / np.linalg.norm(query))
    order = np.lexsort((ids, -scores))[:CANDIDATES]
    return ids[order].tolist(), len(rows)


def _fts(conn, query, k, filters: Filters) -> list[int]:
    match = fts_query(query)
    if not match:
        return []
    predicate, params = eligibility(filters)
    sql = (
        "SELECT f.rowid FROM unit_fts f JOIN unit c ON c.id = f.rowid JOIN doc d ON d.id = c.doc_id"
        f" WHERE unit_fts MATCH ? AND {predicate} ORDER BY bm25(unit_fts, 2.0, 1.0) LIMIT ?"
    )
    return [r[0] for r in conn.execute(sql, [match, *params, k])]


def _by_note(conn, unit_ids: list[int]) -> list[tuple[str, int]]:
    """Change a ranked unit list into a ranked list of (doc_id, best unit_id)."""
    if not unit_ids:
        return []
    marks = ",".join("?" * len(unit_ids))
    note_of = dict(conn.execute(f"SELECT id, doc_id FROM unit WHERE id IN ({marks})", unit_ids).fetchall())
    seen: dict[str, int] = {}
    for uid in unit_ids:
        nid = note_of.get(uid)
        if nid is not None and nid not in seen:
            seen[nid] = uid
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
    warnings: list[str] | None = None,
) -> list[Hit]:
    if include_private and not cfg.allow_private:
        raise PrivateNotAllowed("include_private needs allow_private = true in the configuration")
    filters = Filters(vault, project, area, type, include_archived, include_private)
    lists: list[list[tuple[str, int]]] = []
    if mode in ("hybrid", "dense"):
        unit_ids, eligible = _dense(conn, embedder.embed_query(query), filters)
        if warnings is not None and eligible > cfg.kb.exact_limit:
            warnings.append(
                f"The exact scan read {eligible} units, above the limit of {cfg.kb.exact_limit}; the search is slow."
            )
        lists.append(_by_note(conn, unit_ids))
    else:
        lists.append([])
    if mode in ("hybrid", "fts"):
        lists.append(_by_note(conn, _fts(conn, query, CANDIDATES, filters)))
    else:
        lists.append([])

    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}
    best: dict[str, tuple[int, int]] = {}  # note -> (rank, chunk) of its best chunk in any list
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
            "SELECT path, vault, title, project, area, type, status, description FROM doc WHERE id = ?", (nid,)
        ).fetchone()
        heading, text = conn.execute("SELECT heading_path, text FROM unit WHERE id = ?", (best[nid][1],)).fetchone()
        words = text.split()
        snippet = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
        hits.append(Hit(*n, heading, snippet, round(scores[nid], 5), *ranks[nid]))
    return hits
