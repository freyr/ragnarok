"""Hybrid search: dense (exact numpy scan) and BM25 (FTS5) chunk lists, ranked by note, joined with RRF."""

from __future__ import annotations

import contextlib
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from .config import WHEN_KINDS, Config
from .trace import TOP, Trace, gold_paths, safely

RRF_K = 60
ENTITY_FIRST = 3  # the entity documents that go before the ranked results, at most
CANDIDATES = 200  # units from each list
SNIPPET_WORDS = 120

STOP = set(
    "a an the and or of to in on for with is are was were be been do does did i my me we our you your it its this "
    "that these those what which who where when how why can could should would will there their from at by as about "
    "into than then also any all some find check show list tell give please".split()
)


class PrivateNotAllowed(PermissionError):
    pass


class SubjectNotResolved(LookupError):
    """A `subject` name with no entity or with more than one. The trace labels it QUERY: the query was wrong."""

    trace_label = "QUERY"

    def __init__(self, message: str, candidates: list[dict]):
        super().__init__(message)
        self.candidates = candidates


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
    score: float | None  # None for an entity-first hit: the score did not set its rank
    dense_rank: int | None
    fts_rank: int | None
    # the scent fields: what the agent needs to judge the document before it reads it
    id: str | None = None
    class_: str | None = None
    subjects: list[dict] = field(default_factory=list)
    canonical: bool = False
    as_of: str | None = None
    review_warning: str | None = None
    successor: dict | None = None
    dispute: dict | None = None
    duplicate: bool = False  # the second document of one identity key: the authority factor halves its score
    entity_first: bool = False  # the entity document of a query subject, before the ranked results

    def as_dict(self) -> dict:
        out = asdict(self)
        out["class"] = out.pop("class_")
        return out


def fts_query(query: str) -> str:
    """An OR of the quoted query words, without stop words."""
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOP and len(w) > 1]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(words))


Values = str | list[str] | tuple[str, ...] | None


@dataclass(frozen=True)
class Filters:
    vault: str | None = None
    project: str | None = None
    area: str | None = None
    type: Values = None
    include_archived: bool = True
    include_private: bool = False
    collection: Values = None
    class_: Values = None
    status: Values = None
    subject: tuple[str, ...] | None = None  # entity IDs, after the names resolve


COLUMNS = ("vault", "project", "area", "type", "collection", "class", "status")
STAT = {"collection", "type", "class", "area", "status"}  # the filter columns that `idx.stat` counts


def _value(filters: Filters, column: str) -> tuple[str, ...] | None:
    """The values of a column filter: one string or a list, as a tuple."""
    value = getattr(filters, "class_" if column == "class" else column)
    return (value,) if isinstance(value, str) else (tuple(value) if value is not None else None)


def eligibility(filters: Filters) -> tuple[str, list]:
    """SQL condition on the alias d (a doc row) and its parameters. A redirect is never eligible."""
    where, params = [], []
    for column in COLUMNS:
        values = _value(filters, column)
        if values is not None:
            where.append(f"d.{column} IN ({', '.join('?' * len(values))})")
            params.extend(values)
    if filters.subject is not None:
        # a document of a subject entity, or the entity document of a subject entity
        marks = ", ".join("?" * len(filters.subject))
        where.append(
            f"(EXISTS (SELECT 1 FROM doc_subject s WHERE s.doc_id = d.id AND s.entity_id IN ({marks}))"
            f" OR EXISTS (SELECT 1 FROM entity e WHERE e.doc_id = d.id AND e.id IN ({marks})))"
        )
        params.extend(filters.subject * 2)
    if not filters.include_archived:
        where.append("d.status = 'active'")
    if not filters.include_private:
        where.append("d.sensitivity = 'normal'")
    where.append("d.status != 'redirect'")
    return " AND ".join(where), params


def active_filters(filters: Filters) -> list[str]:
    """The filters that apply: the columns that have a value, `subject`, `archived` and `private`."""
    names = [c for c in COLUMNS if _value(filters, c) is not None] + ["subject"] * (filters.subject is not None)
    return names + ["archived"] * (not filters.include_archived) + ["private"] * (not filters.include_private)


def filter_checks(doc: dict, filters: Filters) -> dict[str, bool]:
    """For each filter that applies: does the doc row pass it? A redirect fails the rule that no filter turns off."""
    passes = {
        "archived": lambda: doc["status"] == "active",
        "private": lambda: doc["sensitivity"] == "normal",
        "subject": lambda: bool({*doc["subjects"], doc["entity"]} & set(filters.subject or ())),
    }
    checks = {n: passes[n]() if n in passes else doc[n] in _value(filters, n) for n in active_filters(filters)}
    return checks | ({"redirect": False} if doc["status"] == "redirect" else {})


def _plan(conn, filters: Filters, exact_limit: int) -> dict:
    """The eligible units (an SQL count), the selectivity of each filter alone, and the scan mode.

    The selectivity of a filter on a column of `idx.stat` is the sum of the units of its values in `stat`: for a
    list of subjects it is an upper bound, as a document can have more than one of them. `vault` and `project` have
    no `stat` row, so their selectivity is an SQL count.
    """

    def counted(column: str, values) -> int:
        marks = ", ".join("?" * len(values))
        sql = f'SELECT coalesce(sum(n_units), 0) FROM idx.stat WHERE "column" = ? AND value IN ({marks})'
        return conn.execute(sql, [column, *values]).fetchone()[0]

    def units(f: Filters) -> int:
        predicate, params = eligibility(f)
        sql = f"SELECT count(*) FROM unit u JOIN doc d ON d.id = u.doc_id WHERE {predicate}"
        return conn.execute(sql, params).fetchone()[0]

    total = conn.execute("SELECT count(*) FROM unit").fetchone()[0]
    selectivity = {}
    for name in active_filters(filters):
        if name in STAT:
            n = counted(name, _value(filters, name))
        elif name == "subject":
            n = counted("subject", filters.subject)
        elif name == "archived":
            n = counted("status", ["active"])
        elif name == "private":
            n = counted("sensitivity", ["normal"])
        else:
            n = units(Filters(**{name: getattr(filters, name)}, include_private=True))
        selectivity[name] = {"units": n, "selectivity": round(n / total, 5) if total else None}
    eligible = units(filters)
    return {
        "total_units": total,
        "eligible_units": eligible,
        "selectivity": selectivity,
        "mode": "exact" if eligible <= exact_limit else "over_limit",
    }


def resolve_subjects(conn, names: Values, allow_private: bool, trace: Trace | None) -> list[dict]:
    """The entity of each subject name: an ID, else an alias or a name. The trace gets every name and its
    candidates; a name with no entity or more than one raises SubjectNotResolved with the candidates."""
    from . import kb  # kb imports fts_query from this module

    names = [names] if isinstance(names, str) else list(names)
    entries = []
    for name in names:
        candidates = kb.resolve(conn, name, allow_private)
        selected = candidates[0]["id"] if len(candidates) == 1 else None
        entries.append({"name": name, "candidates": candidates, "selected": selected})
    if trace:
        trace.set("ragnarok.entities", entries)
    resolved = []
    for e in entries:
        try:
            entity_id = kb.pick(e["candidates"], e["name"])
        except LookupError as error:
            raise SubjectNotResolved(str(error), e["candidates"]) from None
        resolved.append({"name": e["name"], "id": entity_id, "matched_by": e["candidates"][0]["matched_by"]})
    return resolved


def question_kind(cfg: Config, query: str, when: str | None = None) -> tuple[str, str, str | None]:
    """The kind of question (now, future, history or unknown), its source (parameter or rule) and the rule word
    that matched. With no `when`, a word or phrase of `kb.when_rules` in the query text decides (whole words, any
    case); history goes before future, so that "what was the plan" is a history question. Else `unknown`: no
    `when` factor, no canonical factor and no entity-first, so that a question of no known kind keeps the
    relevance order."""
    if when is not None:
        if when not in WHEN_KINDS:
            raise ValueError(f"when must be one of {', '.join(WHEN_KINDS)}, not {when!r}")
        return when, "parameter", None
    for kind in ("history", "future"):
        for word in cfg.kb.when_rules.get(kind, []):
            if re.search(rf"\b{r'\s+'.join(map(re.escape, word.split()))}\b", query, re.IGNORECASE):
                return kind, "rule", word
    return "unknown", "rule", None


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
    type: Values = None,
    include_archived: bool = True,
    include_private: bool = False,
    collection: Values = None,
    class_: Values = None,
    subject: Values = None,
    status: Values = None,
    limit: int = 8,
    mode: str = "hybrid",
    warnings: list[str] | None = None,
    resolved: list[dict] | None = None,
    trace: Trace | None = None,
    authority: bool = True,
    entity_first: bool | None = None,
    when: str | None = None,
) -> list[Hit]:
    """The ranked hits. `subject` is an entity ID or a name, or a list of them (any of them matches); `resolved`
    gets the entity of each name. The score is the RRF score times the authority factors; `authority=False` turns
    them off. `when` is the kind of question (now, future or history; None: the rule list of `kb.when_rules`
    decides, else the kind is unknown); it selects the factor of each note in `kb.when_factors` (1.0 for
    unknown). The entity documents of the query subjects go first when `entity_first` (None: `kb.entity_first`)
    and `authority` are true and the question is `now` or unknown."""
    if include_private and not cfg.allow_private:
        raise PrivateNotAllowed("include_private needs allow_private = true in the configuration")
    kind, kind_source, kind_word = question_kind(cfg, query, when)
    # an empty list is no filter, not a filter that nothing passes
    type, collection, class_, subject, status = (
        None if isinstance(v, (list, tuple)) and not v else v for v in (type, collection, class_, subject, status)
    )
    # the same rule as find_entity, get_entity and related: an entity ID that they give resolves here too
    subjects = resolve_subjects(conn, subject, cfg.allow_private, trace) if subject is not None else None
    if resolved is not None and subjects:
        resolved.extend(subjects)
    subject_ids = tuple(dict.fromkeys(e["id"] for e in subjects)) if subjects is not None else None
    filters = Filters(
        vault, project, area, type, include_archived, include_private, collection, class_, status, subject_ids
    )
    stage = trace.stage if trace else lambda name: contextlib.nullcontext()
    plan, vector = None, None
    if trace:
        with stage("plan"):
            plan = safely("plan", _plan, conn, filters, cfg.kb.exact_limit)
    lists: list[list[tuple[str, int]]] = []
    pools = {"dense": 0, "fts": 0}
    if mode in ("hybrid", "dense"):
        with stage("embed"):
            vector = embedder.embed_query(query)
        with stage("dense"):
            unit_ids, eligible = _dense(conn, vector, filters)
            lists.append(_by_note(conn, unit_ids))
        pools["dense"] = len(unit_ids)
        if warnings is not None and eligible > cfg.kb.exact_limit:
            warnings.append(
                f"The exact scan read {eligible} units, above the limit of {cfg.kb.exact_limit}; the search is slow."
            )
    else:
        lists.append([])
    if mode in ("hybrid", "fts"):
        with stage("fts"):
            fts_units = _fts(conn, query, CANDIDATES, filters)
            lists.append(_by_note(conn, fts_units))
        pools["fts"] = len(fts_units)
    else:
        lists.append([])

    with stage("rank"):
        scores: dict[str, float] = {}
        ranks: dict[str, list[int | None]] = {}
        best: dict[str, tuple[int, int]] = {}  # note -> (rank, chunk) of its best chunk in any list
        for li, ranked in enumerate(lists):
            for rank, (nid, cid) in enumerate(ranked, start=1):
                scores[nid] = scores.get(nid, 0.0) + 1.0 / (RRF_K + rank)
                ranks.setdefault(nid, [None, None])[li] = rank
                if nid not in best or rank < best[nid][0]:
                    best[nid] = (rank, cid)
        factors = _authority(cfg, conn, query, list(scores), subject_ids or (), kind) if authority else {}
        final = {n: scores[n] * math.prod(factors.get(n, {}).values()) for n in scores}
        order = sorted(scores, key=lambda n: -final[n])
        on = (
            authority and kind in ("now", "unknown") and (cfg.kb.entity_first if entity_first is None else entity_first)
        )
        first = _entity_first(cfg, conn, query, subject_ids or (), filters) if on else None
        ahead = [e["doc_id"] for e in first or []]
        order = ahead + [n for n in order if n not in ahead]
        top = order[:limit]
        hits = []
        for nid in top:
            n = conn.execute(
                "SELECT path, vault, title, project, area, type, status, description FROM doc WHERE id = ?", (nid,)
            ).fetchone()
            if nid in best:
                unit = conn.execute("SELECT heading_path, text FROM unit WHERE id = ?", (best[nid][1],)).fetchone()
            else:  # an entity-first document outside the RRF list: its first unit; a note with no body has none
                sql = "SELECT heading_path, text FROM unit WHERE doc_id = ? ORDER BY ord LIMIT 1"
                unit = conn.execute(sql, (nid,)).fetchone()
            heading, text = unit or (n[2], "")
            words = text.split()
            snippet = " ".join(words[:SNIPPET_WORDS]) + (" …" if len(words) > SNIPPET_WORDS else "")
            duplicate = "duplicate_identity" in factors.get(nid, {})
            scent = _scent(conn, nid, include_private)
            score = None if nid in ahead else round(final[nid], 5)
            dense_rank, fts_rank = ranks.get(nid, (None, None))
            hits.append(
                Hit(
                    *n,
                    heading,
                    snippet,
                    score,
                    dense_rank,
                    fts_rank,
                    **scent,
                    duplicate=duplicate,
                    entity_first=nid in ahead,
                )
            )
    if trace:
        trace.set("ragnarok.when", {"kind": kind, "source": kind_source, "matched": kind_word})
        if first is not None:
            trace.set("ragnarok.entity_first", first)
        if plan is not None:
            trace.set("ragnarok.plan", {**plan, "pools": pools, "pool_limit": CANDIDATES})
        safely(
            "candidates",
            _record,
            conn,
            trace,
            filters,
            (query, vector),
            scores,
            final,
            factors,
            ranks,
            best,
            order,
            top,
            ahead,
        )
    return hits


def _review_warning(class_: str, review_by: str | None) -> str | None:
    """The warning of a state document whose review date is in the past."""
    if class_ == "state" and review_by and review_by < date.today().isoformat():
        return f"The review date {review_by} is in the past: the content can be out of date."
    return None


def _text_subjects(conn, query: str, allow_private: bool) -> list[str]:
    """The entities whose alias occurs in the query text as a phrase of whole words, in the order of their first
    position in the query (equal positions sort by entity ID)."""
    from . import kb  # kb imports fts_query from this module

    words = lambda text: " ".join(re.findall(r"\w+", kb.normalize_alias(text)))  # noqa: E731
    padded = f" {words(query)} "
    rows = conn.execute(
        "SELECT DISTINCT a.entity_id, a.alias_norm FROM entity_alias a JOIN entity e ON e.id = a.entity_id"
        f" JOIN doc d ON d.id = e.doc_id WHERE 1 = 1{kb._visible(allow_private)}"
    )
    position: dict[str, int] = {}
    for entity_id, alias in rows:
        if (w := words(alias)) and (at := padded.find(f" {w} ")) >= 0:
            position[entity_id] = min(at, position.get(entity_id, at))
    return sorted(position, key=lambda e: (position[e], e))


_FUNCTION_WORDS = frozenset(
    "the and for with what which who whom how does did are was were this that these those from into when where why"
    " our your their its not can has have had use uses used".split()
)


def _stem(word: str) -> str:
    """A light stem, so that `modes` and `mode`, `imaging` and `image`, `delayed` and `delay` match."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def _content_words(text: str) -> set[str]:
    """The words of a text that can match a fact: three characters or more (or a number), no function word, with
    a plural `s` removed; an attribute name splits at its underscores."""
    from . import kb

    words = re.findall(r"[^\W_]+", kb.normalize_alias(text))
    return {_stem(w) for w in words if (len(w) >= 3 or w.isdigit()) and w not in _FUNCTION_WORDS}


def _fact_match(conn, entity_id: str, query: str) -> bool:
    """True when the content words of the query, other than the words of the aliases of the entity that the query
    holds as a phrase, hold all words of an attribute or a canonical aspect of the entity, or two of them at least,
    or a word of a value. One shared word of a longer attribute (`temperature` of `outdoor_temperature`) is weak
    evidence, so it is not enough."""
    from . import kb

    words = lambda text: " ".join(re.findall(r"\w+", kb.normalize_alias(text)))  # noqa: E731
    padded = f" {words(query)} "
    named = set()
    for (alias,) in conn.execute("SELECT alias_norm FROM entity_alias WHERE entity_id = ?", (entity_id,)):
        if (w := words(alias)) and f" {w} " in padded:
            named |= _content_words(alias)
    asked = _content_words(query) - named
    if not asked:
        return False
    today = date.today().isoformat()  # an expired or future fact does not count, as in `kb.get`
    rows = conn.execute(
        "SELECT attribute, value FROM fact WHERE entity_id = ? AND status = 'active'"
        " AND (valid_from IS NULL OR valid_from <= ?) AND (valid_until IS NULL OR valid_until >= ?)"
        " UNION ALL SELECT aspect, NULL FROM entity_canonical WHERE entity_id = ?",
        (entity_id, today, today, entity_id),
    )
    for attribute, value in rows:
        words = _content_words(attribute)
        if (words and len(asked & words) >= min(2, len(words))) or asked & _content_words(value or ""):
            return True
    return False


def _entity_first(cfg: Config, conn, query: str, subject_ids: tuple[str, ...], filters: Filters) -> list[dict]:
    """The entity documents that go before the ranked results: those of the `subject` entities, then those of the
    entities in the query text that hold a fact the query asks for (`_fact_match`), each only when the filters
    admit it, at most ENTITY_FIRST."""
    reasons = dict.fromkeys(subject_ids, "subject")
    for entity_id in _text_subjects(conn, query, cfg.allow_private):
        if entity_id not in reasons and _fact_match(conn, entity_id, query):
            reasons[entity_id] = "fact"
    if not reasons:
        return []
    predicate, params = eligibility(filters)
    marks = ",".join("?" * len(reasons))
    sql = f"SELECT e.id, e.doc_id FROM entity e JOIN doc d ON d.id = e.doc_id WHERE e.id IN ({marks}) AND {predicate}"
    docs = dict(conn.execute(sql, [*reasons, *params]).fetchall())
    admitted = [{"id": e, "doc_id": docs[e], "reason": reason} for e, reason in reasons.items() if e in docs]
    return admitted[:ENTITY_FIRST]


def _authority(
    cfg: Config, conn, query: str, doc_ids: list[str], subject_ids: tuple[str, ...], kind: str = "now"
) -> dict[str, dict]:
    """The authority factors of each document: the factors that apply, by their name in `kb.authority`, and
    always `when`, the factor of `kb.when_factors` for the kind and validity of the document and the kind of
    question (1.0 for a document with no row).

    The canonical factor applies only to a `now` question or a question of unknown kind. The subjects of the
    query are the resolved `subject` values, and with `subject_from_text` also the entities whose alias is in the
    query text. The first document of an identity key is the canonical one, else the one with the latest `as_of`,
    else the one with the latest file time; only active documents count.
    """
    if not doc_ids:
        return {}
    a = cfg.kb.authority
    subjects = set(subject_ids) if kind in ("now", "unknown") else set()
    if a.subject_from_text and kind in ("now", "unknown"):
        subjects |= set(_text_subjects(conn, query, cfg.allow_private))
    marks = ",".join("?" * len(doc_ids))
    docs = conn.execute(
        f"SELECT id, class, validity, status, review_by, identity_key FROM doc WHERE id IN ({marks})", doc_ids
    ).fetchall()
    canonical: set[str] = set()
    if subjects:
        smarks = ",".join("?" * len(subjects))
        sql = f"SELECT DISTINCT doc_id FROM entity_canonical WHERE doc_id IN ({marks}) AND entity_id IN ({smarks})"
        canonical = {r[0] for r in conn.execute(sql, [*doc_ids, *subjects])}
    keys = sorted({d[5] for d in docs if d[5] is not None and d[3] == "active"})
    first: dict[str, tuple] = {}
    if keys:
        kmarks = ",".join("?" * len(keys))
        group = conn.execute(
            f"SELECT identity_key, id, canonical, coalesce(as_of, ''), mtime_ns FROM doc"
            f" WHERE status = 'active' AND identity_key IN ({kmarks})",
            keys,
        )
        for key, id_, flag, as_of, mtime in group:
            if key not in first or (flag, as_of, mtime, id_) > first[key][1:]:
                first[key] = (id_, flag, as_of, mtime, id_)
    out: dict[str, dict] = {}
    for id_, class_, validity, status, review_by, key in docs:
        row = cfg.kb.when_factors.get(f"{class_}_{validity}" if validity else class_, {})
        applied = {
            "canonical": a.canonical if id_ in canonical else None,
            "review_overdue": a.review_overdue if _review_warning(class_, review_by) else None,
            "duplicate_identity": a.duplicate_identity
            if key in first and first[key][0] != id_ and status == "active"
            else None,
            "when": row.get(kind, 1.0),
        }
        out[id_] = {name: factor for name, factor in applied.items() if factor is not None}
    return out


def _scent(conn, doc_id: str, include_private: bool) -> dict:
    """The scent fields of a document. A private subject or successor shows no name or path without
    include_private."""
    class_, as_of, review_by, successor, owner, due = conn.execute(
        "SELECT class, as_of, review_by, superseded_by, dispute_owner_text, dispute_due FROM doc WHERE id = ?",
        (doc_id,),
    ).fetchone()
    visible = "" if include_private else " AND d.sensitivity = 'normal'"
    subjects = [
        {"id": entity_id, "name": name or text}
        for entity_id, text, name in conn.execute(
            "SELECT s.entity_id, s.target_text, (SELECT e.name FROM entity e JOIN doc d ON d.id = e.doc_id"
            f" WHERE e.id = s.entity_id{visible}) FROM doc_subject s WHERE s.doc_id = ? ORDER BY s.ord",
            (doc_id,),
        )
    ]
    canonical = conn.execute("SELECT 1 FROM entity_canonical WHERE doc_id = ? LIMIT 1", (doc_id,)).fetchone()
    warning = _review_warning(class_, review_by)
    if successor is not None:
        row = conn.execute(f"SELECT d.path FROM doc d WHERE d.id = ?{visible}", (successor,)).fetchone()
        successor = {"id": successor, "path": row[0] if row else None}
    return {
        "id": doc_id,
        "class_": class_,
        "subjects": subjects,
        "canonical": canonical is not None,
        "as_of": as_of,
        "review_warning": warning,
        "successor": successor,
        "dispute": {"owner": owner, "due": due} if owner or due else None,
    }


def _first_seen(doc_ids) -> dict[str, int]:
    """The rank of the best unit of each document: the 1-based position of its first unit in a ranked list of
    unit owners. A pool holds the first CANDIDATES units, so a rank at or below CANDIDATES is inside the pool."""
    ranks: dict[str, int] = {}
    for position, d in enumerate(doc_ids, start=1):
        ranks.setdefault(d, position)
    return ranks


def _prefilter_ranks(conn, query: str, vector) -> dict[str, tuple[int | None, int | None]]:
    """The unit rank of the best unit of every document with no filter: an exact scan of all units and the
    unfiltered FTS list. None means no vector or no match."""
    dense: dict[str, int] = {}
    if vector is not None:
        rows = conn.execute("SELECT unit_id, doc_id, embedding FROM unit_emb").fetchall()
        if rows:
            ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
            matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(len(rows), -1)
            q = np.asarray(vector, dtype=np.float32)
            order = np.lexsort((ids, -(matrix @ (q / np.linalg.norm(q)))))
            dense = _first_seen(rows[i][1] for i in order)
    fts: dict[str, int] = {}
    if match := fts_query(query):
        sql = (
            "SELECT c.doc_id FROM unit_fts f JOIN unit c ON c.id = f.rowid"
            " WHERE unit_fts MATCH ? ORDER BY bm25(unit_fts, 2.0, 1.0)"
        )
        fts = _first_seen(r[0] for r in conn.execute(sql, [match]))
    return {d: (dense.get(d), fts.get(d)) for d in dense.keys() | fts.keys()}


def _record(
    conn, trace: Trace, filters: Filters, probe, scores, finals, factors, ranks, best, order, top, ahead
) -> None:
    """The candidates (the top 50 of each pool, the entity-first documents and every gold document), the returned
    documents and their status. An entity-first document has the score None."""
    final_rank = {nid: i for i, nid in enumerate(order, start=1)}
    in_top = {nid for nid, r in ranks.items() if any(x is not None and x <= TOP for x in r)} | set(ahead)
    gold_ids = {}
    for path in safely("gold paths", gold_paths) or []:
        row = conn.execute("SELECT id FROM doc WHERE path = ?", (path,)).fetchone()
        gold_ids[path] = row[0] if row else None
    wanted = in_top | {i for i in gold_ids.values() if i}
    marks = ",".join("?" * len(wanted))
    cols = ("id", "path", "vault", "project", "area", "type", "collection", "class", "status", "sensitivity")
    docs = {
        r[0]: dict(zip(cols, r, strict=True), subjects=[], entity=None)
        for r in conn.execute(f"SELECT {', '.join(cols)} FROM doc WHERE id IN ({marks})", list(wanted))
    }
    for nid, entity_id in conn.execute(f"SELECT doc_id, id FROM entity WHERE doc_id IN ({marks})", list(wanted)):
        docs[nid]["entity"] = entity_id
    sql = f"SELECT doc_id, entity_id FROM doc_subject WHERE entity_id IS NOT NULL AND doc_id IN ({marks})"
    for nid, entity_id in conn.execute(sql, list(wanted)):
        docs[nid]["subjects"].append(entity_id)

    gold_set = {i for i in gold_ids.values() if i}
    before = (safely("prefilter ranks", _prefilter_ranks, conn, *probe) if gold_set else None) or {}

    def record(nid: str, gold: bool) -> dict:
        d, (dense_rank, fts_rank) = docs[nid], ranks.get(nid, (None, None))
        out = {
            "doc_id": nid,
            "path": d["path"],
            "unit_id": best[nid][1] if nid in best else None,
            "dense_rank": dense_rank,
            "fts_rank": fts_rank,
            "filters": filter_checks(d, filters),
            "rrf": scores.get(nid),
            "authority": factors.get(nid, {}),
            "final_rank": final_rank.get(nid),
        }
        if nid in ahead:
            out["entity_first"] = True
        if not gold:
            return out
        dense_before, fts_before = before.get(nid, (None, None))
        return {**out, "gold": True, "prefilter_dense_rank": dense_before, "prefilter_fts_rank": fts_before}

    candidates = [record(nid, nid in gold_set) for nid in sorted(in_top, key=lambda n: final_rank[n])]
    candidates += [record(nid, True) for nid in gold_ids.values() if nid and nid not in in_top]
    candidates += [
        {"doc_id": None, "path": p, "gold": True, "filters": {}} for p, nid in gold_ids.items() if nid is None
    ]
    trace.set("ragnarok.candidates", candidates)
    trace.returned([(nid, None if nid in ahead else round(finals[nid], 5)) for nid in top])
