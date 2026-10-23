"""SQLite index: notes, chunks, FTS5, sqlite-vec, embedding cache, links and Jira keys."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import sqlite_vec

from .config import Config
from .embed import input_hash
from .metadata import iter_notes, read_note, resolve
from .splitter import chunk_note

SCHEMA_VERSION = 1
GROUP = 64  # notes per transaction during an update
SCAN_LIMIT = 20  # changed notes for each scan before a server call
NOT_CURRENT = "The index is not current. Run `ragnarok index`."

WIKI_LINK = re.compile(r"!?\[\[([^\]|#^]+)")


def _schema(dimension: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY,
    vault TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    project TEXT NOT NULL,
    area TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    description TEXT,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_project ON notes (vault, project);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    note_id INTEGER NOT NULL REFERENCES notes (id),
    ord INTEGER NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_note ON chunks (note_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading, text, tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding float[{dimension}] distance_metric=cosine,
    vault TEXT,
    sensitivity TEXT,
    status TEXT
);
CREATE TABLE IF NOT EXISTS embed_cache (text_hash TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS links (from_note INTEGER NOT NULL, target TEXT NOT NULL, to_note INTEGER);
CREATE INDEX IF NOT EXISTS links_from ON links (from_note);
CREATE INDEX IF NOT EXISTS links_to ON links (to_note);
CREATE TABLE IF NOT EXISTS jira (key TEXT NOT NULL, note_id INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS jira_key ON jira (key);
"""


class IndexMismatch(RuntimeError):
    pass


def connect(cfg: Config) -> sqlite3.Connection:
    """Open the index, create the schema, and check that the model and the dimension match."""
    cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
    # the MCP server runs each call in a worker thread; it serializes the calls with a lock
    conn = sqlite3.connect(cfg.index_path, timeout=30, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(_schema(cfg.dimension))
    expected = {"model": cfg.model, "dimension": str(cfg.dimension), "schema_version": str(SCHEMA_VERSION)}
    stored = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    if not stored:
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", expected.items())
        conn.commit()
    elif any(stored.get(k) != v for k, v in expected.items()):
        conn.close()
        raise IndexMismatch(
            f"The index at {cfg.index_path} uses {stored}, the configuration needs {expected}. "
            "Run `ragnarok index --rebuild`."
        )
    return conn


def _hash_file(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _delete_note(conn: sqlite3.Connection, note_id: int) -> None:
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE note_id = ?", (note_id,))]
    conn.executemany("DELETE FROM chunks_vec WHERE chunk_id = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
    conn.execute("DELETE FROM links WHERE from_note = ?", (note_id,))
    conn.execute("UPDATE links SET to_note = NULL WHERE to_note = ?", (note_id,))
    conn.execute("DELETE FROM jira WHERE note_id = ?", (note_id,))
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))


@dataclass
class UpdateResult:
    notes: int = 0  # notes on disk
    added: int = 0
    changed: int = 0
    moved: int = 0  # same content at a new path
    removed: int = 0
    touched: int = 0  # time or size changed, content did not
    deferred: int = 0  # changed notes left for a later update because of the limit
    chunks: int = 0  # chunks written
    embedded: int = 0  # chunks sent to the model
    redactions: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Pending:
    rel: str
    note_id: int | None
    mtime_ns: int
    size: int
    hash: str
    meta: object
    chunks: list
    links: list[str]
    moved: bool = False


def update(
    cfg: Config,
    conn: sqlite3.Connection,
    embedder,
    progress: Callable[[str], None] | None = None,
    limit: int | None = None,
) -> UpdateResult:
    """Bring the index in line with the vaults: add, change, move and remove notes. Embed only uncached chunks.

    With a limit, rewrite at most `limit` changed notes, newest first, and warn that the index is not current.
    The other changed notes keep their old time in the index, so the next update finds them again.
    """
    start = time.monotonic()
    result = UpdateResult()
    known = {
        path: (nid, mtime, size, h)
        for nid, path, mtime, size, h in conn.execute("SELECT id, path, mtime_ns, size, hash FROM notes")
    }
    on_disk = {}
    for rel in iter_notes(cfg):
        st = (cfg.root / rel).stat()
        on_disk[rel] = (st.st_mtime_ns, st.st_size)
    result.notes = len(on_disk)

    gone = {rel: known[rel] for rel in known.keys() - on_disk.keys()}
    gone_by_hash = {h: rel for rel, (_, _, _, h) in gone.items()}
    todo = sorted(
        (rel for rel, stat in on_disk.items() if rel not in known or known[rel][1:3] != stat),
        key=lambda rel: (-on_disk[rel][0], rel),
    )
    pending: list[_Pending] = []
    for i, rel in enumerate(todo):
        mtime_ns, size = on_disk[rel]
        data = (cfg.root / rel).read_bytes()
        h = _hash_file(data)
        old = known.get(rel)
        if old and old[3] == h:
            conn.execute("UPDATE notes SET mtime_ns = ?, size = ? WHERE id = ?", (mtime_ns, size, old[0]))
            result.touched += 1
            continue
        if limit is not None and result.added + result.changed + result.moved + len(pending) >= limit:
            result.deferred += 1
            continue
        note_id = old[0] if old else None
        moved_from = None if old else gone_by_hash.pop(h, None)
        if moved_from:
            # a move: rewrite the old note at the new path, the metadata can change with the path
            note_id = gone.pop(moved_from)[0]
        fm, body = read_note(cfg.root / rel)
        meta = resolve(cfg, rel, fm, body)
        chunks, redactions = chunk_note(cfg, meta, body)
        result.redactions += redactions
        links = sorted({m.group(1).strip() for m in WIKI_LINK.finditer(body)})
        pending.append(_Pending(rel, note_id, mtime_ns, size, h, meta, chunks, links, moved=bool(moved_from)))
        if len(pending) >= GROUP:
            _write(cfg, conn, embedder, pending, result)
            conn.commit()
            pending = []
            if progress:
                progress(f"{i + 1}/{len(todo)} notes, {result.embedded} chunks embedded")
    if pending:
        _write(cfg, conn, embedder, pending, result)
    for _, (nid, *_) in gone.items():
        _delete_note(conn, nid)
        result.removed += 1
    if result.added or result.changed or result.moved or result.removed:
        resolve_links(conn)
    conn.commit()
    if result.deferred:
        result.warnings.append(NOT_CURRENT)
    result.seconds = time.monotonic() - start
    return result


def _write(cfg: Config, conn: sqlite3.Connection, embedder, pending: list[_Pending], result: UpdateResult) -> None:
    # embed the chunks that are not in the cache, in one call for the group
    hashes = {}
    for p in pending:
        for c in p.chunks:
            hashes[input_hash(cfg, c.heading, c.text)] = (c.heading, c.text)
    cached = set()
    keys = list(hashes)
    for i in range(0, len(keys), 500):
        part = keys[i : i + 500]
        q = f"SELECT text_hash FROM embed_cache WHERE text_hash IN ({','.join('?' * len(part))})"
        cached.update(r[0] for r in conn.execute(q, part))
    missing = [k for k in keys if k not in cached]
    if missing:
        vectors = embedder.embed_docs([hashes[k] for k in missing])
        conn.executemany(
            "INSERT OR REPLACE INTO embed_cache (text_hash, model, vector) VALUES (?, ?, ?)",
            [(k, cfg.model, sqlite_vec.serialize_float32(v)) for k, v in zip(missing, vectors, strict=True)],
        )
        result.embedded += len(missing)

    for p in pending:
        m = p.meta
        if p.note_id is not None:
            _delete_note(conn, p.note_id)
            if p.moved:
                result.moved += 1
            else:
                result.changed += 1
        else:
            result.added += 1
        cur = conn.execute(
            "INSERT INTO notes (vault, path, title, project, area, type, status, sensitivity, description,"
            " mtime_ns, size, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m.vault,
                p.rel,
                m.title,
                m.project,
                m.area,
                m.type,
                m.status,
                m.sensitivity,
                m.description,
                p.mtime_ns,
                p.size,
                p.hash,
            ),
        )
        note_id = cur.lastrowid
        for c in p.chunks:
            th = input_hash(cfg, c.heading, c.text)
            cid = conn.execute(
                "INSERT INTO chunks (note_id, ord, heading, text, text_hash) VALUES (?, ?, ?, ?, ?)",
                (note_id, c.ord, c.heading, c.text, th),
            ).lastrowid
            conn.execute("INSERT INTO chunks_fts (rowid, heading, text) VALUES (?, ?, ?)", (cid, c.heading, c.text))
            conn.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding, vault, sensitivity, status)"
                " SELECT ?, vector, ?, ?, ? FROM embed_cache WHERE text_hash = ?",
                (cid, m.vault, m.sensitivity, m.status, th),
            )
            result.chunks += 1
        conn.executemany("INSERT INTO links (from_note, target) VALUES (?, ?)", [(note_id, t) for t in p.links])
        conn.executemany("INSERT INTO jira (key, note_id) VALUES (?, ?)", [(k, note_id) for k in m.jira])


def _link_key(path: str) -> str:
    path = path.lower()
    return path[:-3] if path.endswith(".md") else path


def resolve_links(conn: sqlite3.Connection) -> None:
    """Resolve each wiki-link target to a note: same vault first, then the shortest path."""
    notes = [(nid, vault, _link_key(path)) for nid, vault, path in conn.execute("SELECT id, vault, path FROM notes")]
    by_name: dict[str, list[tuple[int, str, str]]] = {}
    for n in notes:
        by_name.setdefault(n[2].rsplit("/", 1)[-1], []).append(n)
    vault_of = {nid: vault for nid, vault, _ in notes}
    updates = []
    for rowid, from_note, target in conn.execute("SELECT rowid, from_note, target FROM links").fetchall():
        key = _link_key(target).strip("/")
        candidates = [n for n in by_name.get(key.rsplit("/", 1)[-1], []) if n[2].endswith(key)]
        if not candidates:
            to_note = None
        else:
            home = vault_of.get(from_note)
            to_note = min(candidates, key=lambda n: (n[1] != home, len(n[2])))[0]
        updates.append((to_note, rowid))
    conn.executemany("UPDATE links SET to_note = ? WHERE rowid = ?", updates)


def status(cfg: Config, conn: sqlite3.Connection) -> dict:
    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    return {
        "index": str(cfg.index_path),
        "size_mb": round(cfg.index_path.stat().st_size / 1e6, 1) if cfg.index_path.exists() else 0,
        "model": one("SELECT value FROM meta WHERE key = 'model'"),
        "dimension": int(one("SELECT value FROM meta WHERE key = 'dimension'")),
        "notes": one("SELECT count(*) FROM notes"),
        "private_notes": one("SELECT count(*) FROM notes WHERE sensitivity = 'private'"),
        "archived_notes": one("SELECT count(*) FROM notes WHERE status = 'archived'"),
        "chunks": one("SELECT count(*) FROM chunks"),
        "vectors": one("SELECT count(*) FROM chunks_vec"),
        "cache": one("SELECT count(*) FROM embed_cache"),
        "links": one("SELECT count(*) FROM links"),
        "resolved_links": one("SELECT count(*) FROM links WHERE to_note IS NOT NULL"),
        "jira_keys": one("SELECT count(DISTINCT key) FROM jira"),
        "vaults": dict(conn.execute("SELECT vault, count(*) FROM notes GROUP BY vault").fetchall()),
    }
