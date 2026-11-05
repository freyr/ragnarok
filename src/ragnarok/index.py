"""SQLite store: core.db (node, doc, link, Jira keys, violations, the kept tables) and index.db attached as idx
(unit, FTS5, vectors, stat)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config, rules_hash
from .embed import input_hash
from .metadata import iter_notes, read_note, resolve
from .splitter import chunk_note

SCHEMA_VERSION = 4
GROUP = 64  # notes per transaction during an update
SCAN_LIMIT = 20  # changed notes for each scan before a server call
NOT_CURRENT = "The index is not current. Run `ragnarok index`."
REBUILD = "Run `ragnarok index --rebuild`."
MIGRATE = "Run `ragnarok migrate`."

WIKI_LINK = re.compile(r"!?\[\[([^\]|#^]+)")

# the tables of core.db that a rebuild keeps: they cannot be derived from the files, or only at a high cost
KEPT_TABLES = ("embed_cache", "w0_step")
KEPT_SCHEMA = """
CREATE TABLE IF NOT EXISTS main.embed_cache (
    text_hash TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS main.w0_step (
    id INTEGER PRIMARY KEY,
    created TEXT NOT NULL,
    session_id TEXT NOT NULL,
    step_kind TEXT NOT NULL,
    summary TEXT,
    summary_hash TEXT NOT NULL,
    features TEXT,
    evaluator TEXT,
    evaluator_version TEXT,
    score REAL,
    threshold REAL,
    decided_by TEXT,
    verdict TEXT,
    mode TEXT,
    label_command INTEGER,
    label_pipeline INTEGER,
    label_audit INTEGER
);
"""

# the derived tables of core.db: a rebuild drops them and imports the files again
DERIVED_SCHEMA = """
CREATE TABLE IF NOT EXISTS main.node (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    collection TEXT NOT NULL,
    mode TEXT NOT NULL,
    provisional INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS main.doc (
    id TEXT PRIMARY KEY REFERENCES node (id) ON DELETE CASCADE ON UPDATE CASCADE,
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
    hash TEXT NOT NULL,
    declared_id TEXT
);
CREATE INDEX IF NOT EXISTS main.doc_project ON doc (vault, project);
CREATE TABLE IF NOT EXISTS main.link (
    from_doc TEXT NOT NULL REFERENCES doc (id) ON DELETE CASCADE ON UPDATE CASCADE,
    ord INTEGER NOT NULL,
    target_text TEXT NOT NULL,
    to_node TEXT,
    in_code INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (from_doc, ord)
);
CREATE INDEX IF NOT EXISTS main.link_to ON link (to_node);
CREATE TABLE IF NOT EXISTS main.jira (
    key TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES doc (id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE INDEX IF NOT EXISTS main.jira_key ON jira (key);
CREATE INDEX IF NOT EXISTS main.jira_doc ON jira (doc_id);
CREATE TABLE IF NOT EXISTS main.violation (
    id INTEGER PRIMARY KEY,
    rule TEXT NOT NULL,
    node_ids TEXT NOT NULL,
    detail TEXT,
    found_at TEXT NOT NULL
);
"""

# index.db: no foreign key or trigger crosses the two files, so unit.doc_id is a plain value
INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS idx.unit (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_hash TEXT NOT NULL,
    ord INTEGER NOT NULL,
    heading_path TEXT NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    words INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx.unit_doc ON unit (doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS idx.unit_fts USING fts5(
    heading_path, text, tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS idx.unit_emb (
    unit_id INTEGER PRIMARY KEY REFERENCES unit (id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TRIGGER IF NOT EXISTS idx.unit_delete AFTER DELETE ON unit BEGIN
    DELETE FROM unit_fts WHERE rowid = old.id;
END;
CREATE TABLE IF NOT EXISTS idx.stat (
    "column" TEXT NOT NULL,
    value TEXT NOT NULL,
    n_docs INTEGER NOT NULL,
    n_units INTEGER NOT NULL,
    updated TEXT NOT NULL,
    PRIMARY KEY ("column", value)
);
"""

META_SCHEMA = """
CREATE TABLE IF NOT EXISTS main.meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idx.meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _ddl_hash(ddl: str) -> str:
    return hashlib.sha1(ddl.encode()).hexdigest()


class IndexMismatch(RuntimeError):
    pass


def _utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _meta(conn: sqlite3.Connection, schema: str) -> dict[str, str]:
    """The rows of the meta table of the schema, or {} when it has no meta table."""
    if conn.execute(f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone() is None:
        return {}
    return dict(conn.execute(f"SELECT key, value FROM {schema}.meta").fetchall())


def _is_v1(path: Path) -> bool:
    """True when the file is an index of schema version 1: one file with the embedding cache in it."""
    conn = _read_only(path)
    try:
        has_cache = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'embed_cache'").fetchone()
        return has_cache is not None and _meta(conn, "main").get("schema_version") == "1"
    except sqlite3.DatabaseError:
        return False
    finally:
        conn.close()


def connect_core(cfg: Config) -> sqlite3.Connection:
    """Open core.db only and make its kept tables. It never reads or writes the hash of the derived DDL."""
    cfg.core_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cfg.core_path.resolve().as_uri(), uri=True, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(KEPT_SCHEMA)
    return conn


def import_cache(cfg: Config, source) -> int:
    """Copy the embed_cache rows of another file (a version 1 index or a core.db) into core.db.

    Skip a row that exists. Return the count.
    """
    conn = connect_core(cfg)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (f"{Path(source).resolve().as_uri()}?mode=ro",))
        before = conn.execute("SELECT count(*) FROM main.embed_cache").fetchone()[0]
        conn.execute("INSERT OR IGNORE INTO main.embed_cache SELECT text_hash, model, vector FROM src.embed_cache")
        conn.commit()
        return conn.execute("SELECT count(*) FROM main.embed_cache").fetchone()[0] - before
    finally:
        conn.close()


def connect(cfg: Config) -> sqlite3.Connection:
    """Open core.db, attach index.db as idx, create the schema, and check the versions, the DDL and the model."""
    if not cfg.core_path.exists() and cfg.index_path.exists():
        # an index with no core file: check before any write, so that the file does not change
        hint = MIGRATE if _is_v1(cfg.index_path) else REBUILD
        raise IndexMismatch(f"The index at {cfg.index_path} has no core file {cfg.core_path}. {hint}")
    cfg.core_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.index_path.parent.mkdir(parents=True, exist_ok=True)
    # the MCP server runs each call in a worker thread; it serializes the calls with a lock
    conn = sqlite3.connect(cfg.core_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("ATTACH DATABASE ? AS idx", (str(cfg.index_path),))
    conn.execute("PRAGMA idx.journal_mode = WAL")
    conn.execute("PRAGMA idx.synchronous = NORMAL")
    core = {"schema_version": str(SCHEMA_VERSION), "derived_ddl": _ddl_hash(DERIVED_SCHEMA)}
    index = {"model": cfg.model, "dimension": str(cfg.dimension), "index_ddl": _ddl_hash(INDEX_SCHEMA)}
    stored_core, stored_index = _meta(conn, "main"), _meta(conn, "idx")
    # check before the DDL, so a file of another schema gets no new tables
    for path, stored, expected in ((cfg.core_path, stored_core, core), (cfg.index_path, stored_index, index)):
        if stored and any(stored.get(k) != v for k, v in expected.items()):
            conn.close()
            found = {k: stored.get(k) for k in expected}
            raise IndexMismatch(f"The file {path} uses {found}, the configuration needs {expected}. {REBUILD}")
    conn.executescript(META_SCHEMA + KEPT_SCHEMA + DERIVED_SCHEMA + INDEX_SCHEMA)
    if not stored_core:
        rows = [*core.items(), ("rules_hash", rules_hash(cfg))]
        conn.executemany("INSERT INTO main.meta (key, value) VALUES (?, ?)", rows)
    if not stored_index:
        conn.executemany("INSERT INTO idx.meta (key, value) VALUES (?, ?)", list(index.items()))
    conn.commit()
    return conn


def _delete_index_file(cfg: Config) -> None:
    for suffix in ("", "-wal", "-shm"):
        cfg.index_path.with_name(cfg.index_path.name + suffix).unlink(missing_ok=True)


def rebuild(cfg: Config) -> sqlite3.Connection:
    """Delete index.db, drop each table of core.db that is not kept, and open both files again. Never delete core.db.

    A table of an earlier layout of core.db is dropped too. The update after this call imports the files.
    """
    if cfg.index_path.exists() and _is_v1(cfg.index_path):
        raise IndexMismatch(f"The index at {cfg.index_path} holds the embedding cache of schema version 1. {MIGRATE}")
    _delete_index_file(cfg)
    if cfg.core_path.exists():
        conn = sqlite3.connect(cfg.core_path, timeout=30)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            names = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
            for name in [n for (n,) in names.fetchall() if n not in KEPT_TABLES]:
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
            conn.commit()
        finally:
            conn.close()
    return connect(cfg)


def migrate(cfg: Config) -> tuple[Path, int]:
    """Move a version 1 index into core.db and a new index.db with no new embedding.

    Back up the old file, copy its embedding cache into core.db, delete it after that commit, and make the new
    files. Return the backup path and the number of cache rows copied.
    """
    if not (cfg.index_path.exists() and _is_v1(cfg.index_path)):
        raise IndexMismatch(f"No index of schema version 1 at {cfg.index_path}.")
    target = cfg.data_dir / "backups" / f"index-v1-{_utc()}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(cfg.index_path, timeout=30)
    try:
        src.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dst = sqlite3.connect(target)
        src.backup(dst)
        dst.close()
    finally:
        src.close()
    copied = import_cache(cfg, cfg.index_path)
    _delete_index_file(cfg)
    rebuild(cfg).close()
    return target, copied


def backup(cfg: Config, to: Path | None = None) -> Path:
    """Copy core.db with the SQLite backup API, also while another connection writes to it. Return the path."""
    target = to or cfg.data_dir / "backups" / f"core-{_utc()}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(cfg.core_path, timeout=30)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def _unit_vector(blob: bytes) -> bytes:
    """The float32 vector at length 1, so that a dot product is the cosine."""
    v = np.frombuffer(blob, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return (v / norm if norm else v).astype(np.float32).tobytes()


def _hash_file(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def provisional_id(rel: str) -> str:
    """The ID of a note with no frontmatter `id`: `~` and the first 12 hex characters of the SHA-1 of its path."""
    return "~" + hashlib.sha1(rel.encode()).hexdigest()[:12]


def _frontmatter_id(fm: dict) -> str | None:
    value = fm.get("id")
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    return str(value).strip() or None


def _delete_note(conn: sqlite3.Connection, path: str) -> None:
    """Delete the doc at the path, with its node, units, links and Jira keys."""
    row = conn.execute("SELECT id FROM doc WHERE path = ?", (path,)).fetchone()
    if row is None:
        return
    doc_id = row[0]
    # the foreign key and the trigger of idx.unit delete the vectors and the FTS rows
    conn.execute("DELETE FROM unit WHERE doc_id = ?", (doc_id,))
    conn.execute("UPDATE link SET to_node = NULL WHERE to_node = ?", (doc_id,))
    # the foreign keys delete the doc, its links and its Jira keys
    conn.execute("DELETE FROM node WHERE id = ?", (doc_id,))


@dataclass
class UpdateResult:
    notes: int = 0  # notes on disk
    added: int = 0
    changed: int = 0
    moved: int = 0  # same content at a new path
    removed: int = 0
    touched: int = 0  # time or size changed, content did not
    rederived: int = 0  # same content, new metadata or units after a change of the rules
    deferred: int = 0  # changed notes left for a later update because of the limit
    reconciled: int = 0  # docs whose units were split again because their hash did not agree
    chunks: int = 0  # units written
    embedded: int = 0  # units sent to the model
    redactions: int = 0
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Pending:
    rel: str
    old_path: str | None  # the path of the doc that this note replaces
    frontmatter_id: str | None
    mtime_ns: int
    size: int
    hash: str
    meta: object
    chunks: list
    links: list[str]
    moved: bool = False
    rederived: bool = False


def _same(cfg: Config, conn: sqlite3.Connection, doc_id: str, m, chunks) -> bool:
    """True when the index already holds this metadata and these units for the note."""
    row = conn.execute(
        "SELECT title, project, area, type, status, sensitivity, description FROM doc WHERE id = ?", (doc_id,)
    ).fetchone()
    if row != (m.title, m.project, m.area, m.type, m.status, m.sensitivity, m.description):
        return False
    jira = [k for (k,) in conn.execute("SELECT key FROM jira WHERE doc_id = ? ORDER BY key", (doc_id,))]
    stored = [h for (h,) in conn.execute("SELECT text_hash FROM unit WHERE doc_id = ? ORDER BY ord", (doc_id,))]
    return jira == m.jira and stored == [input_hash(cfg, c.heading, c.text) for c in chunks]


def update(
    cfg: Config,
    conn: sqlite3.Connection,
    embedder,
    progress: Callable[[str], None] | None = None,
    limit: int | None = None,
) -> UpdateResult:
    """Bring the index in line with the vaults: add, change, move and remove notes. Embed only uncached units.

    With a limit, rewrite at most `limit` changed notes, newest first, and warn that the index is not current.
    The other changed notes keep their old time in the index, so the next update finds them again.

    When the metadata rules of the configuration changed, derive the metadata of all notes again, with no limit,
    and rewrite the notes whose metadata or units changed. The embedding cache keeps the unchanged units.
    When the scan adds, removes or changes a project marker file, do the same for the notes under the folder of
    that file.
    """
    start = time.monotonic()
    result = UpdateResult()
    rules = rules_hash(cfg)
    rederive = conn.execute("SELECT value FROM main.meta WHERE key = 'rules_hash'").fetchone() != (rules,)
    if rederive:
        limit = None
    known = {
        path: (doc_id, mtime, size, h)
        for doc_id, path, mtime, size, h in conn.execute("SELECT id, path, mtime_ns, size, hash FROM doc")
    }
    on_disk = {}
    for rel in iter_notes(cfg):
        st = cfg.note_path(rel).stat()
        on_disk[rel] = (st.st_mtime_ns, st.st_size)
    result.notes = len(on_disk)

    gone = {rel: known[rel] for rel in known.keys() - on_disk.keys()}
    gone_by_hash = {h: rel for rel, (_, _, _, h) in gone.items()}
    # a marker file that appears or disappears (a move does both) changes the project of the notes below its folder,
    # and an edited marker file can change their status (a SUMMARY.md with state superseded)
    changed = {rel for rel in on_disk.keys() & known.keys() if known[rel][1:3] != on_disk[rel]}
    marker_dirs = tuple(
        rel.rpartition("/")[0] + "/"
        for rel in (on_disk.keys() - known.keys()) | gone.keys() | changed
        if rel.rpartition("/")[2] in cfg.project_markers
    )
    todo = sorted(
        (
            rel
            for rel, stat in on_disk.items()
            if rel not in known or rederive or known[rel][1:3] != stat or rel.startswith(marker_dirs)
        ),
        key=lambda rel: (-on_disk[rel][0], rel),
    )
    pending: list[_Pending] = []
    contested: list[str] = []  # notes whose frontmatter id another doc held when they were written
    for i, rel in enumerate(todo):
        mtime_ns, size = on_disk[rel]
        data = cfg.note_path(rel).read_bytes()
        h = _hash_file(data)
        old = known.get(rel)
        if old and old[3] == h:
            if old[1:3] != (mtime_ns, size):
                conn.execute("UPDATE doc SET mtime_ns = ?, size = ? WHERE id = ?", (mtime_ns, size, old[0]))
                result.touched += 1
            if not (rederive or rel.startswith(marker_dirs)):
                continue
            fm, body = read_note(cfg.note_path(rel))
            meta = resolve(cfg, rel, fm, body)
            chunks, _ = chunk_note(cfg, meta, body)
            if _same(cfg, conn, old[0], meta, chunks):
                continue
            links = sorted({m.group(1).strip() for m in WIKI_LINK.finditer(body)})
            fid = _frontmatter_id(fm)
            pending.append(_Pending(rel, rel, fid, mtime_ns, size, h, meta, chunks, links, rederived=True))
        else:
            # a re-derived note does not count toward the limit, also while it waits in pending
            written = result.added + result.changed + result.moved + sum(not p.rederived for p in pending)
            if limit is not None and written >= limit:
                result.deferred += 1
                continue
            old_path = rel if old else None
            moved_from = None if old else gone_by_hash.pop(h, None)
            if moved_from:
                # a move: rewrite the old note at the new path; the metadata and a provisional ID change with the path
                gone.pop(moved_from)
                old_path = moved_from
            fm, body = read_note(cfg.note_path(rel))
            meta = resolve(cfg, rel, fm, body)
            chunks, redactions = chunk_note(cfg, meta, body)
            result.redactions += redactions
            links = sorted({m.group(1).strip() for m in WIKI_LINK.finditer(body)})
            fid = _frontmatter_id(fm)
            pending.append(_Pending(rel, old_path, fid, mtime_ns, size, h, meta, chunks, links, moved=bool(moved_from)))
        if len(pending) >= GROUP:
            _write(cfg, conn, embedder, pending, result, on_disk.keys(), contested)
            conn.commit()
            pending = []
            if progress:
                progress(f"{i + 1}/{len(todo)} notes, {result.embedded} units embedded")
    if pending:
        _write(cfg, conn, embedder, pending, result, on_disk.keys(), contested)
    for rel in gone:
        _delete_note(conn, rel)
        result.removed += 1
    claimed = _claim_free_ids(conn)
    _warn_duplicate_ids(conn, contested, result)
    if result.added or result.changed or result.moved or result.removed or result.rederived or claimed:
        resolve_links(conn)
    if rederive:
        conn.execute("INSERT OR REPLACE INTO main.meta (key, value) VALUES ('rules_hash', ?)", (rules,))
    conn.commit()
    result.reconciled = reconcile(cfg, conn, embedder, result)
    if result.deferred:
        result.warnings.append(NOT_CURRENT)
    result.seconds = time.monotonic() - start
    return result


def _claim_id(conn: sqlite3.Connection, p: _Pending, present, contested: list[str]) -> tuple[str, bool]:
    """The ID of the note and True when it is provisional.

    The frontmatter `id` is the ID. A doc that holds that ID at a path that is not a note of this scan is the same
    note at a new path, so it is deleted. A doc that holds it at a path of this scan can still give the ID up later
    in the update: this note gets its provisional ID for now, and `_claim_free_ids()` tries again at the end.
    """
    if p.frontmatter_id is None:
        return provisional_id(p.rel), True
    row = conn.execute("SELECT path FROM doc WHERE id = ?", (p.frontmatter_id,)).fetchone()
    if row is not None:
        if row[0] in present:
            contested.append(p.rel)
            return provisional_id(p.rel), True
        _delete_note(conn, row[0])
    return p.frontmatter_id, False


def _claim_free_ids(conn: sqlite3.Connection) -> int:
    """Give each provisional doc the ID that its frontmatter declares, when no doc holds that ID. Return the count.

    When two docs declare the same free ID, the first path gets it.
    """
    rows = conn.execute(
        "SELECT d.id, d.declared_id FROM doc d JOIN node n ON n.id = d.id"
        " WHERE n.provisional = 1 AND d.declared_id IS NOT NULL"
        " AND d.declared_id NOT IN (SELECT id FROM doc) ORDER BY d.path"
    ).fetchall()
    taken: set[str] = set()
    for old, new in rows:
        if new in taken:
            continue
        taken.add(new)
        # ON UPDATE CASCADE moves doc.id, link.from_doc and jira.doc_id; no foreign key reaches index.db
        conn.execute("UPDATE node SET id = ?, provisional = 0 WHERE id = ?", (new, old))
        conn.execute("UPDATE unit SET doc_id = ? WHERE doc_id = ?", (new, old))
        conn.execute("UPDATE unit_emb SET doc_id = ? WHERE doc_id = ?", (new, old))
        conn.execute("UPDATE link SET to_node = ? WHERE to_node = ?", (new, old))
    return len(taken)


def _warn_duplicate_ids(conn: sqlite3.Connection, contested: list[str], result: UpdateResult) -> None:
    """Warn for each note of this update that still has a provisional ID because another note holds its ID."""
    for rel in contested:
        row = conn.execute(
            "SELECT d.declared_id, h.path FROM doc d JOIN node n ON n.id = d.id JOIN doc h ON h.id = d.declared_id"
            " WHERE d.path = ? AND n.provisional = 1",
            (rel,),
        ).fetchone()
        if row:
            result.warnings.append(f"{rel} has the id {row[0]} of {row[1]}; it gets a provisional ID")


def _embed_missing(cfg: Config, conn: sqlite3.Connection, embedder, chunks) -> int:
    """Embed the units that are not in the cache, in one call. Return the count."""
    hashes = {input_hash(cfg, c.heading, c.text): (c.heading, c.text) for c in chunks}
    cached = set()
    keys = list(hashes)
    for i in range(0, len(keys), 500):
        part = keys[i : i + 500]
        # the cache outlives a rebuild, so a vector of another dimension (same model, other truncation) is a miss
        marks = ",".join("?" * len(part))
        q = f"SELECT text_hash FROM embed_cache WHERE length(vector) = ? AND text_hash IN ({marks})"
        cached.update(r[0] for r in conn.execute(q, [cfg.dimension * 4, *part]))
    missing = [k for k in keys if k not in cached]
    if missing:
        vectors = embedder.embed_docs([hashes[k] for k in missing])
        conn.executemany(
            "INSERT OR REPLACE INTO embed_cache (text_hash, model, vector) VALUES (?, ?, ?)",
            [(k, cfg.model, np.asarray(v, dtype=np.float32).tobytes()) for k, v in zip(missing, vectors, strict=True)],
        )
    return len(missing)


def _write_units(cfg: Config, conn: sqlite3.Connection, doc_id: str, doc_hash: str, chunks) -> int:
    """Write the units of a doc with their FTS rows and their vectors from the cache. Return the count."""
    for c in chunks:
        th = input_hash(cfg, c.heading, c.text)
        uid = conn.execute(
            "INSERT INTO unit (doc_id, doc_hash, ord, heading_path, text, text_hash, words)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, doc_hash, c.ord, c.heading, c.text, th, len(c.text.split())),
        ).lastrowid
        conn.execute("INSERT INTO unit_fts (rowid, heading_path, text) VALUES (?, ?, ?)", (uid, c.heading, c.text))
        row = conn.execute(
            "SELECT vector FROM embed_cache WHERE text_hash = ? AND length(vector) = ?",
            (th, cfg.dimension * 4),
        ).fetchone()
        if row is not None:
            conn.execute(
                "INSERT INTO unit_emb (unit_id, doc_id, embedding) VALUES (?, ?, ?)",
                (uid, doc_id, _unit_vector(row[0])),
            )
    return len(chunks)


def _write(
    cfg: Config,
    conn: sqlite3.Connection,
    embedder,
    pending: list[_Pending],
    result: UpdateResult,
    present,
    contested: list[str],
) -> None:
    result.embedded += _embed_missing(cfg, conn, embedder, [c for p in pending for c in p.chunks])
    for p in pending:
        m = p.meta
        if p.old_path is not None:
            _delete_note(conn, p.old_path)
            if p.rederived:
                result.rederived += 1
            elif p.moved:
                result.moved += 1
            else:
                result.changed += 1
        else:
            result.added += 1
        # another writer (the scan of the MCP server before each call) can index the path after this update read
        # its snapshot of the notes
        _delete_note(conn, p.rel)
        doc_id, provisional = _claim_id(conn, p, present, contested)
        conn.execute(
            "INSERT INTO node (id, node_type, collection, mode, provisional) VALUES (?, 'doc', ?, 'files', ?)",
            (doc_id, m.vault, int(provisional)),
        )
        conn.execute(
            "INSERT INTO doc (id, vault, path, title, project, area, type, status, sensitivity, description,"
            " mtime_ns, size, hash, declared_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
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
                p.frontmatter_id,
            ),
        )
        result.chunks += _write_units(cfg, conn, doc_id, p.hash, p.chunks)
        conn.executemany(
            "INSERT INTO link (from_doc, ord, target_text) VALUES (?, ?, ?)",
            [(doc_id, i, t) for i, t in enumerate(p.links)],
        )
        conn.executemany("INSERT INTO jira (key, doc_id) VALUES (?, ?)", [(k, doc_id) for k in m.jira])


def reconcile(cfg: Config, conn: sqlite3.Connection, embedder, result: UpdateResult | None = None) -> int:
    """Split again each doc whose units were built from another hash, or that has no units. Return the count.

    The two files are not atomic together, so core.db is the truth: a unit carries the `doc_hash` it was built
    from. A doc whose file no longer has `doc.hash` is left to the next update. A note with no units (an empty
    note) is read again but not counted. A unit of a doc that does not exist is deleted: another index file that
    shares core.db (a --min-words variant) was not open when the doc was deleted or got its frontmatter ID.
    """
    orphans = "FROM unit WHERE doc_id NOT IN (SELECT id FROM doc)"
    if conn.execute(f"SELECT 1 {orphans} LIMIT 1").fetchone():  # a write only when there is one, as the scan is hot
        conn.execute(f"DELETE {orphans}")
        conn.commit()
    rows = conn.execute(
        "SELECT d.id, d.path, d.hash FROM doc d"
        " WHERE NOT EXISTS (SELECT 1 FROM unit u WHERE u.doc_id = d.id)"
        " OR EXISTS (SELECT 1 FROM unit u WHERE u.doc_id = d.id AND u.doc_hash != d.hash)"
    ).fetchall()
    todo = []
    for doc_id, rel, h in rows:
        path = cfg.note_path(rel)
        try:
            same = _hash_file(path.read_bytes()) == h
        except OSError:
            continue
        if not same:
            continue
        fm, body = read_note(path)
        chunks, _ = chunk_note(cfg, resolve(cfg, rel, fm, body), body)
        if chunks or conn.execute("SELECT 1 FROM unit WHERE doc_id = ?", (doc_id,)).fetchone():
            todo.append((doc_id, h, chunks))
    for i in range(0, len(todo), GROUP):
        group = todo[i : i + GROUP]
        embedded = _embed_missing(cfg, conn, embedder, [c for _, _, chunks in group for c in chunks])
        written = 0
        for doc_id, h, chunks in group:
            conn.execute("DELETE FROM unit WHERE doc_id = ?", (doc_id,))
            written += _write_units(cfg, conn, doc_id, h, chunks)
        conn.commit()
        if result is not None:
            result.embedded += embedded
            result.chunks += written
    return len(todo)


def _link_key(path: str) -> str:
    path = path.lower()
    return path[:-3] if path.endswith(".md") else path


def _common_folders(folders: list[str], path: str) -> int:
    """The number of leading folders that the list and the path share."""
    count = 0
    for a, b in zip(folders, path.split("/")[:-1], strict=False):
        if a != b:
            break
        count += 1
    return count


def resolve_links(conn: sqlite3.Connection) -> None:
    """Resolve each wiki-link target to a doc node.

    Rank the candidates by the same vault as the note with the link, then the most folders in common with that
    note, then the shortest path.
    """
    docs = [(doc_id, vault, _link_key(path)) for doc_id, vault, path in conn.execute("SELECT id, vault, path FROM doc")]
    by_name: dict[str, list[tuple[str, str, str]]] = {}
    for n in docs:
        by_name.setdefault(n[2].rsplit("/", 1)[-1], []).append(n)
    vault_of = {doc_id: vault for doc_id, vault, _ in docs}
    path_of = {doc_id: path for doc_id, _, path in docs}
    updates = []
    for from_doc, ord_, target in conn.execute("SELECT from_doc, ord, target_text FROM link").fetchall():
        key = _link_key(target).strip("/")
        candidates = [n for n in by_name.get(key.rsplit("/", 1)[-1], []) if n[2].endswith(key)]
        if not candidates:
            to_node = None
        else:
            home = vault_of.get(from_doc)
            folders = path_of.get(from_doc, "").split("/")[:-1]
            to_node = min(candidates, key=lambda n: (n[1] != home, -_common_folders(folders, n[2]), len(n[2])))[0]
        updates.append((to_node, from_doc, ord_))
    conn.executemany("UPDATE link SET to_node = ? WHERE from_doc = ? AND ord = ?", updates)


def status(cfg: Config, conn: sqlite3.Connection) -> dict:
    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    size = lambda path: round(path.stat().st_size / 1e6, 1) if path.exists() else 0  # noqa: E731
    return {
        "core": str(cfg.core_path),
        "core_size_mb": size(cfg.core_path),
        "index": str(cfg.index_path),
        "size_mb": size(cfg.index_path),
        "model": one("SELECT value FROM idx.meta WHERE key = 'model'"),
        "dimension": int(one("SELECT value FROM idx.meta WHERE key = 'dimension'")),
        "rules_hash": (conn.execute("SELECT value FROM main.meta WHERE key = 'rules_hash'").fetchone() or [None])[0],
        "notes": one("SELECT count(*) FROM doc"),
        "private_notes": one("SELECT count(*) FROM doc WHERE sensitivity = 'private'"),
        "archived_notes": one("SELECT count(*) FROM doc WHERE status = 'archived'"),
        "chunks": one("SELECT count(*) FROM unit"),
        "vectors": one("SELECT count(*) FROM unit_emb"),
        "cache": one("SELECT count(*) FROM embed_cache"),
        "w0_steps": one("SELECT count(*) FROM w0_step"),
        "violations": one("SELECT count(*) FROM violation"),
        "links": one("SELECT count(*) FROM link"),
        "resolved_links": one("SELECT count(*) FROM link WHERE to_node IS NOT NULL"),
        "jira_keys": one("SELECT count(DISTINCT key) FROM jira"),
        "vaults": dict(conn.execute("SELECT vault, count(*) FROM doc GROUP BY vault").fetchall()),
    }
