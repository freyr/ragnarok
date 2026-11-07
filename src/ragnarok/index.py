"""SQLite index: notes, chunks, FTS5, sqlite-vec, embedding cache, links and Jira keys."""

from __future__ import annotations

import hashlib
import posixpath
import re
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import sqlite_vec

from .config import Config, rules_hash
from .embed import input_hash
from .metadata import iter_notes, link_targets, normalize_alias, read_note, resolve
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
    hash TEXT NOT NULL,
    note_class TEXT,
    validity TEXT,
    as_of TEXT,
    superseded_by TEXT,
    subject TEXT
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
    _add_kb_columns(conn)
    _add_kb_tables(conn)
    expected = {"model": cfg.model, "dimension": str(cfg.dimension), "schema_version": str(SCHEMA_VERSION)}
    stored = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    if not stored:
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)", [*expected.items(), ("rules_hash", rules_hash(cfg))]
        )
        conn.commit()
    elif any(stored.get(k) != v for k, v in expected.items()):
        conn.close()
        raise IndexMismatch(
            f"The index at {cfg.index_path} uses {stored}, the configuration needs {expected}. "
            "Run `ragnarok index --rebuild`."
        )
    return conn


KB_COLUMNS = ("note_class", "validity", "as_of", "superseded_by", "subject")


def _add_kb_columns(conn: sqlite3.Connection) -> None:
    """Add the knowledge-base label columns to an older index. The rules hash is removed, so the next update
    derives the metadata of all notes again from the embedding cache, with no new embedding."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(notes)")}
    missing = [c for c in KB_COLUMNS if c not in have]
    for column in missing:
        conn.execute(f"ALTER TABLE notes ADD COLUMN {column} TEXT")
    if missing:
        conn.execute("DELETE FROM meta WHERE key = 'rules_hash'")
        conn.commit()


# the entity notes and the references of the notes; derived from the frontmatter like `links` and `jira`
KB_TABLES = """
CREATE TABLE IF NOT EXISTS entities (note_id INTEGER PRIMARY KEY, kind TEXT);
CREATE TABLE IF NOT EXISTS entity_aliases (note_id INTEGER NOT NULL, alias TEXT NOT NULL, norm TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS entity_aliases_note ON entity_aliases (note_id);
CREATE TABLE IF NOT EXISTS refs (note_id INTEGER NOT NULL, key TEXT NOT NULL, target TEXT NOT NULL, to_note INTEGER);
CREATE INDEX IF NOT EXISTS refs_note ON refs (note_id);
CREATE INDEX IF NOT EXISTS refs_to ON refs (to_note, key);
"""
REF_KEYS = ("subject", "owner", "supersedes", "superseded_by")
ENTITY_KEYS = ("subject", "owner")  # these resolve only to entity notes


def _add_kb_tables(conn: sqlite3.Connection) -> None:
    """Add the entity and reference tables. In an older index the rules hash is removed, so the next update
    writes their rows from the frontmatter, with no new embedding."""
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if {"entities", "entity_aliases", "refs"} <= have:
        return
    conn.executescript(KB_TABLES)
    conn.execute("DELETE FROM meta WHERE key = 'rules_hash'")
    conn.commit()


def _refs(m) -> list[tuple[str, str]]:
    """(key, target) of the reference keys of a note, in order."""
    successor = [m.superseded_by] if m.superseded_by else []
    keys = zip(REF_KEYS, (m.subject, m.owner, m.supersedes, successor), strict=True)
    return [(k, t) for k, targets in keys for t in targets]


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
    conn.execute("DELETE FROM refs WHERE note_id = ?", (note_id,))
    conn.execute("UPDATE refs SET to_note = NULL WHERE to_note = ?", (note_id,))
    conn.execute("DELETE FROM entities WHERE note_id = ?", (note_id,))
    conn.execute("DELETE FROM entity_aliases WHERE note_id = ?", (note_id,))
    conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))


@dataclass
class UpdateResult:
    notes: int = 0  # notes on disk
    added: int = 0
    changed: int = 0
    moved: int = 0  # same content at a new path
    removed: int = 0
    touched: int = 0  # time or size changed, content did not
    rederived: int = 0  # same content, new metadata or chunks after a change of the rules
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
    rederived: bool = False


def _same(cfg: Config, conn: sqlite3.Connection, note_id: int, m, chunks) -> bool:
    """True when the index already holds this metadata and these chunks for the note."""
    row = conn.execute(
        "SELECT title, project, area, type, status, sensitivity, description, note_class, validity, as_of,"
        " superseded_by, subject FROM notes WHERE id = ?",
        (note_id,),
    ).fetchone()
    if row != (
        m.title, m.project, m.area, m.type, m.status, m.sensitivity, m.description,
        m.note_class, m.validity, m.as_of, m.superseded_by, "\n".join(m.subject) or None,
    ):  # fmt: skip
        return False
    jira = [k for (k,) in conn.execute("SELECT key FROM jira WHERE note_id = ? ORDER BY key", (note_id,))]
    refs = conn.execute("SELECT key, target FROM refs WHERE note_id = ? ORDER BY rowid", (note_id,)).fetchall()
    entity = conn.execute("SELECT kind FROM entities WHERE note_id = ?", (note_id,)).fetchall()
    aliases = [
        a for (a,) in conn.execute("SELECT alias FROM entity_aliases WHERE note_id = ? ORDER BY rowid", (note_id,))
    ]
    if (refs, entity, aliases) != (_refs(m), [(m.kind,)] if m.entity else [], m.aliases):
        return False
    stored = [h for (h,) in conn.execute("SELECT text_hash FROM chunks WHERE note_id = ? ORDER BY ord", (note_id,))]
    return jira == m.jira and stored == [input_hash(cfg, c.heading, c.text) for c in chunks]


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

    When the metadata rules of the configuration changed, derive the metadata of all notes again, with no limit,
    and rewrite the notes whose metadata or chunks changed. The embedding cache keeps the unchanged chunks.
    When the scan adds, removes or changes a project marker file, do the same for the notes under the folder of
    that file.
    """
    start = time.monotonic()
    result = UpdateResult()
    rules = rules_hash(cfg)
    rederive = conn.execute("SELECT value FROM meta WHERE key = 'rules_hash'").fetchone() != (rules,)
    if rederive:
        limit = None
    known = {
        path: (nid, mtime, size, h)
        for nid, path, mtime, size, h in conn.execute("SELECT id, path, mtime_ns, size, hash FROM notes")
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
    for i, rel in enumerate(todo):
        mtime_ns, size = on_disk[rel]
        data = cfg.note_path(rel).read_bytes()
        h = _hash_file(data)
        old = known.get(rel)
        if old and old[3] == h:
            if old[1:3] != (mtime_ns, size):
                conn.execute("UPDATE notes SET mtime_ns = ?, size = ? WHERE id = ?", (mtime_ns, size, old[0]))
                result.touched += 1
            if not (rederive or rel.startswith(marker_dirs)):
                continue
            fm, body = read_note(cfg.note_path(rel))
            meta = resolve(cfg, rel, fm, body)
            chunks, _ = chunk_note(cfg, meta, body)
            if _same(cfg, conn, old[0], meta, chunks):
                continue
            links = sorted({m.group(1).strip() for m in WIKI_LINK.finditer(body)})
            pending.append(_Pending(rel, old[0], mtime_ns, size, h, meta, chunks, links, rederived=True))
        else:
            # a re-derived note does not count toward the limit, also while it waits in pending
            written = result.added + result.changed + result.moved + sum(not p.rederived for p in pending)
            if limit is not None and written >= limit:
                result.deferred += 1
                continue
            note_id = old[0] if old else None
            moved_from = None if old else gone_by_hash.pop(h, None)
            if moved_from:
                # a move: rewrite the old note at the new path, the metadata can change with the path
                note_id = gone.pop(moved_from)[0]
            fm, body = read_note(cfg.note_path(rel))
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
    if result.added or result.changed or result.moved or result.removed or result.rederived:
        resolve_links(conn)
        resolve_refs(conn)
    if rederive:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('rules_hash', ?)", (rules,))
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
        other = conn.execute("SELECT id FROM notes WHERE path = ?", (p.rel,)).fetchone()
        if other:
            _delete_note(conn, other[0])
        cur = conn.execute(
            "INSERT INTO notes (vault, path, title, project, area, type, status, sensitivity, description,"
            " mtime_ns, size, hash, note_class, validity, as_of, superseded_by, subject)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                m.note_class,
                m.validity,
                m.as_of,
                m.superseded_by,
                "\n".join(m.subject) or None,
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
        conn.executemany(
            "INSERT INTO refs (note_id, key, target) VALUES (?, ?, ?)", [(note_id, k, t) for k, t in _refs(m)]
        )
        if m.entity:
            conn.execute("INSERT INTO entities (note_id, kind) VALUES (?, ?)", (note_id, m.kind))
            conn.executemany(
                "INSERT INTO entity_aliases (note_id, alias, norm) VALUES (?, ?, ?)",
                [(note_id, a, normalize_alias(a)) for a in m.aliases],
            )


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


class _Paths:
    """Resolve a link target to a note by its path: the last segment, then the folders that the target gives.

    Rank the candidates by the same vault as the note with the link, then the most folders in common with that
    note, then the shortest path.
    """

    def __init__(self, conn: sqlite3.Connection):
        notes = [
            (nid, vault, _link_key(path)) for nid, vault, path in conn.execute("SELECT id, vault, path FROM notes")
        ]
        self.by_name: dict[str, list[tuple[int, str, str]]] = {}
        for n in notes:
            self.by_name.setdefault(n[2].rsplit("/", 1)[-1], []).append(n)
        self.vault_of = {nid: vault for nid, vault, _ in notes}
        self.path_of = {nid: path for nid, _, path in notes}

    def find(
        self, target: str, from_note: int | None, among: set[int] | None = None, from_path: str | None = None
    ) -> int | None:
        """The note of a link target. from_path is the path of a note with the link that is not in the index."""
        key = _link_key(target).strip("/")
        if from_note is None and from_path:
            home, folders = from_path.split("/")[0], _link_key(from_path).split("/")[:-1]
        else:
            home = self.vault_of.get(from_note)
            folders = self.path_of.get(from_note, "").split("/")[:-1]
        if key.startswith(("./", "../")):
            if home is None:
                return None
            # a relative link: the path from the folder of the note with the link, inside its vault
            joined = posixpath.normpath(posixpath.join("/".join(folders[1:]), key))
            return next(
                (
                    n[0]
                    for n in self.by_name.get(joined.rsplit("/", 1)[-1], [])
                    if n[1] == home and n[2] == f"{_link_key(home)}/{joined}" and (among is None or n[0] in among)
                ),
                None,
            )
        candidates = [
            n
            for n in self.by_name.get(key.rsplit("/", 1)[-1], [])
            if n[2].endswith(key) and (among is None or n[0] in among)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: (n[1] != home, -_common_folders(folders, n[2]), len(n[2])))[0]


def resolve_links(conn: sqlite3.Connection) -> None:
    """Resolve each wiki-link target to a note by its path."""
    paths = _Paths(conn)
    links = conn.execute("SELECT rowid, from_note, target FROM links").fetchall()
    updates = [(paths.find(target, from_note), rowid) for rowid, from_note, target in links]
    conn.executemany("UPDATE links SET to_note = ? WHERE rowid = ?", updates)


class _Resolver:
    """Resolve the target of a reference key: by path, then by a unique alias of an entity note (its title counts
    as an alias). An alias of two entity notes resolves to nothing."""

    def __init__(self, conn: sqlite3.Connection):
        self.paths = _Paths(conn)
        self.entities = {nid for (nid,) in conn.execute("SELECT note_id FROM entities")}
        names = conn.execute(
            "SELECT note_id, norm FROM entity_aliases"
            " UNION SELECT n.id, n.title FROM notes n JOIN entities e ON e.note_id = n.id"
        ).fetchall()
        self.by_alias: dict[str, set[int]] = {}
        for nid, name in names:
            self.by_alias.setdefault(normalize_alias(name), set()).add(nid)

    def _alias(self, text: str) -> int | None:
        found = self.by_alias.get(normalize_alias(text), set())
        return next(iter(found)) if len(found) == 1 else None

    def entity(self, text: str, from_note: int | None, from_path: str | None = None) -> int | None:
        """A `subject` or an `owner`: an entity note only."""
        found = self.paths.find(text, from_note, self.entities, from_path)
        return found if found is not None else self._alias(text)

    def note(self, text: str, from_note: int | None, from_path: str | None = None) -> int | None:
        """A `supersedes` or a `superseded_by`: any note, or an entity note by its alias."""
        found = self.paths.find(text, from_note, None, from_path)
        return found if found is not None else self._alias(text)


def resolve_refs(conn: sqlite3.Connection) -> None:
    """Resolve the targets of the reference keys to notes. A target that resolves to nothing keeps a null note."""
    r = _Resolver(conn)
    rows = conn.execute("SELECT rowid, note_id, key, target FROM refs").fetchall()
    updates = [
        ((r.entity if key in ENTITY_KEYS else r.note)(target, note_id), rowid) for rowid, note_id, key, target in rows
    ]
    conn.executemany("UPDATE refs SET to_note = ? WHERE rowid = ?", updates)


def find_entity(conn: sqlite3.Connection, value: str) -> int | None:
    """The entity note of a name, an alias or a path (a wiki-link is fine), as a `subject` target resolves."""
    targets = link_targets(value)
    return _Resolver(conn).entity(targets[0], None) if targets else None


def status(cfg: Config, conn: sqlite3.Connection) -> dict:
    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    return {
        "index": str(cfg.index_path),
        "size_mb": round(cfg.index_path.stat().st_size / 1e6, 1) if cfg.index_path.exists() else 0,
        "model": one("SELECT value FROM meta WHERE key = 'model'"),
        "dimension": int(one("SELECT value FROM meta WHERE key = 'dimension'")),
        "rules_hash": (conn.execute("SELECT value FROM meta WHERE key = 'rules_hash'").fetchone() or [None])[0],
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
