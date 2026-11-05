"""The entity rows of core.db: the vocabularies, the entity, alias, fact, relation and canonical rows of a document,
and the reference pass that resolves the target texts to nodes."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date

from .config import Config
from .metadata import _identity_key, _iso_date, _targets

DOC_SUFFIX = "~doc"  # the doc node of an entity document is `<entity id>~doc` (decision 6)

KB_SCHEMA = """
CREATE TABLE IF NOT EXISTS main.kind (
    name TEXT PRIMARY KEY,
    standard TEXT NOT NULL,
    description TEXT
);
CREATE TABLE IF NOT EXISTS main.doc_type (
    name TEXT PRIMARY KEY,
    class TEXT NOT NULL,
    one_per TEXT NOT NULL,
    required_sections TEXT NOT NULL,
    review_months INTEGER,
    standard TEXT NOT NULL,
    UNIQUE (name, class)
);
CREATE TABLE IF NOT EXISTS main.relation_type (
    name TEXT PRIMARY KEY,
    src_kinds TEXT NOT NULL,
    dst_kinds TEXT NOT NULL,
    inverse TEXT NOT NULL,
    standard TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS main.attribute_def (
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    value_type TEXT NOT NULL,
    live INTEGER NOT NULL,
    system TEXT NOT NULL,
    skill TEXT NOT NULL,
    scoped INTEGER NOT NULL,
    PRIMARY KEY (kind, name)
);
CREATE TABLE IF NOT EXISTS main.entity (
    id TEXT PRIMARY KEY REFERENCES node (id) ON DELETE CASCADE ON UPDATE CASCADE,
    kind TEXT,
    name TEXT NOT NULL,
    owner TEXT,
    owner_text TEXT,
    status TEXT NOT NULL,
    merged_into TEXT,
    as_of TEXT,
    review_by TEXT,
    doc_id TEXT NOT NULL UNIQUE REFERENCES doc (id) ON DELETE CASCADE ON UPDATE CASCADE
);
CREATE TABLE IF NOT EXISTS main.entity_alias (
    entity_id TEXT NOT NULL REFERENCES entity (id) ON DELETE CASCADE ON UPDATE CASCADE,
    alias_norm TEXT NOT NULL,
    alias TEXT NOT NULL,
    PRIMARY KEY (entity_id, alias_norm)
);
CREATE INDEX IF NOT EXISTS main.entity_alias_norm ON entity_alias (alias_norm);
CREATE TABLE IF NOT EXISTS main.fact (
    id INTEGER PRIMARY KEY,
    entity_id TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT,
    scope TEXT,
    valid_from TEXT,
    valid_until TEXT,
    as_of TEXT,
    source_doc TEXT,
    source_ref TEXT,
    origin TEXT NOT NULL,
    confidence REAL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS main.fact_entity ON fact (entity_id);
CREATE INDEX IF NOT EXISTS main.fact_source ON fact (source_doc);
CREATE TABLE IF NOT EXISTS main.relation (
    id INTEGER PRIMARY KEY,
    src TEXT NOT NULL,
    rel TEXT NOT NULL,
    dst TEXT,
    dst_text TEXT,
    origin TEXT NOT NULL,
    confidence REAL,
    valid_from TEXT,
    valid_until TEXT,
    source_doc TEXT
);
CREATE INDEX IF NOT EXISTS main.relation_src ON relation (src);
CREATE INDEX IF NOT EXISTS main.relation_dst ON relation (dst);
CREATE INDEX IF NOT EXISTS main.relation_source ON relation (source_doc);
CREATE TABLE IF NOT EXISTS main.entity_canonical (
    entity_id TEXT,
    aspect TEXT,
    doc_id TEXT,
    doc_text TEXT,
    source_doc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS main.entity_canonical_source ON entity_canonical (source_doc);
"""


# letters that NFKD does not decompose and that have a clear Latin base
_FOLD = str.maketrans({"ł": "l", "ø": "o", "đ": "d", "ħ": "h", "ß": "ss", "æ": "ae", "œ": "oe"})


def normalize_alias(text: str) -> str:
    """Lower case, no diacritics, single spaces. A letter with no decomposition (such as ł) folds by a map."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.lower().translate(_FOLD).split())


def entity_id_of(doc_id: str) -> str:
    return doc_id.removesuffix(DOC_SUFFIX)


# --- the vocabularies -------------------------------------------------------------------------------------------


def _vocabulary_rows(cfg: Config) -> dict[str, list[tuple]]:
    kb = cfg.kb
    return {
        "kind": [(name, standard, None) for name, standard in kb.kinds.items()],
        "doc_type": [
            (name, t.class_, t.one_per, json.dumps(t.required_sections), t.review_months, t.standard)
            for name, t in kb.doc_types.items()
        ],
        "relation_type": [
            (name, json.dumps(t.src_kinds), json.dumps(t.dst_kinds), t.inverse, t.standard)
            for name, t in kb.relation_types.items()
        ],
        "attribute_def": [
            (kind, name, a.value_type, int(a.live), a.system, a.skill, int(a.scoped))
            for kind, names in kb.attributes.items()
            for name, a in names.items()
        ],
    }


def write_vocabulary(conn: sqlite3.Connection, cfg: Config) -> None:
    """Write the vocabulary tables from the configuration. A table whose rows agree is not written."""
    for table, rows in _vocabulary_rows(cfg).items():
        stored = conn.execute(f"SELECT * FROM main.{table}").fetchall()
        key = lambda r: tuple("" if v is None else str(v) for v in r)  # noqa: E731
        if sorted(stored, key=key) == sorted(rows, key=key):
            continue
        conn.execute(f"DELETE FROM main.{table}")
        if rows:
            conn.executemany(f"INSERT INTO main.{table} VALUES ({', '.join('?' * len(rows[0]))})", rows)


# --- the rows of one document -----------------------------------------------------------------------------------


@dataclass
class KbDoc:
    """The entity rows that one document declares, before the references resolve."""

    entity: bool = False
    kind: str | None = None
    name: str = ""
    status: str = "active"
    aliases: list[tuple[str, str]] = field(default_factory=list)  # (alias_norm, alias); the name comes first
    # (attribute, value, scope, valid_from, valid_until, as_of, source_ref)
    facts: list[tuple] = field(default_factory=list)
    relations: list[tuple[str, str]] = field(default_factory=list)  # (rel, dst_text)
    canonical: list[tuple[str, str]] = field(default_factory=list)  # the canonical list: (aspect, doc_text)
    canonical_task: bool = False  # `canonical: true`: canonical for the primary subject and the task
    task: str | None = None
    owner_text: str | None = None
    as_of: str | None = None
    review_by: str | None = None


def _text(value) -> str | None:
    """A frontmatter value as text: a string as it is, a date in ISO form, a map or a list as JSON."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def _date(value) -> str | None:
    return None if value is None else _iso_date(value)


def parse(cfg: Config, meta, fm: dict) -> KbDoc:
    """The entity rows of the note. `meta` is the resolved metadata, `fm` the frontmatter."""
    doc = KbDoc(
        canonical_task=meta.canonical,
        task=meta.task,
        owner_text=meta.owner_text,
        as_of=meta.as_of,
        review_by=meta.review_by,
    )
    for rel in cfg.kb.relation_types:
        value = fm.get(rel)
        if isinstance(value, str | list):
            for target in _targets(value):
                if (rel, target) not in doc.relations:
                    doc.relations.append((rel, target))
    if meta.type != "entity":
        return doc
    doc.entity = True
    kind = fm.get("kind")
    doc.kind = str(kind).strip().lower() or None if kind is not None else None
    doc.name = meta.title
    doc.status = "archived" if meta.status == "archived" else "active"
    aliases = fm.get("aliases")
    if isinstance(aliases, str):
        aliases = [aliases]
    seen = set()
    for alias in [doc.name, *(aliases if isinstance(aliases, list) else [])]:
        text = str(alias).strip()
        norm = normalize_alias(text)
        if norm and norm not in seen:
            seen.add(norm)
            doc.aliases.append((norm, text))
    for item in fm.get("facts") or []:
        if not isinstance(item, dict) or not str(item.get("attribute") or "").strip():
            continue
        attribute = str(item["attribute"]).strip()
        doc.facts.append(
            (
                attribute,
                _text(item.get("value")),
                _text(item.get("scope")),
                _date(item.get("valid_from")),
                _date(item.get("valid_until")),
                _date(item.get("as_of")),
                _text(item.get("source")),
            )
        )
    canonical = fm.get("canonical")
    if isinstance(canonical, dict):
        for aspect, link in canonical.items():
            for target in _targets(link)[:1] if isinstance(link, str | list) else []:
                doc.canonical.append((str(aspect), target))
    return doc


def _owned(conn: sqlite3.Connection, doc_id: str) -> dict:
    """The stored rows that the document declares, in the form of `_expected()`."""
    one = lambda sql: conn.execute(sql, (doc_id,)).fetchall()  # noqa: E731
    entity = one("SELECT kind, name, status, owner_text, as_of, review_by FROM entity WHERE doc_id = ?")
    return {
        "entity": entity[0] if entity else None,
        "aliases": one(
            "SELECT alias_norm, alias FROM entity_alias WHERE entity_id IN (SELECT id FROM entity WHERE doc_id = ?)"
            " ORDER BY rowid"
        ),
        "facts": one(
            "SELECT attribute, value, scope, valid_from, valid_until, as_of, source_ref FROM fact"
            " WHERE source_doc = ? AND origin = 'declared' ORDER BY id"
        ),
        "relations": one(
            "SELECT src, rel, dst_text FROM relation WHERE source_doc = ? AND origin = 'declared' ORDER BY id"
        ),
        "canonical": one("SELECT aspect, doc_text FROM entity_canonical WHERE source_doc = ? ORDER BY rowid"),
    }


def _expected(doc_id: str, kb: KbDoc) -> dict:
    entity_id = entity_id_of(doc_id)
    src = entity_id if kb.entity else doc_id
    canonical = [(aspect, text) for aspect, text in kb.canonical]
    if kb.canonical_task:
        canonical.append((kb.task, None))
    return {
        "entity": (kb.kind, kb.name, kb.status, kb.owner_text, kb.as_of, kb.review_by) if kb.entity else None,
        "aliases": list(kb.aliases),
        "facts": list(kb.facts),
        "relations": [(src, rel, text) for rel, text in kb.relations],
        "canonical": canonical,
    }


def same(conn: sqlite3.Connection, doc_id: str, kb: KbDoc) -> bool:
    """True when the index already holds the entity rows that the document declares."""
    owned = _owned(conn, doc_id)
    expected = _expected(doc_id, kb)
    return {k: [tuple(r) for r in v] if isinstance(v, list) else v for k, v in owned.items()} == expected


def write(conn: sqlite3.Connection, doc_id: str, kb: KbDoc) -> None:
    """Write the entity rows of a doc after its doc row. The references stay null until `resolve_refs()`."""
    rows = _expected(doc_id, kb)
    entity_id = entity_id_of(doc_id)
    if kb.entity:
        conn.execute(
            "INSERT INTO node (id, node_type, collection, mode, provisional)"
            " SELECT ?, 'entity', collection, mode, provisional FROM node WHERE id = ?",
            (entity_id, doc_id),
        )
        conn.execute(
            "INSERT INTO entity (id, kind, name, owner_text, status, as_of, review_by, doc_id)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (entity_id, kb.kind, kb.name, kb.owner_text, kb.status, kb.as_of, kb.review_by, doc_id),
        )
        conn.executemany(
            "INSERT INTO entity_alias (entity_id, alias_norm, alias) VALUES (?, ?, ?)",
            [(entity_id, norm, alias) for norm, alias in rows["aliases"]],
        )
        conn.executemany(
            "INSERT INTO fact (entity_id, attribute, value, scope, valid_from, valid_until, as_of, source_ref,"
            " source_doc, origin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'declared')",
            [(entity_id, *f, doc_id) for f in rows["facts"]],
        )
    conn.executemany(
        "INSERT INTO relation (src, rel, dst_text, source_doc, origin) VALUES (?, ?, ?, ?, 'declared')",
        [(*r, doc_id) for r in rows["relations"]],
    )
    conn.executemany(
        "INSERT INTO entity_canonical (entity_id, aspect, doc_id, doc_text, source_doc) VALUES (?, ?, ?, ?, ?)",
        [
            (entity_id if kb.entity and text is not None else None, aspect, None if text else doc_id, text, doc_id)
            for aspect, text in rows["canonical"]
        ],
    )


def delete(conn: sqlite3.Connection, doc_id: str) -> None:
    """Delete the entity node of the doc and the facts, relations and canonical rows that it declares."""
    conn.execute("DELETE FROM node WHERE id IN (SELECT id FROM entity WHERE doc_id = ?)", (doc_id,))
    conn.execute("DELETE FROM fact WHERE source_doc = ? AND origin = 'declared'", (doc_id,))
    conn.execute("DELETE FROM relation WHERE source_doc = ? AND origin = 'declared'", (doc_id,))
    conn.execute("DELETE FROM entity_canonical WHERE source_doc = ?", (doc_id,))


def rename(conn: sqlite3.Connection, old: str, new: str) -> None:
    """Give a node a new ID in the rows that refer to it by value (no foreign key reaches them)."""
    for table, columns in (
        ("fact", ("entity_id", "source_doc")),
        ("relation", ("src", "dst", "source_doc")),
        ("entity_canonical", ("entity_id", "doc_id", "source_doc")),
        ("entity", ("owner",)),
        ("doc", ("owner", "primary_subject", "superseded_by")),
        ("doc_subject", ("entity_id",)),
    ):
        for column in columns:
            conn.execute(f"UPDATE {table} SET {column} = ? WHERE {column} = ?", (new, old))


# --- the reference pass -----------------------------------------------------------------------------------------


def link_key(path: str) -> str:
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


class PathIndex:
    """Resolve a link target to a doc by its path.

    Rank the candidates by the same vault as the note with the link, then the most folders in common with that
    note, then the shortest path.
    """

    def __init__(self, conn: sqlite3.Connection):
        docs = [
            (doc_id, vault, link_key(path)) for doc_id, vault, path in conn.execute("SELECT id, vault, path FROM doc")
        ]
        self.by_name: dict[str, list[tuple[str, str, str]]] = {}
        for n in docs:
            self.by_name.setdefault(n[2].rsplit("/", 1)[-1], []).append(n)
        self.vault_of = {doc_id: vault for doc_id, vault, _ in docs}
        self.path_of = {doc_id: path for doc_id, _, path in docs}

    def find(self, target: str, from_doc: str | None, among: set[str] | None = None) -> str | None:
        key = link_key(target).strip("/")
        candidates = [
            n
            for n in self.by_name.get(key.rsplit("/", 1)[-1], [])
            if n[2].endswith(key) and (among is None or n[0] in among)
        ]
        if not candidates:
            return None
        home = self.vault_of.get(from_doc)
        folders = self.path_of.get(from_doc, "").split("/")[:-1]
        return min(candidates, key=lambda n: (n[1] != home, -_common_folders(folders, n[2]), len(n[2])))[0]


class _Resolver:
    def __init__(self, conn: sqlite3.Connection):
        self.paths = PathIndex(conn)
        self.doc_ids = set(self.paths.vault_of)
        self.doc_of_entity = dict(conn.execute("SELECT id, doc_id FROM entity").fetchall())
        self.entity_of_doc = {d: e for e, d in self.doc_of_entity.items()}
        self.entity_docs = set(self.entity_of_doc)
        self.by_alias: dict[str, set[str]] = {}
        for entity_id, norm in conn.execute("SELECT entity_id, alias_norm FROM entity_alias"):
            self.by_alias.setdefault(norm, set()).add(entity_id)

    def _alias(self, text: str) -> str | None:
        found = self.by_alias.get(normalize_alias(text), set())
        return next(iter(found)) if len(found) == 1 else None

    def entity(self, text: str | None, from_doc: str | None) -> str | None:
        """A subject, an owner or a relation end: entity ID, then path among the entity docs, then one alias."""
        if not text:
            return None
        if text in self.doc_of_entity:
            return text
        doc = self.paths.find(text, from_doc, self.entity_docs)
        if doc is not None:
            return self.entity_of_doc[doc]
        return self._alias(text)

    def doc(self, text: str | None, from_doc: str | None) -> str | None:
        """A doc target: doc ID, entity ID to its doc, path, then one alias to its entity doc."""
        if not text:
            return None
        if text in self.doc_ids:
            return text
        if text in self.doc_of_entity:
            return self.doc_of_entity[text]
        doc = self.paths.find(text, from_doc)
        if doc is not None:
            return doc
        entity = self._alias(text)
        return self.doc_of_entity[entity] if entity else None


def _update(conn: sqlite3.Connection, sql: str, rows: list[tuple]) -> None:
    if rows:
        conn.executemany(sql, rows)


def resolve_refs(conn: sqlite3.Connection) -> None:
    """Resolve the subjects, owners, relation ends, canonical rows, successors and the identity keys.

    Run after `resolve_links()`. A body link gets the doc of an `id` match over the path match, and the doc of a
    unique alias when no path matches. An absent or ambiguous target stays null. Only the changed values are written.
    """
    r = _Resolver(conn)

    links = conn.execute("SELECT from_doc, ord, target_text, to_node FROM link").fetchall()
    rows = []
    for from_doc, ord_, text, to_node in links:
        new = to_node
        if text in r.doc_ids:
            new = text
        elif text in r.doc_of_entity:
            new = r.doc_of_entity[text]
        elif to_node is None and (entity := r._alias(text)):
            new = r.doc_of_entity[entity]
        if new != to_node:
            rows.append((new, from_doc, ord_))
    _update(conn, "UPDATE link SET to_node = ? WHERE from_doc = ? AND ord = ?", rows)

    subjects = conn.execute("SELECT doc_id, ord, entity_id, target_text FROM doc_subject").fetchall()
    primary: dict[str, str | None] = {}
    rows = []
    for doc_id, ord_, old, text in subjects:
        new = r.entity(text, doc_id)
        if ord_ == 0:
            primary[doc_id] = new
        if new != old:
            rows.append((new, doc_id, ord_))
    _update(conn, "UPDATE doc_subject SET entity_id = ? WHERE doc_id = ? AND ord = ?", rows)

    rows = []
    for entity_id, doc_id, old, text in conn.execute("SELECT id, doc_id, owner, owner_text FROM entity").fetchall():
        new = r.entity(text, doc_id)
        if new != old:
            rows.append((new, entity_id))
    _update(conn, "UPDATE entity SET owner = ? WHERE id = ?", rows)

    rows = []
    for id_, source_doc, old, text in conn.execute("SELECT id, source_doc, dst, dst_text FROM relation").fetchall():
        new = r.entity(text, source_doc)
        if new != old:
            rows.append((new, id_))
    _update(conn, "UPDATE relation SET dst = ? WHERE id = ?", rows)

    # the derived successor: a doc that lists the target in `supersedes`; the first path wins
    texts = {d: t for d, t in conn.execute("SELECT id, superseded_by_text FROM doc")}
    derived: dict[str, str] = {}
    for doc_id, text in conn.execute(
        "SELECT s.doc_id, s.target_text FROM doc_supersedes s JOIN doc d ON d.id = s.doc_id ORDER BY d.path, s.ord"
    ).fetchall():
        target = r.doc(text, doc_id)
        if target is not None and target != doc_id and texts.get(target) is None:
            derived.setdefault(target, doc_id)

    types = dict(conn.execute("SELECT name, one_per FROM doc_type").fetchall())
    rows = []
    docs = conn.execute(
        "SELECT id, vault, type, project, task, owner, owner_text, primary_subject, superseded_by, identity_key"
        " FROM doc"
    ).fetchall()
    for doc_id, vault, type_, project, task, owner, owner_text, subject, successor, key in docs:
        new_owner = r.entity(owner_text, doc_id)
        new_subject = primary.get(doc_id)
        text = texts.get(doc_id)
        new_successor = r.doc(text, doc_id) if text is not None else derived.get(doc_id)
        parts = {
            "subject": new_subject,
            "task": " ".join(task.lower().split()) if task else None,
            "project": project if project != "none" else None,
        }
        new_key = _identity_key(types.get(type_, "none"), parts, vault, type_)
        if (new_owner, new_subject, new_successor, new_key) != (owner, subject, successor, key):
            rows.append((new_owner, new_subject, new_successor, new_key, doc_id))
    _update(
        conn,
        "UPDATE doc SET owner = ?, primary_subject = ?, superseded_by = ?, identity_key = ? WHERE id = ?",
        rows,
    )

    # `canonical: true` names the primary subject of its doc; a canonical list names a doc by its text
    subject_of = dict(conn.execute("SELECT id, primary_subject FROM doc").fetchall())
    rows = []
    canonical = conn.execute("SELECT rowid, entity_id, doc_id, doc_text, source_doc FROM entity_canonical").fetchall()
    for rowid, entity_id, doc_id, text, source_doc in canonical:
        if text is None:
            new = (subject_of.get(source_doc), source_doc)
        else:
            new = (entity_id, r.doc(text, source_doc))
        if new != (entity_id, doc_id):
            rows.append((*new, rowid))
    _update(conn, "UPDATE entity_canonical SET entity_id = ?, doc_id = ? WHERE rowid = ?", rows)
