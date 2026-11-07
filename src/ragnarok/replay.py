"""The replay set of the write path and the measure R1.

`build` makes the cases from the index and its mirror (the configured vaults):
- a positive case is one newer note with the old notes that it replaces, from the resolved `superseded_by` refs
  and the resolved wiki-links of `supersedes` (a free-text entry adds no pair);
- each group of notes with an equal normalized body of at least `min_words` words (layer D0, the whole mirror)
  is a duplicate case: the draft is the note with the last path, the expected action is `same` for each other
  note. A supersession pair in such a group is a duplicate case only. The normalization (`NORMALIZATION`, recorded
  in the manifest) replaces each wiki-link by its display text before it collapses whitespace, so a moved note
  whose only change is rewritten link targets with the same display text is a duplicate. The similarity of the
  moved-copy split uses the same normalization;
- a negative case (expected action `new`) is a sampled note in no case, with no shared `subject`, no resolved
  wiki-link in or out, and a best chunk distance to the chunks of the other notes above a frozen limit.
Each case has two draft forms (`bare`: the body only; `typed`: the body with the newer note's own `type`,
`subject` and `task`), the leak record, the expected action of each old note and the start state (recorded only).

`score` runs `find_collisions` on each positive case and draft form, with the newer note excluded from the
candidates. The full index stands in for a per-case index of the start state: `find_collisions` selects its
candidates from the identity key, the search with superseded notes included, and the links and `supersedes` of the
draft. None of them reads `supersedes`, `superseded_by` or `validity` of an old note, which are the only keys that
the start state removes. The newer note leaves the pool before the limit. It still counts in the full-text
statistics and in the ranks before the fusion; that difference is small and not corrected.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sqlite3
import subprocess
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import frontmatter
import numpy as np

from .collisions import SOURCES, find_collisions
from .config import Config
from .metadata import iter_notes, link_targets, normalize_alias, parse_note
from .validate import _Context, _link, body_links

MARKER = ".ragnarok-bench"
FORMS = ("bare", "typed")
TYPED_KEYS = ("type", "subject", "task")
REMOVED_KEYS = ("supersedes", "superseded_by", "validity")
DISTANCES = (0.15, 0.20, 0.25, 0.30)
SUBSETS = ("all", "no_leak", "moved_copy", "content_change")
Z = 1.96  # the 95% interval
WIKI_LINK = re.compile(r"!?\[\[([^\[\]\n]*)\]\]")
NORMALIZATION = (
    "each wiki-link ([[t]], [[t|alias]], [[t#heading]], ![[t]]) replaced by its display text (the alias, else the"
    " last path segment of the target without .md and #heading), then whitespace collapsed to single spaces"
)
SIMILARITY = (
    "difflib.SequenceMatcher ratio of the word lists of the normalized bodies (body_normalization), autojunk off"
)


@dataclass
class ReplaySet:
    cases: list[dict]
    counts: dict
    qualifying: dict[str, int] = field(default_factory=dict)  # the negative pool size for each distance limit


# --- bodies and drafts -------------------------------------------------------------------------------------------


def _display(match: re.Match) -> str:
    """The display text of a wiki-link: the alias, else the last path segment of the target, without .md, the
    heading and the block id (a link to a heading of the same note shows the heading)."""
    inner = match.group(1)
    if "|" in inner:
        return inner.split("|", 1)[1].strip()
    target = re.split(r"[#^]", inner, maxsplit=1)[0].rstrip("\\").strip()
    if not target:
        return inner.lstrip("#^").strip()
    return target.rsplit("/", 1)[-1].removesuffix(".md")


def normalize(body: str) -> str:
    """The body with each wiki-link replaced by its display text and whitespace collapsed to single spaces (see
    NORMALIZATION). Code is not treated apart: both bodies of a comparison get the same change."""
    return " ".join(WIKI_LINK.sub(_display, body).split())


def body_hash(body: str) -> str:
    return hashlib.sha256(normalize(body).encode()).hexdigest()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a).split(), normalize(b).split(), autojunk=False).ratio()


def drafts(fm: dict, body: str) -> dict[str, str]:
    """The two draft forms of a note: the body, and the body with only its own type, subject and task."""
    keys = {k: fm[k] for k in TYPED_KEYS if k in fm}
    if not keys:
        return {"bare": body, "typed": body}
    post = frontmatter.Post(body)
    post.metadata.update(keys)
    return {"bare": body, "typed": frontmatter.dumps(post) + "\n"}


# --- build -------------------------------------------------------------------------------------------------------


class _Notes:
    """The text, frontmatter, body and validate view of the notes that the build reads, each read once."""

    def __init__(self, cfg: Config, conn: sqlite3.Connection):
        self.cfg, self.ctx = cfg, _Context(cfg, conn)
        self.path_of = {nid: path for path, nid in self.ctx.ids.items()}
        self._cache: dict[str, tuple] = {}

    def get(self, path: str) -> tuple[dict, str, object]:
        if path not in self._cache:
            text = self.cfg.note_path(path).read_text(errors="replace")
            fm, body = parse_note(text)
            self._cache[path] = (fm, body, self.ctx.note(text, path, strict=False))
        return self._cache[path]

    def links(self, path: str) -> set[int]:
        """The notes that the body of the note links to (outside code)."""
        nid = self.ctx.ids[path]
        found = {self.ctx.resolver.paths.find(t, nid, None, path) for t in body_links(self.get(path)[1]) if t}
        return found - {None}


def _case(notes: _Notes, kind: str, newer: str, old: list[dict]) -> dict:
    fm, body, note = notes.get(newer)
    linked = {notes.path_of[n] for n in notes.links(newer)}
    remove = {}
    for o in old:
        ofm = notes.get(o["path"])[0]
        remove[o["path"]] = [k for k in REMOVED_KEYS if k in ofm]
    return {
        "id": "",
        "kind": kind,
        "newer": newer,
        "old": old,
        "expected": "new" if kind == "negative" else None,
        "leak": any(o["path"] in linked for o in old),
        "positive_kind": None,
        "identity_key": note.identity is not None,
        "drafts": drafts(fm, body),
        "start_state": {"exclude": newer, "remove_keys": remove},
    }


def _expected(notes: _Notes, newer: str, old: str) -> str:
    """merge when the old note is a state note with the identity key of the newer note, else supersede."""
    new_note, old_note = notes.get(newer)[2], notes.get(old)[2]
    if new_note.identity is not None and old_note.meta.note_class == "state" and old_note.identity == new_note.identity:
        return "merge"
    return "supersede"


def _wiki_targets(value) -> set[str]:
    """The targets of the wiki-link entries of a supersedes value (free text is left out)."""
    items = value if isinstance(value, list) else [value]
    return {t for v in items if _link(v) is not None for t in link_targets(v)}


def _pairs(notes: _Notes, conn: sqlite3.Connection, counts: dict) -> dict[str, list[str]]:
    """{newer path: old paths} from the resolved superseded_by refs and the resolved supersedes wiki-links."""
    rows = conn.execute(
        "SELECT n.path, r.to_note FROM refs r JOIN notes n ON n.id = r.note_id"
        " WHERE r.key = 'superseded_by' ORDER BY n.path"
    ).fetchall()
    counts.update(superseded_by_refs=len(rows), bare_resolved=0, bare_dropped=0, unresolved_links=0)
    groups: dict[str, set[str]] = {}
    for old, to in rows:
        value = notes.get(old)[0].get("superseded_by")
        bare = _link(value) is None and "[[" not in str(value)
        if to is None:
            counts["bare_dropped" if bare else "unresolved_links"] += 1
            continue
        if bare:
            counts["bare_resolved"] += 1
        newer = notes.path_of[to]
        if newer != old:
            groups.setdefault(newer, set()).add(old)
    rows = conn.execute(
        "SELECT n.path, r.target, r.to_note FROM refs r JOIN notes n ON n.id = r.note_id"
        " WHERE r.key = 'supersedes' ORDER BY n.path"
    ).fetchall()
    counts.update(supersedes_refs=len(rows), supersedes_pairs=0)
    for newer, target, to in rows:
        if to is None or target not in _wiki_targets(notes.get(newer)[0].get("supersedes")):
            continue
        old = notes.path_of[to]
        if old != newer and old not in groups.get(newer, set()):
            groups.setdefault(newer, set()).add(old)
            counts["supersedes_pairs"] += 1
    return {newer: sorted(olds) for newer, olds in groups.items()}


def _equal_bodies(notes: _Notes, min_words: int) -> list[list[str]]:
    """The groups (sorted paths) of two or more notes with an equal normalized body of at least min_words words.
    A shorter body (a stub) does not show that two notes are the same."""
    groups: dict[str, list[str]] = {}
    for path in sorted(notes.ctx.ids):
        body = notes.get(path)[1]
        if len(normalize(body).split()) >= min_words:
            groups.setdefault(body_hash(body), []).append(path)
    return [g for g in groups.values() if len(g) > 1]


def build(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    neg_count: int = 0,
    neg_distance: float | None = None,
    copy_similarity: float = 0.9,
    seed: int = 0,
) -> ReplaySet:
    """The replay set of the index. neg_count negative cases need neg_distance."""
    if neg_count and neg_distance is None:
        raise ValueError("negative cases need a distance limit (--neg-distance)")
    notes = _Notes(cfg, conn)
    counts: dict = {}
    groups = _pairs(notes, conn, counts)
    equal = _equal_bodies(notes, cfg.min_words)
    same = {path: set(g) for g in equal for path in g}
    duplicates = [
        _case(notes, "duplicate", g[-1], [{"path": old, "expected": "same", "similarity": 1.0} for old in g[:-1]])
        for g in equal
    ]
    positives = []
    for newer in sorted(groups):
        new_body = notes.get(newer)[1]
        kept = []
        for old in groups[newer]:
            if old in same.get(newer, ()):
                continue  # a duplicate case only
            ratio = round(similarity(notes.get(old)[1], new_body), 4)
            kept.append({"path": old, "expected": _expected(notes, newer, old), "similarity": ratio})
        if not kept:
            continue
        case = _case(notes, "positive", newer, kept)
        case["positive_kind"] = (
            "moved_copy" if all(o["similarity"] >= copy_similarity for o in kept) else "content_change"
        )
        positives.append(case)
    duplicates.sort(key=lambda c: c["newer"])
    for prefix, group in (("p", positives), ("d", duplicates)):
        for i, case in enumerate(group, start=1):
            case["id"] = f"{prefix}{i:03d}"

    pool = negative_pool(cfg, conn, positives + duplicates)
    qualifying = {f"{d:.2f}": sum(v > d for v in pool.values()) for d in DISTANCES}
    negatives = []
    if neg_count:
        eligible = sorted(p for p, v in pool.items() if v > neg_distance)
        if neg_count > len(eligible):
            raise ValueError(f"{neg_count} negative cases asked, {len(eligible)} notes qualify at {neg_distance}")
        for i, path in enumerate(random.Random(seed).sample(eligible, neg_count), start=1):
            case = _case(notes, "negative", path, [])
            case["id"] = f"n{i:03d}"
            negatives.append(case)

    private = {p for (p,) in conn.execute("SELECT path FROM notes WHERE sensitivity = 'private'")}
    olds = [o for c in positives for o in c["old"]]
    counts.update(
        newer_notes=len(groups),
        positive=len(positives),
        moved_copy=sum(c["positive_kind"] == "moved_copy" for c in positives),
        content_change=sum(c["positive_kind"] == "content_change" for c in positives),
        old_notes=len(olds),
        duplicate=len(duplicates),
        duplicate_targets=sum(len(c["old"]) for c in duplicates),
        negative=len(negatives),
        leak=sum(c["leak"] for c in positives),
        duplicate_leak=sum(c["leak"] for c in duplicates),
        merge=sum(o["expected"] == "merge" for o in olds),
        supersede=sum(o["expected"] == "supersede" for o in olds),
        identity_key=sum(c["identity_key"] for c in positives),
        private_old=sum(o["path"] in private for o in olds),
    )
    return ReplaySet(positives + duplicates + negatives, counts, qualifying)


def negative_pool(cfg: Config, conn: sqlite3.Connection, cases: list[dict]) -> dict[str, float]:
    """{path: best chunk distance} of the notes in no case, with no shared subject and no link in or out.
    The best distance is the least cosine distance of a chunk of the note to a chunk of another note."""
    in_case = {c["newer"] for c in cases} | {o["path"] for c in cases for o in c["old"]}
    linked = set()
    for a, b in conn.execute("SELECT from_note, to_note FROM links WHERE to_note IS NOT NULL AND to_note != from_note"):
        linked.update((a, b))
    holders: dict[object, set[int]] = {}
    for nid, target, to in conn.execute("SELECT note_id, target, to_note FROM refs WHERE key = 'subject'"):
        holders.setdefault(to if to is not None else normalize_alias(target), set()).add(nid)
    shared = {nid for ids in holders.values() if len(ids) > 1 for nid in ids}
    paths = dict(conn.execute("SELECT id, path FROM notes").fetchall())
    keep = {nid for nid, p in paths.items() if p not in in_case and nid not in linked and nid not in shared}

    rows = conn.execute("SELECT c.note_id, v.embedding FROM chunks c JOIN chunks_vec v ON v.chunk_id = c.id").fetchall()
    if not rows or not keep:
        return {}
    owners = np.array([nid for nid, _ in rows])
    vectors = np.array([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    picked = np.flatnonzero(np.isin(owners, list(keep)))
    best: dict[int, float] = {}
    for start in range(0, len(picked), 256):
        block = picked[start : start + 256]
        sims = vectors[block] @ vectors.T
        sims[owners[block][:, None] == owners[None, :]] = -np.inf  # the chunks of the same note
        for row, top in zip(block, sims.max(axis=1), strict=True):
            if np.isfinite(top):
                nid = int(owners[row])
                best[nid] = min(best.get(nid, math.inf), 1.0 - float(top))
    return {paths[nid]: d for nid, d in best.items()}


# --- the files of a set ------------------------------------------------------------------------------------------


def mirror_hash(cfg: Config) -> tuple[str, int]:
    """The sha256 of the sorted lines `<note key>\\t<sha256 of the file>` of all notes, and the note count."""
    keys = sorted(iter_notes(cfg))
    lines = "".join(f"{k}\t{hashlib.sha256(cfg.note_path(k).read_bytes()).hexdigest()}\n" for k in keys)
    return hashlib.sha256(lines.encode()).hexdigest(), len(keys)


def code_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def write_set(cfg: Config, out: Path, result: ReplaySet, parameters: dict) -> dict:
    """Write cases.jsonl and manifest.json into out; return the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in result.cases).encode()
    (out / "cases.jsonl").write_bytes(data)
    mirror, count = mirror_hash(cfg)
    manifest = {
        "parameters": {**parameters, "body_normalization": NORMALIZATION, "similarity_measure": SIMILARITY},
        "commit": code_commit(),
        "counts": result.counts,
        "negative_qualifying": result.qualifying,
        "cases_sha256": hashlib.sha256(data).hexdigest(),
        "mirror_sha256": mirror,
        "mirror_notes": count,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_set(folder: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((folder / "manifest.json").read_text())
    cases = [json.loads(line) for line in (folder / "cases.jsonl").read_text().splitlines() if line.strip()]
    return manifest, cases


def format_build(manifest: dict) -> str:
    c = manifest["counts"]
    lines = [
        f"positive cases: {c['positive']} (moved_copy {c['moved_copy']}, content_change {c['content_change']}),"
        f" old notes {c['old_notes']} (merge {c['merge']}, supersede {c['supersede']}),"
        f" leak {c['leak']}, identity key {c['identity_key']}, private old notes {c['private_old']}",
        f"duplicate cases: {c['duplicate']} (targets {c['duplicate_targets']}, leak {c['duplicate_leak']})",
        f"negative cases: {c['negative']}",
        f"superseded_by refs: {c['superseded_by_refs']}, newer notes {c['newer_notes']},"
        f" bare paths resolved {c['bare_resolved']}, dropped {c['bare_dropped']},"
        f" unresolved links {c['unresolved_links']}; supersedes refs {c['supersedes_refs']},"
        f" pairs added {c['supersedes_pairs']}",
        "negative notes that qualify: " + ", ".join(f"D {d}: {n}" for d, n in manifest["negative_qualifying"].items()),
        f"cases sha256: {manifest['cases_sha256']}",
        f"mirror sha256: {manifest['mirror_sha256']} ({manifest['mirror_notes']} notes)",
        f"commit: {manifest['commit']}",
    ]
    return "\n".join(lines)


# --- score -------------------------------------------------------------------------------------------------------


def wilson(hits: int, n: int, z: float = Z) -> tuple[float | None, float | None]:
    """The Wilson score interval of hits out of n."""
    if n == 0:
        return None, None
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def score_cases(cfg: Config, conn: sqlite3.Connection, embedder, cases: list[dict]) -> list[dict]:
    """One record for each positive case and draft form: the candidates and the position of each old note."""
    records = []
    for case in cases:
        if case["kind"] != "positive":
            continue
        for form in FORMS:
            found = find_collisions(cfg, conn, embedder, draft=case["drafts"][form], exclude=case["newer"])
            paths = [c["path"] for c in found]
            old = []
            for o in case["old"]:
                at = paths.index(o["path"]) if o["path"] in paths else None
                old.append(
                    {
                        "path": o["path"],
                        "position": None if at is None else at + 1,
                        "reasons": [] if at is None else found[at]["reasons"],
                    }
                )
            records.append(
                {
                    "id": case["id"],
                    "form": form,
                    "leak": case["leak"],
                    "positive_kind": case["positive_kind"],
                    "hit": all(o["position"] is not None for o in old),
                    "old": old,
                    "candidates": paths,
                }
            )
    return records


def _rate(hits: int, n: int) -> dict:
    low, high = wilson(hits, n)
    return {"hits": hits, "cases": n, "r1": hits / n if n else None, "low": low, "high": high}


def summarize(records: list[dict]) -> dict:
    """R1 with the Wilson interval for each draft form and subset, and for each source (a hit when the source
    found every old note of the case). The share of old notes that each source found is an extra count."""
    summary = {}
    for form in FORMS:
        mine = [r for r in records if r["form"] == form]
        hit = [all(o["position"] is not None for o in r["old"]) for r in mine]
        chosen = {
            "all": [True] * len(mine),
            "no_leak": [not r["leak"] for r in mine],
            "moved_copy": [r["positive_kind"] == "moved_copy" for r in mine],
            "content_change": [r["positive_kind"] == "content_change" for r in mine],
        }
        out = {
            name: _rate(sum(h for h, c in zip(hit, mask, strict=True) if c), sum(mask)) for name, mask in chosen.items()
        }
        olds = [o for r in mine for o in r["old"]]
        out["sources"] = {}
        for source in SOURCES:
            hits = sum(all(source in o["reasons"] for o in r["old"]) for r in mine)
            found = sum(source in o["reasons"] for o in olds)
            out["sources"][source] = {
                **_rate(hits, len(mine)),
                "old_found": found,
                "old_notes": len(olds),
                "old_share": found / len(olds) if olds else None,
            }
        summary[form] = out
    return summary


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def format_summary(summary: dict) -> str:
    lines = [f"{'form':6} {'subset':15} {'hits':>9} {'R1':>6}  95% interval"]
    for form, out in summary.items():
        for name in SUBSETS:
            r = out[name]
            lines.append(
                f"{form:6} {name:15} {r['hits']:>4}/{r['cases']:<4} {_num(r['r1']):>6}"
                f"  {_num(r['low'])} - {_num(r['high'])}"
            )
        for source, s in out["sources"].items():
            lines.append(
                f"{form:6} {source:15} {s['hits']:>4}/{s['cases']:<4} {_num(s['r1']):>6}"
                f"  {_num(s['low'])} - {_num(s['high'])}  old notes {s['old_found']}/{s['old_notes']}"
            )
    return "\n".join(lines)
