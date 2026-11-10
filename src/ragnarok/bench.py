"""The bench: the index of a variant on a corpus, the question sets, `bench run` and `bench compare`."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import Config, variant_hash
from .evaluate import keywords
from .index import IndexMismatch, connect, update
from .metadata import resolve_file
from .search import CANDIDATES, search
from .tools import Tools


def open_variant(cfg: Config, variant: str, corpus: str) -> tuple[Config, sqlite3.Connection]:
    """The configuration and the index of a variant on a corpus.

    The index keeps the hash of the variant settings: a change of the settings does not make the change scan rewrite
    the notes, so an index of other settings is refused. Raise KeyError for an unknown variant or corpus."""
    vcfg = cfg.variant(variant, corpus)
    other_settings = IndexMismatch(
        f"The index at {vcfg.index_path} was built with other settings of the variant {variant}. "
        f"Delete the folder {vcfg.index_path.parent} and run `ragnarok bench index` again."
    )
    try:
        conn = connect(vcfg)
    except IndexMismatch as e:
        # a changed model or dimension: connect() gives the advice of the default index, which does not apply here
        raise other_settings from e
    expected = variant_hash(vcfg)
    stored = conn.execute("SELECT value FROM meta WHERE key = 'variant_hash'").fetchone()
    if stored is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('variant_hash', ?)", (expected,))
        conn.commit()
    elif stored[0] != expected:
        conn.close()
        raise other_settings
    return vcfg, conn


# --- the set format ------------------------------------------------------------------------------------------------

SET_VERSION = 2
KINDS = ("synthetic", "real")
QUESTION_KEYS = ("id", "question", "stratum", "area", "cluster")
LIMIT = 20  # the notes of each search
RECALL_AT = 5
SAMPLES = 10_000
SEED = 0
ALPHA = 0.05
MARGIN = 0.05
P95_MS = 500


def set_hash(questions: list[dict]) -> str:
    """The SHA-256 of the canonical JSON of the question list."""
    text = json.dumps(questions, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def write_set(path: Path, name: str, kind: str, corpus: str, questions: list[dict]) -> None:
    data = {
        "name": name,
        "kind": kind,
        "version": SET_VERSION,
        "corpus": corpus,
        "hash": set_hash(questions),
        "questions": questions,
    }
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def load_set(path: Path) -> dict:
    """A set file of version 2. Raise ValueError when the header, a question or the hash is wrong."""
    data = json.loads(Path(path).read_text())
    for key in ("name", "kind", "version", "corpus", "hash", "questions"):
        if key not in data:
            raise ValueError(f"{path}: the set has no {key!r}")
    if data["version"] != SET_VERSION:
        raise ValueError(f"{path}: the set version is {data['version']!r}, not {SET_VERSION}")
    if data["kind"] not in KINDS:
        raise ValueError(f"{path}: the kind must be one of {', '.join(KINDS)}, not {data['kind']!r}")
    ids = set()
    for q in data["questions"]:
        bad = [k for k in QUESTION_KEYS if not isinstance(q.get(k), str) or not q[k]]
        expected = q.get("expected")
        if not isinstance(expected, list) or not expected or not all(isinstance(e, str) and e for e in expected):
            bad.append("expected")
        if bad:
            raise ValueError(f"{path}: question {q.get('id')!r} has no valid {', '.join(bad)}")
        if q["cluster"] not in expected:
            raise ValueError(f"{path}: the cluster of question {q['id']!r} is not one of its expected paths")
        if q["id"] in ids:
            raise ValueError(f"{path}: the question id {q['id']!r} is not unique")
        ids.add(q["id"])
    if set_hash(data["questions"]) != data["hash"]:
        raise ValueError(f"{path}: the hash of the questions does not agree with the header")
    return data


def convert(old: list[dict], cfg: Config) -> tuple[list[dict], list[tuple[str, str]]]:
    """The questions of a draft set of evaluate.py in the format v2, and the (id, reason) of each question that it
    cannot convert. stratum = the old type, cluster = the first expected path, area = the area that the metadata rules
    give the first expected note on cfg ("unknown" when the note is absent)."""
    questions, skipped = [], []
    for q in old:
        expected = list(q.get("expected") or [])
        if not expected:
            skipped.append((q["id"], "no expected path"))
            continue
        if not all(e.endswith(".md") for e in expected):
            skipped.append((q["id"], "an expected path is a folder, not a note"))
            continue
        first = expected[0]
        area = resolve_file(cfg, first).area if cfg.note_path(first).is_file() else "unknown"
        new = {
            "id": q["id"],
            "question": q["question"],
            "expected": expected,
            "stratum": q.get("type") or "none",
            "area": area,
            "cluster": first,
        }
        if q.get("note"):
            new["notes"] = q["note"]
        questions.append(new)
    return questions, skipped


# --- bench run -----------------------------------------------------------------------------------------------------


class MissingGold(ValueError):
    def __init__(self, missing: list[tuple[str, str, str]]):
        super().__init__(f"{len(missing)} expected paths cannot be found")
        self.missing = missing


def missing_gold(conn: sqlite3.Connection, questions: list[dict]) -> list[tuple[str, str, str]]:
    """The (id, path, reason) of each expected path that the search cannot return: "not in the index" or "private"."""
    sensitivity = dict(conn.execute("SELECT path, sensitivity FROM notes"))
    missing = []
    for q in questions:
        for e in q["expected"]:
            if e not in sensitivity:
                missing.append((q["id"], e, "not in the index"))
            elif sensitivity[e] != "normal":
                missing.append((q["id"], e, "private"))
    return missing


def overlap(cfg: Config, question: str, path: str) -> float:
    """The share of the question keywords that are in the text of the note."""
    kws = keywords(question)
    if not kws:
        return 0.0
    text = cfg.note_path(path).read_text(errors="ignore").lower()
    return round(sum(k in text for k in kws) / len(kws), 3)


def bm25_rank(cfg: Config, conn: sqlite3.Connection, question: str, path: str) -> int | None:
    """The rank of the note in the BM25 list of notes, None when it is not in the list."""
    paths = [h.path for h in search(cfg, conn, None, question, limit=CANDIDATES, mode="fts")]
    return paths.index(path) + 1 if path in paths else None


def result_path(vcfg: Config, bench_set: dict) -> Path:
    """The result file of a variant on a set: a second variant never writes over the first one."""
    return vcfg.data_dir / "bench" / f"{bench_set['hash'][:12]}-{variant_hash(vcfg)}.json"


def run_set(vcfg: Config, conn: sqlite3.Connection, embedder, bench_set: dict, variant: str) -> dict:
    """Each question through Tools.search with limit 20 and the change scan. Raise MissingGold before the first
    question when an expected path is not in the index.

    One update with no limit first makes the index current: the change scan of a tool call takes at most SCAN_LIMIT
    notes, so an index far behind would give false missing paths and would change between the timed questions."""
    update(vcfg, conn, embedder, limit=None)
    tools = Tools(vcfg, conn, embedder)
    if missing := missing_gold(conn, bench_set["questions"]):
        raise MissingGold(missing)
    embedder.embed_query("warm up")  # the times are with the model warm
    rows = []
    for q in bench_set["questions"]:
        start = time.perf_counter()
        out = tools.search(q["question"], limit=LIMIT)
        total = (time.perf_counter() - start) * 1000
        scan = tools.last_scan["ms"]
        paths = [r["path"] for r in out["results"]]
        rows.append(
            {
                "id": q["id"],
                "ranks": [paths.index(e) + 1 if e in paths else None for e in q["expected"]],
                "scan_ms": scan,
                "scan_changed": tools.last_scan["changed"],
                "search_ms": round(max(total - scan, 0.0), 1),
                "total_ms": round(total, 1),
                "overlap": overlap(vcfg, q["question"], q["cluster"]),
                "bm25_rank": bm25_rank(vcfg, conn, q["question"], q["cluster"]),
            }
        )
    return {
        "set": bench_set["name"],
        "set_hash": bench_set["hash"],
        "kind": bench_set["kind"],
        "corpus": bench_set["corpus"],
        "variant": variant,
        "variant_hash": variant_hash(vcfg),
        "limit": LIMIT,
        "questions": rows,
    }


# --- bench compare -------------------------------------------------------------------------------------------------


def measures(ranks: list[int | None]) -> tuple[int, float]:
    """recall@5 (1 when all expected notes are in the first 5) and the reciprocal rank of the first expected note."""
    found = [r for r in ranks if r is not None]
    recall = int(len(found) == len(ranks) and max(found) <= RECALL_AT)
    return recall, 1.0 / min(found) if found else 0.0


def cluster_bootstrap(
    d: list[float], clusters: list[str], alpha: float = ALPHA, n: int = SAMPLES, seed: int = SEED
) -> tuple[float, float, float, float]:
    """The mean of the paired difference d, its 1 - alpha interval and the two-sided p-value, from a bootstrap that
    draws whole clusters with replacement."""
    names = list(dict.fromkeys(clusters))
    pos = {c: i for i, c in enumerate(names)}
    sums, counts = np.zeros(len(names)), np.zeros(len(names))
    for value, c in zip(d, clusters, strict=True):
        sums[pos[c]] += value
        counts[pos[c]] += 1
    idx = np.random.default_rng(seed).integers(0, len(names), (n, len(names)))
    s = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(float((s <= 0).mean()), float((s >= 0).mean())))
    return float(np.mean(d)), float(lo), float(hi), p


def holm(pvalues: list[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """The Holm level and the decision of each hypothesis. The i-th smallest p of m gets alpha / (m - i); after the
    first hypothesis that is kept, all later ones are kept."""
    m = len(pvalues)
    levels, rejected = [alpha] * m, [False] * m
    going = True
    for i, j in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        levels[j] = alpha / (m - i)
        going = going and pvalues[j] <= levels[j]
        rejected[j] = going
    return levels, rejected


def _sign(x: float) -> str:
    return "+" if x > 0 else "-" if x < 0 else "0"


def _p95(result: dict) -> float:
    return float(np.percentile([r["total_ms"] for r in result["questions"]], 95))


def compare(bench_set: dict, results: list[tuple[str, dict]], margin: float = MARGIN) -> list[str]:
    """The report of bench compare: results[0] is A, the others are B. It ends with one verdict line for each B."""
    qs = bench_set["questions"]
    ids = [q["id"] for q in qs]
    clusters = [q["cluster"] for q in qs]
    groups: dict[str, dict[str, list[int]]] = {"stratum": defaultdict(list), "area": defaultdict(list)}
    vaults: dict[str, list[int]] = defaultdict(list)
    for i, q in enumerate(qs):
        groups["stratum"][q["stratum"]].append(i)
        groups["area"][q["area"]].append(i)
        vaults[q["cluster"].split("/", 1)[0]].append(i)
    synthetic = bench_set["kind"] == "synthetic"
    lines = [
        f"# set {bench_set['name']} ({bench_set['kind']}, {len(qs)} questions, {len(set(clusters))} clusters), "
        f"corpus {bench_set['corpus']}"
    ]
    table = []  # (name, recall@5 of each question, reciprocal rank of each question, result rows by id)
    for name, result in results:
        rows = {r["id"]: r for r in result["questions"]}
        scores = [measures(rows[i]["ranks"]) for i in ids]
        rec = np.array([s[0] for s in scores], dtype=float)
        mrr = np.array([s[1] for s in scores])
        table.append((name, rec, mrr, rows))
        lines.append(f"variant {name}: recall@5 {rec.mean():.3f} MRR {mrr.mean():.3f} p95 {_p95(result):.0f} ms")
        for kind, by_value in groups.items():
            for value, at in sorted(by_value.items()):
                lines.append(f"  {kind} {value} n={len(at)}: recall@5 {rec[at].mean():.3f} MRR {mrr[at].mean():.3f}")
    a_name, a_rec, a_mrr, a_rows = table[0]
    # for each measure, the interval of each B at its Holm level among the B variants
    stats: dict[str, list[tuple]] = {}
    for m, col, base in (("recall@5", 1, a_rec), ("MRR", 2, a_mrr)):
        diffs = [list(row[col] - base) for row in table[1:]]
        levels, rejected = holm([cluster_bootstrap(d, clusters)[3] for d in diffs])
        stats[m] = [
            (*cluster_bootstrap(d, clusters, alpha=level), level, rej)
            for d, level, rej in zip(diffs, levels, rejected, strict=True)
        ]
    verdicts = []
    for k, (b_name, b_rec, b_mrr, b_rows) in enumerate(table[1:]):
        lines.append(f"## {b_name} - {a_name}")
        for m in ("recall@5", "MRR"):
            mean, lo, hi, p, level, _ = stats[m][k]
            lines.append(f"  {m} B-A {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] at level {level:.3f}, p {p:.3f}")
        changed = [i for i in range(len(ids)) if b_rec[i] != a_rec[i] or b_mrr[i] != a_mrr[i]]
        lines.append(
            f"  discordant {len(changed) / len(ids):.2f} "
            f"(recall@5 {np.mean(b_rec != a_rec):.2f}, MRR {np.mean(b_mrr != a_mrr):.2f})"
        )
        for kind, by_value in groups.items():
            for value, at in sorted(by_value.items()):
                dr, dm = (b_rec[at] - a_rec[at]).mean(), (b_mrr[at] - a_mrr[at]).mean()
                lines.append(f"  {kind} {value} n={len(at)}: recall@5 B-A {dr:+.3f} MRR B-A {dm:+.3f}")
        if synthetic:
            for value, at in sorted(vaults.items()):
                dr, dm = (b_rec[at] - a_rec[at]).mean(), (b_mrr[at] - a_mrr[at]).mean()
                lines.append(f"  vault {value}: recall@5 {_sign(dr)} MRR {_sign(dm)}")
        for i in changed:
            a, b = a_rows[ids[i]], b_rows[ids[i]]
            lines.append(
                f"  changed {ids[i]}: ranks {a['ranks']} -> {b['ranks']}, overlap {b['overlap']:.2f}, "
                f"BM25 rank of the gold {a['bm25_rank']} -> {b['bm25_rank']}"
            )
        p95 = _p95(results[k + 1][1])
        latency = f"p95 {p95:.0f} ms {'<' if p95 < P95_MS else '>='} {P95_MS} ms, scan included"
        rec_s, mrr_s = stats["recall@5"][k], stats["MRR"][k]
        if synthetic:
            above = {"recall@5": rec_s[5] and rec_s[1] > 0, "MRR": mrr_s[5] and mrr_s[1] > 0}
            win = p95 < P95_MS and ((above["recall@5"] and mrr_s[0] >= 0) or (above["MRR"] and rec_s[0] >= 0))
            why = ", ".join(f"{m} interval {'above' if ok else 'not above'} 0" for m, ok in above.items())
            verdicts.append(f"verdict {b_name}: {'win' if win else 'no win'} ({why}; {latency})")
        else:
            ok = rec_s[1] > -margin and mrr_s[1] > -margin
            verdicts.append(
                f"verdict {b_name}: no win (a real set does not decide a win); "
                f"no regression: {'pass' if ok else 'fail'} (lower ends recall@5 {rec_s[1]:+.3f}, "
                f"MRR {mrr_s[1]:+.3f} against -{margin:.3f}); latency: {'pass' if p95 < P95_MS else 'fail'} ({latency})"
            )
    return lines + verdicts
