"""Mechanical evaluation: grep, FTS5, dense and hybrid search on the draft set and the hard set.

Draft set: recall@K (share of expected notes in the top K, notes with the same file name count as one) and any-hit@K.
K is 10 for synthesis questions, else 5.
Hard set, continuation: project hit@K (share of expected prefixes with a note in the top K) and primary hit@K
(the first prefix has a note in the top K). Hard set, category: area precision@K and area any-hit@K.
Hard set, both types: note precision@K (share of the top K that the user checked in the pool).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metadata import is_private, iter_notes
from .search import STOP, search

METHODS = ["grep", "fts", "dense", "hybrid"]
PAIRS = [("hybrid", "grep"), ("hybrid", "fts"), ("hybrid", "dense"), ("dense", "fts")]
DEPTH = 50


def excluded(cfg, rel: str) -> bool:
    """A note the evaluation ignores (eval_exclude), for example a note that quotes the questions."""
    return rel.startswith(tuple(cfg.eval_exclude))


def keywords(query: str) -> list[str]:
    return [w for w in re.findall(r"[\w\-\.]+", query.lower()) if w not in STOP and len(w) > 2]


class Grep:
    """The baseline: rank notes by the number of distinct query words they contain, then by the total hits."""

    def __init__(self, cfg):
        self.texts = {
            rel: (cfg.root / rel).read_text(errors="ignore").lower()
            for rel in iter_notes(cfg)
            if not is_private(cfg, rel)
        }

    def rank(self, query: str) -> list[str]:
        kws = keywords(query)
        scored = []
        for rel, text in self.texts.items():
            hits = [text.count(k) for k in kws]
            n = sum(1 for h in hits if h)
            if n:
                scored.append((n, sum(hits), rel))
        scored.sort(reverse=True)
        return [rel for _, _, rel in scored[:DEPTH]]


def rankings(cfg, conn, embedder, grep: Grep, query: str) -> dict[str, list[str]]:
    out = {"grep": grep.rank(query)}
    for mode in ("fts", "dense", "hybrid"):
        out[mode] = [h.path for h in search(cfg, conn, embedder, query, limit=DEPTH + 20, mode=mode)]
    return {m: [p for p in ranked if not excluded(cfg, p)][:DEPTH] for m, ranked in out.items()}


def inside(note: str, prefix: str) -> bool:
    return note == prefix or note.startswith(prefix.rstrip("/") + "/")


def score_draft(q: dict, ranked: dict[str, list[str]], known: set[str], methods=METHODS) -> dict | None:
    expected = [e for e in q.get("expected", []) if e in known]
    if not expected:
        return None
    groups = defaultdict(list)
    for e in expected:
        groups[e.rsplit("/", 1)[-1]].append(e)
    k = 10 if (q.get("type") or "").startswith("synthesis") else 5
    out = {"id": q["id"], "type": q.get("type"), "k": k}
    for m in methods:
        top = ranked[m][:k]
        out[f"{m}_recall"] = sum(any(e in top for e in g) for g in groups.values()) / len(groups)
        out[f"{m}_any"] = float(any(e in top for e in expected))
        out[f"{m}_top"] = ranked[m][:5]
    return out


def score_hard(q: dict, ranked: dict[str, list[str]], methods=METHODS) -> dict:
    exp = q["expected"]
    out = {"id": q["id"], "type": q["type"], "subtype": q.get("subtype"), "grep_finds": q.get("grep_finds")}
    for m in methods:
        for k in (5, 10):
            top = ranked[m][:k]
            if q["type"] == "continuation":
                out[f"{m}_proj@{k}"] = float(np.mean([any(inside(n, p) for n in top) for p in exp]))
                out[f"{m}_prim@{k}"] = float(any(inside(n, exp[0]) for n in top))
            else:
                out[f"{m}_prec@{k}"] = sum(any(inside(n, p) for p in exp) for n in top) / k if top else 0.0
                out[f"{m}_any@{k}"] = float(any(inside(n, p) for n in top for p in exp))
            # note-level: the notes the user checked in the pool, not every note in the expected folders
            if "relevant" in q:
                out[f"{m}_nprec@{k}"] = sum(n in q["relevant"] for n in top) / k if top else 0.0
        out[f"{m}_top"] = ranked[m][:5]
    return out


def bootstrap(a: list[float], b: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the paired difference a - b, with the 95% bootstrap interval."""
    d = np.array(a) - np.array(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    s = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _table(rows: list[dict], measure: str, title: str) -> list[str]:
    lines = [
        f"{title} n={len(rows)}",
        "  " + "  ".join(f"{m} {np.mean([r[f'{m}_{measure}'] for r in rows]):.2f}" for m in METHODS),
    ]
    for a, b in PAIRS:
        d, lo, hi = bootstrap([r[f"{a}_{measure}"] for r in rows], [r[f"{b}_{measure}"] for r in rows])
        lines.append(f"  {a}-{b}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return lines


def report_draft(results: list[dict], name: str) -> list[str]:
    lines = [f"## {name}: recall@K and any-hit@K"]
    lines += _table(results, "recall", "recall all")
    lines += ["  any-hit " + "  ".join(f"{m} {np.mean([r[f'{m}_any'] for r in results]):.2f}" for m in METHODS)]
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    for t, rs in sorted(by_type.items(), key=lambda x: str(x[0])):
        lines.append(
            f"  [{t} n={len(rs)}] recall "
            + "  ".join(f"{m} {np.mean([r[f'{m}_recall'] for r in rs]):.2f}" for m in METHODS)
        )
    return lines


def report_hard(results: list[dict]) -> list[str]:
    lines = ["## hard set"]
    for typ, measures in (("continuation", ["prim@5", "proj@5", "proj@10"]), ("category", ["prec@5", "any@5"])):
        rs = [r for r in results if r["type"] == typ]
        if not rs:
            continue
        lines += _table(rs, measures[0], f"{typ} {measures[0]}")
        if all("hybrid_nprec@5" in r for r in rs):
            lines += _table(rs, "nprec@5", f"{typ} nprec@5")
        for ms in measures[1:]:
            lines.append(f"  {ms} " + "  ".join(f"{m} {np.mean([r[f'{m}_{ms}'] for r in rs]):.2f}" for m in METHODS))
        for sub in sorted({str(r["subtype"]) for r in rs}):
            ss = [r for r in rs if str(r["subtype"]) == sub]
            lines.append(
                f"  [{sub} n={len(ss)}] {measures[0]} "
                + "  ".join(f"{m} {np.mean([r[f'{m}_{measures[0]}'] for r in ss]):.2f}" for m in METHODS)
            )
    return lines


def mechanical(cfg, conn, embedder, sets_dir: Path, out_dir: Path | None = None) -> list[str]:
    grep = Grep(cfg)
    known = {p for (p,) in conn.execute("SELECT path FROM notes WHERE sensitivity = 'normal'")}
    lines: list[str] = []
    all_results = {}
    for path in sorted(sets_dir.glob("*.json")):
        questions = json.loads(path.read_text())
        is_hard = all("subtype" in q for q in questions)
        results = []
        for q in questions:
            ranked = rankings(cfg, conn, embedder, grep, q["question"])
            r = score_hard(q, ranked) if is_hard else score_draft(q, ranked, known)
            if r:
                results.append(r)
        all_results[path.stem] = results
        lines += report_hard(results) if is_hard else report_draft(results, path.stem)
        lines.append("")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"mechanical-min{cfg.min_words}.json").write_text(json.dumps(all_results, indent=1))
    return lines


def pool(cfg, conn, embedder, sets_dir: Path, out: Path, k: int = 5) -> int:
    """Write the union of the top K notes of each method for each question, for the user to mark."""
    grep = Grep(cfg)
    rows = []
    for path in sorted(sets_dir.glob("*.json")):
        for q in json.loads(path.read_text()):
            ranked = rankings(cfg, conn, embedder, grep, q["question"])
            found = {}
            for m in METHODS:
                for n in ranked[m][:k]:
                    found.setdefault(n, []).append(m)
            rows.append(
                {
                    "set": path.stem,
                    "id": q["id"],
                    "question": q["question"],
                    "expected": q.get("expected", []),
                    "pool": [{"path": n, "methods": ms, "relevant": None} for n, ms in found.items()],
                }
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    return len(rows)


def pool_markdown(rows: list[dict]) -> str:
    """A checklist for the user: one section per question id, the expected notes already checked."""
    by_id: dict[str, dict] = {}
    for r in rows:
        q = by_id.setdefault(r["id"], {"questions": [], "expected": r["expected"], "pool": {}})
        q["questions"].append(r["question"])
        for p in r["pool"]:
            q["pool"].setdefault(p["path"], set()).update(p["methods"])
    lines = [
        "# Pooled check",
        "",
        "Check each note that answers the question. Remove the check from a note that does not answer it.",
        "For a hard-set question, check a note when it is in the correct project or area.",
        "",
    ]
    for qid, q in by_id.items():
        lines.append(f"## {qid}")
        lines += [f"> {text}" for text in dict.fromkeys(q["questions"])]
        lines.append("")
        exp = q["expected"]
        paths = dict.fromkeys([*q["pool"], *[e for e in exp if e.endswith(".md")]])
        for path in paths:
            checked = any(inside(path, e) for e in exp)
            methods = ", ".join(m for m in METHODS if m in q["pool"].get(path, set())) or "expected only"
            lines.append(f"- [{'x' if checked else ' '}] `{path}` ({methods})")
        lines.append("")
    return "\n".join(lines)


def read_pool_markdown(text: str) -> dict[str, list[str]]:
    """The checked paths of each question id."""
    marks: dict[str, list[str]] = {}
    qid = None
    for line in text.split("\n"):
        if line.startswith("## "):
            qid = line[3:].strip()
            marks[qid] = []
        m = re.match(r"- \[[xX]\] `([^`]+)`", line)
        if m and qid:
            marks[qid].append(m.group(1))
    return marks
