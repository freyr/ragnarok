"""The KB evaluation: the measures of the question classes, the mechanical KB runner, the filtered no-regression
runner and the attribution of a failed run to one stage.

The question sets are JSON lists in <data_dir>/eval/sets-kb/<class>.json. A question has `id`, `class`
(single-truth, join, live, negative), `question` and `question_alt`, `gold` (paths with a grade),
`gold_eligibility` (search filters that admit the gold), `gold_route`, `gold_entities`, `gold_claims`,
`distractors` (paths with a kind: stale, duplicate, homonym) and `origin`.

A question of a kind of question has `class` now, future or history, `when` (the same value), `question`,
`question_alt`, `answer`, `gold` (paths with a grade; it can be empty) and `wrong` (paths). Its measures are
answer-first and wrong (wrong@5 in the mechanical runner, wrong citation for an agent). A question with empty gold
stays in the runs: its answer-first is 0.

The unit of analysis is the question: its value is the mean of its runs, and the paired bootstrap resamples the
questions. A RUN run (it crashed, timed out, or gave no final JSON object) gets the worst value of each measure.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import kb
from .agent_eval import gold_paths, is_run_failure, read_run
from .config import Config
from .evaluate import bootstrap, score_draft
from .search import CANDIDATES, search
from .trace import find_lines

MEASURES = ("canonical_first", "stale", "f1", "route", "no_answer", "answer_first", "wrong")
WORST = {
    "canonical_first": 0.0,
    "stale": 1.0,
    "f1": 0.0,
    "route": 0.0,
    "no_answer": 0.0,
    "answer_first": 0.0,
    "wrong": 1.0,
    "run": True,
}
STALE_TOP = 5  # the mechanical stale citation: a stale distractor in the top 5
WRONG_TOP = 5  # the mechanical wrong@5: a wrong note in the top 5
KB_LIMIT = 10  # the documents that the mechanical KB runner asks for
KINDS = ("now", "future", "history")  # the classes of a kind of question
# the systems of the mechanical runner: the baseline over MCP, the new code with no `when` (the rule list decides),
# the new code with the gold eligibility as filters, the new code with the `when` of the question (a diagnostic)
SYSTEMS = ("base", "kb", "kb-eligibility", "kb-when")
DEFAULT_SYSTEMS = ("base", "kb", "kb-eligibility")
# the mechanical measures: title and sign (+1: a higher value is better, -1: a lower value is better)
MECHANICAL = {
    "canonical_first": ("canonical-first", 1),
    "stale": (f"stale@{STALE_TOP}", -1),
    "answer_first": ("answer-first", 1),
    "wrong": (f"wrong@{WRONG_TOP}", -1),
}


class Matcher:
    """The one name matcher of both systems: an entity ID, or a name or an alias through `entity_alias` of the new
    core. A name with no entity, or with more than one, stays as its normalized text, so it matches no ID."""

    def __init__(self, cfg: Config, conn=None):
        self.cfg, self.conn = cfg, conn

    def __call__(self, names) -> set[str]:
        if self.conn is None:
            from .index import connect

            self.conn = connect(self.cfg)
        out = set()
        for name in names:
            candidates = kb.resolve(self.conn, str(name), self.cfg.allow_private)
            out.add(candidates[0]["id"] if len(candidates) == 1 else kb.normalize_alias(str(name)))
        return out


def top_gold(q: dict) -> set[str]:
    """The gold paths of the highest grade."""
    grades = [g.get("grade", 1) for g in q.get("gold", [])]
    return {g["path"] for g in q.get("gold", []) if g.get("grade", 1) == max(grades)}


def stale_paths(q: dict) -> set[str]:
    return {d["path"] for d in q.get("distractors", []) if d.get("kind") == "stale"}


def wrong_paths(q: dict) -> set[str]:
    return set(q.get("wrong", []))


def score_ranked(q: dict, ranked: list[str]) -> dict:
    """The mechanical measures of one returned list: canonical-first and a stale distractor in the top 5; for a
    kind of question, answer-first (rank 1 is a top gold) and a wrong note in the top 5."""
    first = float(bool(ranked) and ranked[0] in top_gold(q))
    if q.get("class") in KINDS:
        return {"answer_first": first, "wrong": float(bool(wrong_paths(q) & set(ranked[:WRONG_TOP])))}
    return {
        "canonical_first": first,
        "stale": float(bool(stale_paths(q) & set(ranked[:STALE_TOP]))),
    }


def f1(answer: set[str], gold: set[str]) -> float:
    if not answer and not gold:
        return 1.0
    return 2 * len(answer & gold) / (len(answer) + len(gold))


def score_run(q: dict, run: dict, match) -> dict:
    """The measures of one agent run. A RUN run gets the worst value of each measure."""
    if is_run_failure(run):
        return dict(WORST)
    notes, entities = run["notes"], run["entities"]
    return {
        "canonical_first": float(bool(notes) and notes[0] in top_gold(q)),
        "stale": float(bool(stale_paths(q) & set(notes))),
        "f1": f1(match(entities), match(q.get("gold_entities", []))),
        "route": float(run["route"] == q.get("gold_route")),
        "no_answer": float(not notes and not entities),
        "answer_first": float(bool(notes) and notes[0] in top_gold(q)),
        "wrong": float(bool(wrong_paths(q) & set(notes))),
        "run": False,
    }


def failed(q: dict, score: dict) -> bool:
    """A failed run, by the definition of its class."""
    cls = q["class"]
    if cls == "single-truth":
        return score["canonical_first"] == 0 or score["stale"] == 1
    if cls == "join":
        return score["f1"] < 1.0
    if cls == "live":
        return score["route"] == 0
    if cls == "negative":
        return score["no_answer"] == 0
    raise ValueError(f"unknown question class {cls!r}")


def question_value(scores: list[dict]) -> dict:
    """The value of a question: the mean of each measure over its runs."""
    return {m: float(np.mean([s[m] for s in scores])) for m in MEASURES}


# --- the agent report ----------------------------------------------------------------------------------------------

# the rows of each class: measure, title, sign (+1: test minus reference is the gain, -1: reference minus test)
ROWS = {
    "single-truth": [("canonical_first", "Canonical-first", 1), ("stale", "Stale citation", -1)],
    "join": [("f1", "Set F1", 1)],
    "live": [("route", "Correct route", 1)],
    "negative": [("route", "Correct route", 1), ("no_answer", "No-answer accuracy", 1)],
    **{kind: [("answer_first", "Answer-first", 1), ("wrong", "Wrong citation", -1)] for kind in KINDS},
}
AGENT = {measure: (title, sign) for rows in ROWS.values() for measure, title, sign in rows}


def agent_values(sets: dict[str, list[dict]], results: dict, agents: list[str], match) -> dict[str, list[dict]]:
    """The rows of each agent: for each question that each agent ran, id, set, class and the mean over the runs
    of each measure of its class. Run n of one agent pairs with run n of the other."""
    values: dict[str, list[dict]] = {a: [] for a in agents}
    for name, questions in sets.items():
        for q in questions:
            r = results.get(name, {}).get(q["id"])
            if not r or any(not r.get(a) for a in agents):
                continue
            n = min(len(r[a]) for a in agents)
            for a in agents:
                value = question_value([score_run(q, run, match) for run in r[a][:n]])
                measures = {m: value[m] for m, _, _ in ROWS[q["class"]]}
                values[a].append({"id": q["id"], "set": name, "class": q["class"], **measures})
    return values


def write_per_question(
    path: Path, runner: str, measures: dict, values: dict[str, list[dict]], extra: dict | None = None
) -> None:
    """The per-question file of a run, for `evaluate compare`: the runner, the title and sign of each measure, the
    rows of each system in the order of the run, and the keys of extra."""
    used = {m for rows in values.values() for r in rows for m in r}
    data = {"runner": runner, "measures": {m: list(t) for m, t in measures.items() if m in used}, "systems": values}
    data.update(extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def report_rows(sets: dict[str, list[dict]], results: dict, agents: list[str], match) -> list[str]:
    """The table rows of the KB measures: for each class, the mean of each agent and the paired difference with
    its 95% interval. Canonical-first and stale citation are condition 1 (both intervals above 0); set F1 is
    condition 2."""
    test, ref = agents[0], agents[-1]
    if test == ref:
        return [
            f"KB rows need two agents (test and baseline) in one run; got {', '.join(agents)}: no paired difference"
        ]
    values = agent_values(sets, results, agents, match)
    by_class: dict[str, list[dict]] = {}
    for i, row in enumerate(values[test]):
        by_class.setdefault(row["class"], []).append({a: values[a][i] for a in agents})
    lines = []
    for cls, rows in by_class.items():
        for measure, title, sign in ROWS[cls]:
            a, b = (test, ref) if sign > 0 else (ref, test)
            d, lo, hi = bootstrap([r[a][measure] for r in rows], [r[b][measure] for r in rows])
            means = ", ".join(f"{x} {np.mean([r[x][measure] for r in rows]):.2f}" for x in (ref, test))
            lines.append(
                f"| {title}, {cls} (n={len(rows)}) | {a} - {b}, interval above 0 | {means} | "
                f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}] |"
            )
    return lines


# --- the baseline over MCP ----------------------------------------------------------------------------------------


def mcp_caller(checkout: Path, home: Path):
    """A function that makes a list of tool calls on the baseline server (MCP over stdio, one session) and gives
    the result of each: the JSON object, or {"error": text} for a tool error."""

    def call(batch: list[tuple[str, dict]]) -> list[dict]:
        import anyio
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command="uv",
            args=["--directory", str(checkout), "run", "ragnarok", "serve"],
            env={"RAGNAROK_HOME": str(home)},
        )

        async def main() -> list[dict]:
            out = []
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                for tool, arguments in batch:
                    result = await session.call_tool(tool, arguments)
                    text = "".join(getattr(c, "text", "") for c in result.content)
                    out.append({"error": text} if result.is_error else json.loads(text))
            return out

        return anyio.run(main)

    return call


# --- the mechanical KB runner -------------------------------------------------------------------------------------


def _forms(q: dict) -> list[str]:
    """The two query forms of a question: the two runs of the mechanical runner."""
    return [q["question"], *([q["question_alt"]] if q.get("question_alt") else [])]


def _eligibility(q: dict) -> dict:
    return {("class_" if k == "class" else k): v for k, v in (q.get("gold_eligibility") or {}).items()}


def _paths(hits) -> list[str]:
    return [h.path for h in hits]


def _listed(q: dict) -> list[str]:
    """The gold, distractor and wrong paths of a question."""
    return [*gold_paths(q), *(d["path"] for d in q.get("distractors", [])), *q.get("wrong", [])]


def kb_mechanical(
    ecfg: Config,
    conn,
    embedder,
    sets_dir: Path,
    base_call,
    systems=DEFAULT_SYSTEMS,
    out: Path | None = None,
    collect: list | None = None,
) -> list[str]:
    """The mechanical measures of each system in systems, for each set of sets_dir: canonical-first and stale@5
    for a single-truth question, answer-first and wrong@5 for a kind of question.

    The systems: base, the baseline through its tool calls (base_call; None when base is not in systems); kb,
    the new code with the query only, so the rule list decides the kind of question; kb-eligibility, the new code
    with the gold eligibility as filters; kb-when, the new code with the `when` of the question (a diagnostic).
    The two query forms are the two runs of a question. A question with no gold stays out, except a kind of
    question. Each gold, distractor or wrong path that a system does not hold gives a MEASURE line. With out, write
    the per-question file there. With collect, append the returned paths of each query form and system to it (set,
    id, system, form, paths).
    """
    systems = list(systems)
    if unknown := [s for s in systems if s not in SYSTEMS]:
        raise ValueError(f"unknown systems {', '.join(unknown)}; the systems are {', '.join(SYSTEMS)}")
    if "base" in systems and base_call is None:
        raise ValueError("the system base needs the baseline")
    new_code = any(s != "base" for s in systems)
    known = {p for (p,) in conn.execute("SELECT path FROM doc")} if new_code else set()
    lines: list[str] = []
    values: dict[str, list[dict]] = {s: [] for s in systems}
    for path in sorted(sets_dir.glob("*.json")):
        questions = json.loads(path.read_text())
        listed = sorted({p for q in questions for p in _listed(q)})
        graded = [q for q in questions if q.get("gold") or q.get("class") in KINDS]
        held: set[str] = set()
        base = iter(())
        if "base" in systems:
            searches = [("search", {"query": form, "limit": KB_LIMIT}) for q in graded for form in _forms(q)]
            answers = base_call([("open_note", {"path": p, "outline": True}) for p in listed] + searches)
            held = {p for p, a in zip(listed, answers, strict=False) if "error" not in a}
            base = iter(answers[len(listed) :])
        checks = [*([("base", held)] if "base" in systems else []), *([("kb", known)] if new_code else [])]
        for q in questions:
            for p in _listed(q):
                lines += [f"MEASURE {q['id']} {p} absent from {s}" for s, h in checks if p not in h]
        rows = []
        for q in graded:
            runs: dict[str, list[dict]] = {s: [] for s in systems}
            for form in _forms(q):
                ranked: dict[str, list[str]] = {}
                if "base" in systems:
                    result = next(base)
                    if "error" in result:
                        lines.append(f"ERROR {q['id']} baseline search: {result['error']}")
                    ranked["base"] = [h["path"] for h in result.get("results", [])]
                if "kb" in systems:
                    ranked["kb"] = _paths(search(ecfg, conn, embedder, form, limit=KB_LIMIT))
                if "kb-eligibility" in systems:
                    try:
                        hits = search(ecfg, conn, embedder, form, limit=KB_LIMIT, **_eligibility(q))
                    except LookupError as e:  # a gold subject that does not resolve
                        lines.append(f"QUERY {q['id']} gold_eligibility: {e}")
                        hits = []
                    ranked["kb-eligibility"] = _paths(hits)
                if "kb-when" in systems:
                    ranked["kb-when"] = _paths(search(ecfg, conn, embedder, form, limit=KB_LIMIT, when=q.get("when")))
                for s, paths in ranked.items():
                    runs[s].append(score_ranked(q, paths))
                    if collect is not None:
                        collect.append({"set": path.stem, "id": q["id"], "system": s, "form": form, "paths": paths})
            row = {s: {m: float(np.mean([r[m] for r in rs])) for m in rs[0]} for s, rs in runs.items()}
            rows.append(row)
            for s in systems:
                values[s].append({"id": q["id"], "set": path.stem, "class": q.get("class"), **row[s]})
        measures = list(score_ranked(graded[0], [])) if graded else ["canonical_first", "stale"]
        lines.append(f"## {path.stem}: {' and '.join(MECHANICAL[m][0] for m in measures)}, n={len(rows)}")
        if not rows:
            continue
        for measure in measures:
            title, sign = MECHANICAL[measure]
            lines.append(
                f"  {title} " + "  ".join(f"{s} {np.mean([r[s][measure] for r in rows]):.2f}" for s in rows[0])
            )
            if "base" not in systems:
                continue  # a system of another home pairs with this one in `evaluate compare`
            for system in [s for s in systems if s != "base"]:
                a, b = (system, "base") if sign > 0 else ("base", system)
                d, lo, hi = bootstrap([r[a][measure] for r in rows], [r[b][measure] for r in rows])
                note = " (diagnostic)" if system == "kb-when" else ""
                lines.append(f"    {a} - {b}{note}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    if out is not None:
        write_per_question(out, "mechanical", MECHANICAL, values)
    return lines


# --- the filtered no-regression runner ----------------------------------------------------------------------------


def read_path_map(text: str) -> dict[str, list[str]]:
    """The path map of a conversion: one line for each old path, `<old> → <new>, <new>` (or `->`). A path in
    backticks is one path, also with a comma in it; with no backticks, a comma separates the new paths. A leading
    `- `, blank lines and lines that start with # are ignored."""
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix("- ").strip()
        if not line or line.startswith("#"):
            continue
        sep = "→" if "→" in line else "->"
        if sep not in line:
            raise ValueError(f"a path map line with no → or ->: {raw!r}")
        old, new = line.split(sep, 1)
        quoted = re.findall(r"`([^`]+)`", new)
        news = quoted if quoted else new.split(",")
        out[old.replace("`", "").strip()] = [p.strip() for p in news if p.strip()]
    return out


def filtered(
    ecfg: Config,
    conn,
    embedder,
    questions: list[dict],
    base_call,
    new_search=search,
    path_map: dict[str, list[str]] | None = None,
):
    """Recall@K of the baseline and of the new code (authority off) with the filters of the first expected note.

    The questions are those with an expected note in the index. K is the K of score_draft. The filters are vault,
    project and area of the first expected note, and its type only when both systems give the same type to each
    expected note. Return the lines and the ids of the questions where the new recall is lower.

    path_map (old path to new paths) maps an expected note that a conversion moved or split: each of its new paths
    in the index stands for it, a returned new path counts as the old note, and the filters come from its first
    new path.
    """
    known = {p for (p,) in conn.execute("SELECT path FROM doc WHERE sensitivity = 'normal'")}
    path_map = path_map or {}

    def resolve(e: str) -> list[str]:
        """The paths in the index that stand for an expected note."""
        return [*([e] if e in known else []), *(n for n in path_map.get(e, []) if n in known and n != e)]

    olds = dict.fromkeys(e for q in questions for e in q.get("expected", []) if e in path_map)
    absent = [f"MAP {e} {n} absent from the index" for e in olds for n in path_map[e] if n not in known]
    questions = [q for q in questions if any(resolve(e) for e in q.get("expected", []))]
    expected = {q["id"]: [e for e in q["expected"] if resolve(e)] for q in questions}
    paths = sorted({p for es in expected.values() for e in es for p in resolve(e)})
    meta = {}
    for p in paths:
        row = conn.execute("SELECT vault, project, area, type FROM doc WHERE path = ?", (p,)).fetchone()
        meta[p] = dict(zip(("vault", "project", "area", "type"), row, strict=True))
    base_meta = dict(zip(paths, base_call([("open_note", {"path": p, "outline": True}) for p in paths]), strict=True))
    plans = []
    for q in questions:
        first = meta[resolve(expected[q["id"]][0])[0]]
        filters = {k: first[k] for k in ("vault", "project", "area") if first[k] is not None}
        resolved = [p for e in expected[q["id"]] for p in resolve(e)]
        if first["type"] is not None and all(base_meta[p].get("type") == meta[p]["type"] for p in resolved):
            filters["type"] = first["type"]
        k = 10 if (q.get("type") or "").startswith("synthesis") else 5
        plans.append((q, filters, k))
    base = base_call([("search", {"query": q["question"], **f, "limit": k}) for q, f, k in plans])
    lines, regressed = [f"# filtered recall@K, baseline and new code (authority off), n={len(plans)}", *absent], []
    for (q, f, k), result in zip(plans, base, strict=True):
        # a returned path that stands for an expected note counts as that note (the first one, for a merge)
        alias = {p: e for e in reversed(expected[q["id"]]) for p in resolve(e)}
        ranked = {
            "base": [alias.get(h["path"], h["path"]) for h in result.get("results", [])],
            "new": [
                alias.get(p, p)
                for p in _paths(new_search(ecfg, conn, embedder, q["question"], **f, limit=k, authority=False))
            ],
        }
        if "error" in result:  # a failed baseline search is no measurement: the question cannot pass
            lines.append(f"ERROR {q['id']} baseline search: {result['error']}")
            regressed.append(q["id"])
            continue
        s = score_draft(q, ranked, known | set(expected[q["id"]]), ["base", "new"])
        lower = s["new_recall"] < s["base_recall"]
        if lower:
            regressed.append(q["id"])
        lines.append(
            f"{q['id']} K={k} base {s['base_recall']:.2f} new {s['new_recall']:.2f}{' LOWER' if lower else ''}"
            f" filters {json.dumps(f, ensure_ascii=False, sort_keys=True)}"
        )
    lines.append(f"regressed: {', '.join(regressed) or 'none'}")
    return lines, regressed


# --- attribution --------------------------------------------------------------------------------------------------

ORDER = ["S2", "S3", "S4", "S5", "S6", "S7", "S8"]
NEEDS_EVIDENCE = {"S7", "S8"}
# the stage of a note or an entity that a tool gave, for a negative question
TOOL_STAGE = {
    "search": "S6",
    "find_projects": "S6",
    "list_project": "S6",
    "open_projects": "S6",
    "find_by_jira": "S6",
    "open_note": "S7",
    "Read": "S7",
    "find_entity": "S3",
    "get_entity": "S3",
    "related": "S4",
}


def _name(tool: dict) -> str:
    """The tool name without the MCP prefix."""
    return str(tool.get("name") or "").rsplit("__", 1)[-1]


def _ragnarok(run: dict) -> bool:
    return any(str(t.get("name") or "").startswith("mcp__ragnarok__") for t in run["tools"])


def _docs(record: dict) -> list[str]:
    return [d.get("path") for d in record.get("gen_ai.retrieval.documents") or []]


def _reach(candidate: dict) -> int:
    """The last stage a gold candidate of one search call passed: 3 (not in a pool before the filters),
    4 (removed by a filter), 5 (passed the filters)."""
    ranks = (candidate.get("prefilter_dense_rank"), candidate.get("prefilter_fts_rank"))
    if not any(r is not None and r <= CANDIDATES for r in ranks):
        return 3
    return 4 if not all((candidate.get("filters") or {}).values()) else 5


def _single_truth(q: dict, run: dict, records: list[dict]) -> tuple[str, str | None]:
    gold, stale = top_gold(q), stale_paths(q)
    if not _ragnarok(run) or run["route"] != q.get("gold_route"):
        return "S2", "route"
    if any(_name(t) == "open_note" and (t.get("input") or {}).get("path") in gold for t in run["tools"]):
        return "S8", "open_note"
    returned = [docs for docs in map(_docs, records) if gold & set(docs)]
    if returned:

        def stale_first(docs):
            first = min(docs.index(g) for g in gold if g in docs)
            return bool(stale & set(docs[:first]))

        if all(stale_first(docs) for docs in returned):
            return "S6", "gen_ai.retrieval.documents"
        if not gold & set(run["notes"]):
            return "S7", "gen_ai.retrieval.documents"
        return "S8", "notes" if stale & set(run["notes"]) else None
    if gold & set(run["notes"]):
        return "S8", "notes" if stale & set(run["notes"]) else None
    if not any(_name(t) in ("search", "find_projects") for t in run["tools"]):
        return "S3", "gen_ai.tool.name"
    best = None
    for r in records:
        if r.get("gen_ai.tool.name") not in ("search", "find_projects"):
            continue
        if r.get("error.type") == "QUERY":
            reach = 2
        else:
            cands = [c for c in r.get("ragnarok.candidates") or [] if c.get("gold") and c.get("doc_id")]
            cands = [c for c in cands if c.get("path") in gold]
            if not cands:
                continue
            reach = max(map(_reach, cands))
        best = reach if best is None else max(best, reach)
    if best is None:
        return "UNLABELED", None
    field = {2: "error.type", 3: "prefilter_dense_rank, prefilter_fts_rank", 4: "filters", 5: "final_rank"}[best]
    return f"S{best + 1}", field


def _entity_stage(entity: str, run: dict, records: list[dict]) -> tuple[str, str | None]:
    """The stage that lost one gold entity of a join question."""
    if any(_name(t) == "get_entity" and (t.get("input") or {}).get("id") == entity for t in run["tools"]):
        return "S8", "get_entity"
    entries = {name: [] for name in ("find_entity", "related")}
    for r in records:
        if r.get("gen_ai.tool.name") in entries:
            entries[r["gen_ai.tool.name"]] += r.get("ragnarok.entities") or []

    def ends(edges):
        return {e.get(k) for e in edges or [] for k in ("src", "dst")}

    found = any(entity in {c.get("id") for c in e.get("candidates") or []} for e in entries["find_entity"])
    if found or any(entity in ends(e.get("edges")) for e in entries["related"]):
        return "S7", "ragnarok.entities"
    if any(entity in {x.get("id") for x in e.get("removed") or []} for e in entries["related"]):
        return "S5", "ragnarok.entities.removed"
    if entries["related"]:
        return "S4", "ragnarok.entities.edges_before"
    return "S3", "ragnarok.entities"


def _join(q: dict, run: dict, records: list[dict], match) -> tuple[str, str | None]:
    if not _ragnarok(run) or run["route"] != q.get("gold_route"):
        return "S2", "route"
    missing = match(q.get("gold_entities", [])) - match(run["entities"])
    if not missing:
        return "S8", None  # every gold entity is in the answer, with a wrong one
    return min((_entity_stage(e, run, records) for e in sorted(missing)), key=lambda s: ORDER.index(s[0]))


def _negative(run: dict, match) -> tuple[str, str | None]:
    """The stage of the tool that gave the first cited note or entity, else S8."""
    cited = run["notes"][0] if run["notes"] else run["entities"][0]
    texts = {cited} if run["notes"] else {cited, *match([cited])}
    for tool in run["tools"]:
        given = str((tool.get("input") or {}).get("file_path") or "")
        if any(t in tool.get("result", "") for t in texts) or (_name(tool) == "Read" and given.endswith(cited)):
            stage = TOOL_STAGE.get(_name(tool))
            if stage:
                return stage, "tool_result"
    return "S8", "tool_result"


def label_run(q: dict, run: dict, records: list[dict], mirror: Path, match) -> tuple[str, str | None]:
    """The label of one failed run and its evidence field: RUN, MEASURE, the first stage that loses the gold,
    or UNLABELED."""
    if is_run_failure(run):
        return "RUN", "stream-json"
    absent = [p for p in gold_paths(q) if not (mirror / p).is_file()]
    if absent:
        return "MEASURE", "gold path not in the mirror"
    cls = q["class"]
    if cls == "single-truth":
        return _single_truth(q, run, records)
    if cls == "join":
        return _join(q, run, records, match)
    if cls == "live":
        return "S2", "route"
    return _negative(run, match)


def measure_candidates(q: dict, run: dict) -> list[str]:
    """The reasons for a person to check the gold set of a failed run (a MEASURE candidate)."""
    reasons = []
    listed = set(gold_paths(q)) | {d["path"] for d in q.get("distractors", [])}
    if q["class"] in ("single-truth", "join") and run["notes"] and run["notes"][0] not in listed:
        reasons.append(f"the first cited note {run['notes'][0]} is not a gold or distractor note")
    gold = top_gold(q)
    opened = any(_name(t) == "open_note" and (t.get("input") or {}).get("path") in gold for t in run["tools"])
    if q["class"] == "single-truth" and opened and not gold & set(run["notes"]):
        reasons.append("the agent opened the gold note and did not cite it: does it answer the question?")
    return reasons


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return math.nan, math.nan
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def attribute(
    sets_dir: Path, run_dir: Path, trace_folder: Path, mirror: Path, match, agent: str = "mcp-kb", confirmed=()
) -> dict:
    """Label each failed run of the agent in run_dir; a failed question gets the label of its first failed run.

    Condition 3 counts the failed questions with no RUN run: a question counts as labeled with a stage label (S7 and
    S8 only with an evidence field), with MEASURE for a gold path not in the mirror, or with a MEASURE candidate that
    the user confirmed (confirmed: question ids).
    """
    failed_questions, candidates = {}, []
    for path in sorted(sets_dir.glob("*.json")):
        if not (run_dir / path.stem).is_dir():
            continue
        for q in json.loads(path.read_text()):
            files = sorted(
                (run_dir / path.stem).glob(f"{q['id']}.{agent}.*.jsonl"), key=lambda p: int(p.stem.rsplit(".", 1)[1])
            )
            runs = [(int(f.stem.rsplit(".", 1)[1]), read_run(f, mirror)) for f in files]
            bad = [(n, r) for n, r in runs if failed(q, score_run(q, r, match))]
            if not bad:
                continue
            records = find_lines(trace_folder, {t for _, r in bad for t in r["trace_ids"]})
            labels = []
            for n, r in bad:
                label, evidence = label_run(q, r, [records[t] for t in r["trace_ids"] if t in records], mirror, match)
                if label != "RUN" and (reasons := measure_candidates(q, r)):
                    candidates.append({"id": q["id"], "run": n, "reasons": reasons})
                labels.append((n, label, evidence))
            n, label, evidence = labels[0]
            if q["id"] in confirmed and label != "RUN":
                label, evidence = "MEASURE", "confirmed by the user"
            failed_questions[q["id"]] = {
                "class": q["class"],
                "run": n,
                "label": label,
                "evidence": evidence,
                "counts": label not in ("RUN", "UNLABELED") and (label not in NEEDS_EVIDENCE or evidence is not None),
                "has_run": any(lb == "RUN" for _, lb, _ in labels),
            }
    scored = [r for r in failed_questions.values() if not r["has_run"]]
    counted = sum(r["counts"] for r in scored)
    lo, hi = wilson(counted, len(scored))
    distribution = dict(Counter(r["label"] for r in scored))
    lines = [
        f"{qid} {r['class']} {r['label']} run {r['run']} evidence {r['evidence']}"
        for qid, r in failed_questions.items()
    ]
    lines += [f"MEASURE candidate {c['id']} run {c['run']}: {'; '.join(c['reasons'])}" for c in candidates]
    share = counted / len(scored) if scored else math.nan
    lines += [
        f"failed questions {len(failed_questions)}, with a RUN run {len(failed_questions) - len(scored)}",
        f"labeled share {share:.2f} ({counted} of {len(scored)}), Wilson 95% [{lo:.2f}, {hi:.2f}]",
        "labels " + ", ".join(f"{k} {v}" for k, v in sorted(distribution.items())),
    ]
    return {
        "questions": failed_questions,
        "candidates": candidates,
        "n": len(scored),
        "counted": counted,
        "run": len(failed_questions) - len(scored),
        "distribution": distribution,
        "lines": lines,
    }
