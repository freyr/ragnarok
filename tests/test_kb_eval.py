"""The KB measures, the mechanical KB runner, the filtered no-regression runner and `eval attribute`, on synthetic
question sets, run folders and trace lines."""

import json
from dataclasses import replace

import pytest
from test_search import TopicEmbedder as FakeEmbedder

from ragnarok import agent_eval, cli, kb_eval
from ragnarok.config import Config, KbConfig, dump_config
from ragnarok.index import connect, update

GOLD, GOLD_LOW = "Kb/howto/deploy.md", "Kb/howto/deploy-notes.md"
STALE, DUP, OTHER = "Kb/howto/deploy-2024.md", "Kb/howto/deploy-copy.md", "Kb/misc/other.md"

SINGLE = {
    "id": "st-01",
    "class": "single-truth",
    "question": "How do I deploy?",
    "question_alt": "Deploy steps?",
    "gold": [{"path": GOLD, "grade": 2}, {"path": GOLD_LOW, "grade": 1}],
    "gold_eligibility": {"type": ["how-to"]},
    "gold_route": "kb",
    "gold_entities": [],
    "gold_claims": [],
    "distractors": [{"path": STALE, "kind": "stale"}, {"path": DUP, "kind": "duplicate"}],
    "origin": "core",
}
JOIN = {
    "id": "jn-01",
    "class": "join",
    "question": "Which clients asked for tiers?",
    "gold": [],
    "gold_route": "kb",
    "gold_entities": ["client-acme", "client-beta"],
    "distractors": [],
    "origin": "core",
}
LIVE = {"id": "lv-01", "class": "live", "question": "Cost of stage now?", "gold": [], "gold_route": "skill"}
NEGATIVE = {"id": "ng-01", "class": "negative", "question": "Who owns the moon?", "gold": [], "gold_route": "none"}


def answer(notes=(), entities=(), route="kb"):
    return {"notes": list(notes), "entities": list(entities), "route": route, "answer": "a"}


def run(notes=(), entities=(), route="kb", error=False, answered=True, tools=()):
    """A parsed agent run, as run_all gives it."""
    return {
        "tools": list(tools),
        "trace_ids": [],
        "turns": 3,
        "cost": 0.0,
        "error": error,
        "answered": answered,
        **answer(notes, entities, route),
    }


def stream(tools=(), final=None, error=False):
    """Raw stream-json lines: each tool call with its result text, then the result event with the final object."""
    events = [{"type": "system", "subtype": "init"}]
    for i, (name, given, result) in enumerate(tools):
        use = {"type": "tool_use", "id": f"t{i}", "name": name, "input": given}
        text = result if isinstance(result, str) else json.dumps(result)
        back = {"type": "tool_result", "tool_use_id": f"t{i}", "content": [{"type": "text", "text": text}]}
        events += [
            {"type": "assistant", "message": {"content": [use]}},
            {"type": "user", "message": {"content": [back]}},
        ]
    text = "Done." + (f"\n{json.dumps(final)}" if final is not None else "")
    events.append({"type": "result", "subtype": "success", "is_error": error, "num_turns": 3, "result": text})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def search_call(trace_id, query="deploy"):
    return ("mcp__ragnarok__search", {"query": query}, {"results": [], "trace_id": trace_id})


def line(trace_id, tool="search", candidates=(), returned=(), entities=(), error=None):
    return {
        "trace_id": trace_id,
        "gen_ai.tool.name": tool,
        "ragnarok.candidates": list(candidates),
        "gen_ai.retrieval.documents": [{"id": f"id-{p}", "path": p, "score": None} for p in returned],
        "ragnarok.entities": list(entities),
        "error.type": error,
    }


def gold(path=GOLD, dense=None, fts=None, filters=None, final=None):
    return {
        "doc_id": f"id-{path}",
        "path": path,
        "gold": True,
        "prefilter_dense_rank": dense,
        "prefilter_fts_rank": fts,
        "filters": filters or {},
        "final_rank": final,
    }


def ids(names):
    return set(names)


class Lab:
    """A mirror, a folder of question sets, a run folder and a trace folder in tmp_path."""

    def __init__(self, tmp_path):
        self.mirror = tmp_path / "mirror"
        self.sets = tmp_path / "sets-kb"
        self.runs = tmp_path / "runs"
        self.traces = tmp_path / "trace"
        for p in (GOLD, GOLD_LOW, STALE, DUP, OTHER):
            (self.mirror / p).parent.mkdir(parents=True, exist_ok=True)
            (self.mirror / p).write_text("# n\n")
        self.sets.mkdir(parents=True)
        self.traces.mkdir(parents=True)

    def questions(self, name, *questions):
        (self.sets / f"{name}.json").write_text(json.dumps(list(questions)))

    def run(self, set_name, qid, n, tools=(), final=None, error=False, raw=None, agent="mcp-kb"):
        path = self.runs / set_name / f"{qid}.{agent}.{n}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stream(tools, final, error) if raw is None else raw)

    def trace(self, *records):
        with (self.traces / "2026-11-05.jsonl").open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def attribute(self, confirmed=(), match=ids):
        return kb_eval.attribute(self.sets, self.runs, self.traces, self.mirror, match, confirmed=confirmed)


@pytest.fixture
def lab(tmp_path):
    return Lab(tmp_path)


# --- measures -----------------------------------------------------------------------------------------------------


def test_canonical_first_and_stale_citation_on_a_synthetic_set(vault, tmp_path):
    # mechanical: the first returned document, and a stale distractor in the top 5
    assert kb_eval.score_ranked(SINGLE, [GOLD, STALE]) == {"canonical_first": 1.0, "stale": 1.0}
    assert kb_eval.score_ranked(SINGLE, [GOLD_LOW, GOLD]) == {"canonical_first": 0.0, "stale": 0.0}
    late = [OTHER, DUP, "a.md", "b.md", "c.md", STALE]  # the stale distractor is sixth
    assert kb_eval.score_ranked(SINGLE, late) == {"canonical_first": 0.0, "stale": 0.0}
    # agent: the first cited note, and a stale distractor in the cited notes
    score = kb_eval.score_run(SINGLE, run([GOLD, STALE]), ids)
    assert (score["canonical_first"], score["stale"]) == (1.0, 1.0)
    score = kb_eval.score_run(SINGLE, run([DUP, GOLD]), ids)
    assert (score["canonical_first"], score["stale"]) == (0.0, 0.0)

    # the mechanical runner: the baseline through its tool calls, the new code in the process, two query forms
    vault.dimension = 8
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    note = "Home/Heating/notes/boiler.md"
    q = {**SINGLE, "gold": [{"path": note, "grade": 2}], "distractors": [{"path": "Kb/absent.md", "kind": "stale"}]}
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "single-truth.json").write_text(json.dumps([q]))
    calls = []

    def base(batch):
        calls.extend(batch)
        out = []
        for tool, args in batch:
            if tool == "search":  # the first form finds the gold note first, the second form finds nothing
                out.append({"results": [{"path": note}] if args["query"] == q["question"] else []})
            else:
                out.append({"error": "not a note"} if args["path"] == "Kb/absent.md" else {"type": "note"})
        return out

    lines = kb_eval.kb_mechanical(vault, conn, FakeEmbedder(), sets, base)
    assert [args["query"] for tool, args in calls if tool == "search"] == [q["question"], q["question_alt"]]
    assert "  canonical-first base 0.50" in "\n".join(lines)
    assert "MEASURE st-01 Kb/absent.md absent from base" in lines
    assert "MEASURE st-01 Kb/absent.md absent from kb" in lines


def test_set_f1_matches_names_through_aliases(tmp_path):
    def entity(id_, name, kind, extra=""):
        return f"---\nid: {id_}\ntype: entity\nkind: {kind}\n{extra}---\n# {name}\nThe {name}.\n"

    cfg = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["Kb"], dimension=8, kb=KbConfig())
    notes = {
        "Kb/entities/Acme.md": entity("client-acme", "Acme", "client", "aliases: [Acme Corp]\n"),
        "Kb/entities/Beta.md": entity("client-beta", "Beta", "client"),
        "Kb/entities/Gamma.md": entity("client-gamma", "Gamma", "client"),
    }
    for rel, text in notes.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    conn = connect(cfg)
    update(cfg, conn, FakeEmbedder())
    match = kb_eval.Matcher(cfg, conn)
    assert match(["ACME  corp", "client-beta", "Gamma", "Nobody"]) == {
        "client-acme",
        "client-beta",
        "client-gamma",
        "nobody",
    }
    # gold: Acme and Beta; the answer names Acme by an alias, Beta by its name, and one wrong client
    score = kb_eval.score_run(JOIN, run(entities=["acme corp", "Beta", "Gamma"]), match)
    assert score["f1"] == pytest.approx(0.8)
    assert kb_eval.score_run(JOIN, run(entities=["Acme", "beta"]), match)["f1"] == 1.0


def test_question_value_is_the_mean_of_its_runs(vault):
    q = {**SINGLE, "gold": [{"path": "Home/Heating/notes/boiler.md", "grade": 2}], "distractors": []}
    good, bad = run(["Home/Heating/notes/boiler.md"]), run(["Home/Bike/shortlist.md"])
    assert kb_eval.question_value([kb_eval.score_run(q, r, ids) for r in (good, bad)])["canonical_first"] == 0.5
    sets = {"single-truth": [q]}
    results = {"single-truth": {"st-01": {"mcp-kb": [good, bad], "mcp-base": [good, good]}}}
    lines = agent_eval.report(vault, sets, results, [], ["mcp-kb", "mcp-base"], match=ids)
    row = next(x for x in lines if x.startswith("| Canonical-first"))
    # one row for the question: the mean of its two runs, and the paired difference over the questions
    assert row.startswith("| Canonical-first, single-truth (n=1)") and "mcp-base 1.00, mcp-kb 0.50" in row
    assert "-0.50 [-0.50, -0.50]" in row


def test_failed_run_gets_the_worst_value_of_each_measure(vault):
    worst = {"canonical_first": 0.0, "stale": 1.0, "f1": 0.0, "route": 0.0, "no_answer": 0.0, "run": True}
    crashed = run([GOLD], route="kb", error=True)
    no_object = run(answered=False, route=None)
    for r in (crashed, no_object):
        for q in (SINGLE, JOIN, LIVE, NEGATIVE):
            assert kb_eval.score_run(q, r, ids) == worst
    sets = {"single-truth": [SINGLE]}
    results = {"single-truth": {"st-01": {"mcp-kb": [crashed, run([GOLD])], "mcp-base": [run([GOLD])] * 2}}}
    lines = agent_eval.report(vault, sets, results, [], ["mcp-kb", "mcp-base"], match=ids)
    assert "failed runs (RUN): mcp-kb 1, mcp-base 0" in lines


def test_failure_definition_for_each_class():
    def failed(q, r, match=ids):
        return kb_eval.failed(q, kb_eval.score_run(q, r, match))

    assert not failed(SINGLE, run([GOLD, DUP]))
    assert failed(SINGLE, run([GOLD_LOW, GOLD]))  # canonical-first 0
    assert failed(SINGLE, run([GOLD, STALE]))  # a stale citation
    assert not failed(JOIN, run(entities=["client-beta", "client-acme"]))
    assert failed(JOIN, run(entities=["client-acme"]))  # set F1 below 1.0
    assert not failed(LIVE, run(route="skill"))
    assert failed(LIVE, run([GOLD], route="kb"))
    assert not failed(NEGATIVE, run(route="none"))
    assert failed(NEGATIVE, run([OTHER], route="none"))  # an answer with a note
    assert failed(NEGATIVE, run(entities=["client-acme"], route="none"))  # an answer with an entity


# --- filtered no-regression runner --------------------------------------------------------------------------------


@pytest.fixture
def indexed(vault):
    vault.dimension = 8
    conn = connect(vault)
    update(vault, conn, FakeEmbedder())
    return vault, conn


def filtered_run(indexed, questions, base_type=None, new_hits=None):
    cfg, conn = indexed
    calls, searched = [], []

    def base(batch):
        calls.extend(batch)
        out = []
        for tool, args in batch:
            if tool == "open_note":
                row = conn.execute("SELECT vault, project, area, type FROM doc WHERE path = ?", (args["path"],))
                meta = dict(zip(("vault", "project", "area", "type"), row.fetchone(), strict=True))
                out.append({**meta, "type": base_type or meta["type"]})
            else:  # the baseline finds every expected note
                q = next(q for q in questions if q["question"] == args["query"])
                out.append({"results": [{"path": p} for p in q["expected"]]})
        return out

    def new_search(cfg_, conn_, embedder, query, **kwargs):
        searched.append(kwargs)
        return [type("Hit", (), {"path": p}) for p in (new_hits or {}).get(query, [])]

    lines, regressed = kb_eval.filtered(cfg, conn, FakeEmbedder(), questions, base, new_search)
    return lines, regressed, calls, searched


def test_filtered_runner_lists_a_regressed_question(indexed):
    questions = [
        {"id": "d1", "type": "fact", "question": "boiler?", "expected": ["Home/Heating/notes/boiler.md"]},
        {"id": "d2", "type": "synthesis", "question": "cpu?", "expected": ["Work/Devops/db-resize/research-cpu.md"]},
        {"id": "d3", "type": "fact", "question": "none?", "expected": []},  # no expected note: not a question here
    ]
    new_hits = {"boiler?": ["Home/Heating/notes/boiler.md"], "cpu?": ["Home/Bike/shortlist.md"]}
    lines, regressed, calls, searched = filtered_run(indexed, questions, new_hits=new_hits)
    assert regressed == ["d2"]
    assert [args["limit"] for tool, args in calls if tool == "search"] == [5, 10]  # K of score_draft
    assert all(kw["authority"] is False for kw in searched)
    cfg, conn = indexed
    vault, project, area = conn.execute(
        "SELECT vault, project, area FROM doc WHERE path = 'Work/Devops/db-resize/research-cpu.md'"
    ).fetchone()
    base_cpu = next(args for tool, args in calls if tool == "search" and args["query"] == "cpu?")
    assert (base_cpu["vault"], base_cpu["project"], base_cpu["area"]) == (vault, project, area)
    assert {k: searched[1][k] for k in ("vault", "project", "area")} == {
        "vault": vault,
        "project": project,
        "area": area,
    }
    assert "d1 K=5 base 1.00 new 1.00" in "\n".join(lines)
    assert "d2 K=10 base 1.00 new 0.00 LOWER" in "\n".join(lines)
    assert "regressed: d2" in lines


def test_filtered_runner_skips_a_type_that_the_systems_map_differently(indexed):
    questions = [{"id": "d1", "type": "fact", "question": "boiler?", "expected": ["Home/Heating/notes/boiler.md"]}]
    hits = {"boiler?": ["Home/Heating/notes/boiler.md"]}
    _, _, calls, searched = filtered_run(indexed, questions, new_hits=hits)
    cfg, conn = indexed
    (doc_type,) = conn.execute("SELECT type FROM doc WHERE path = 'Home/Heating/notes/boiler.md'").fetchone()
    assert next(a for t, a in calls if t == "search")["type"] == doc_type and searched[0]["type"] == doc_type
    _, regressed, calls, searched = filtered_run(indexed, questions, base_type="other-type", new_hits=hits)
    assert "type" not in next(a for t, a in calls if t == "search") and "type" not in searched[0]
    assert regressed == []


# --- attribution --------------------------------------------------------------------------------------------------


def test_attribute_labels_a_filter_loss_as_s5(lab):
    lab.questions("single-truth", SINGLE)
    lab.run("single-truth", "st-01", 1, [search_call("t1")], answer([OTHER]))
    lab.run("single-truth", "st-01", 2, [search_call("t2")], answer([GOLD]))
    lab.trace(
        line("t1", candidates=[gold(dense=3, fts=None, filters={"type": False, "private": True})], returned=[OTHER])
    )
    result = lab.attribute()
    assert result["questions"]["st-01"]["label"] == "S5"
    assert result["questions"]["st-01"]["run"] == 1


def test_attribute_labels_an_absent_gold_path_as_measure(lab):
    absent = {**SINGLE, "gold": [{"path": "Kb/howto/gone.md", "grade": 2}]}
    lab.questions("single-truth", absent)
    lab.run("single-truth", "st-01", 1, [search_call("t1")], answer([OTHER]))
    lab.trace(line("t1", candidates=[gold("Kb/howto/gone.md", dense=3, filters={"type": False})]))
    result = lab.attribute()
    assert result["questions"]["st-01"]["label"] == "MEASURE"
    assert result["counted"] == 1


def test_attribute_lists_other_measure_candidates_for_the_user(lab):
    lab.questions("single-truth", SINGLE)
    # the agent cites first a note outside the gold and distractor sets: a person must check it
    lab.run("single-truth", "st-01", 1, [search_call("t1")], answer([OTHER]))
    lab.trace(line("t1", candidates=[gold(dense=3, filters={"type": False})], returned=[OTHER]))
    result = lab.attribute()
    assert [c["id"] for c in result["candidates"]] == ["st-01"]
    assert result["questions"]["st-01"]["label"] == "S5"  # not confirmed: the stage label stays
    assert any(x.startswith("MEASURE candidate st-01") for x in result["lines"])
    confirmed = lab.attribute(confirmed=["st-01"])
    assert confirmed["questions"]["st-01"]["label"] == "MEASURE"
    assert confirmed["distribution"] == {"MEASURE": 1}


def test_attribute_labels_a_join_entity_lost_in_find_entity_as_s3(lab):
    lab.questions("join", JOIN)
    find = ("mcp__ragnarok__find_entity", {"name": "tiers"}, {"results": [], "trace_id": "t1"})
    lab.run("join", "jn-01", 1, [find], answer(entities=["client-beta"]))
    candidates = [{"id": "client-beta", "matched_by": "name", "score": 1.0}]
    lab.trace(line("t1", "find_entity", entities=[{"name": "tiers", "candidates": candidates, "selected": None}]))
    assert lab.attribute()["questions"]["jn-01"]["label"] == "S3"


def test_attribute_labels_a_join_entity_removed_by_a_filter_as_s5(lab):
    lab.questions("join", JOIN)
    rel = ("mcp__ragnarok__related", {"entity": "feat-tiers", "kind": "client"}, {"edges": [], "trace_id": "t1"})
    lab.run("join", "jn-01", 1, [rel], answer(entities=["client-beta"]))
    entry = {
        "name": "feat-tiers",
        "candidates": [{"id": "feat-tiers", "matched_by": "id", "score": None}],
        "selected": "feat-tiers",
        "edges_before": [
            {"src": "client-acme", "rel": "requests", "dst": "feat-tiers", "depth": 1},
            {"src": "client-beta", "rel": "requests", "dst": "feat-tiers", "depth": 1},
        ],
        "edges": [{"src": "client-beta", "rel": "requests", "dst": "feat-tiers", "depth": 1}],
        "removed": [{"id": "client-acme", "filter": "as_of"}],
    }
    lab.trace(line("t1", "related", entities=[entry]))
    assert lab.attribute()["questions"]["jn-01"]["label"] == "S5"
    # an entity that is not in the edges before the filters is lost at S4
    entry = {**entry, "edges_before": entry["edges"], "removed": []}
    (lab.traces / "2026-11-05.jsonl").unlink()
    lab.trace(line("t1", "related", entities=[entry]))
    assert lab.attribute()["questions"]["jn-01"]["label"] == "S4"


def test_attribute_needs_an_evidence_field_for_s7_and_s8(lab):
    q7, q8, q8e = ({**SINGLE, "id": i} for i in ("st-07", "st-08", "st-09"))
    lab.questions("single-truth", q7, q8, q8e)
    # S7: the trace shows the gold in the returned documents, and the agent did not cite it
    lab.run("single-truth", "st-07", 1, [search_call("t7")], answer([DUP]))
    lab.trace(line("t7", candidates=[gold(dense=1, final=2)], returned=[DUP, GOLD]))
    # S8 with no evidence: the agent cited the gold second, and never opened it
    lab.run("single-truth", "st-08", 1, [search_call("t8")], answer([DUP, GOLD]))
    lab.trace(line("t8", candidates=[gold(dense=1, final=2)], returned=[DUP, GOLD]))
    # S8 with evidence: open_note on the gold, and the answer is still wrong
    opened = ("mcp__ragnarok__open_note", {"path": GOLD}, {"path": GOLD, "trace_id": "t9b"})
    lab.run("single-truth", "st-09", 1, [search_call("t9"), opened], answer([DUP]))
    lab.trace(line("t9", candidates=[gold(dense=1, final=2)], returned=[DUP, GOLD]))
    result = lab.attribute()
    got = {i: (r["label"], r["evidence"], r["counts"]) for i, r in result["questions"].items()}
    assert got == {
        "st-07": ("S7", "gen_ai.retrieval.documents", True),
        "st-08": ("S8", None, False),
        "st-09": ("S8", "open_note", True),
    }
    assert (result["counted"], result["n"]) == (2, 3)


def test_question_with_no_first_stage_is_unlabeled(lab):
    lab.questions("single-truth", SINGLE)
    # a search call whose trace line is not in the trace folder: no stage can be checked
    lab.run("single-truth", "st-01", 1, [search_call("t-missing")], answer([OTHER]))
    result = lab.attribute()
    assert result["questions"]["st-01"]["label"] == "UNLABELED"
    assert (result["counted"], result["n"]) == (0, 1)


def test_question_with_one_failed_run_is_a_failed_question_with_the_label_of_its_first_failed_run(lab):
    q2 = {**SINGLE, "id": "st-02"}
    lab.questions("single-truth", SINGLE, q2)
    # st-01: run 1 passes, run 2 fails at S5
    lab.run("single-truth", "st-01", 1, [search_call("a1")], answer([GOLD]))
    lab.run("single-truth", "st-01", 2, [search_call("a2")], answer([OTHER]))
    lab.trace(line("a2", candidates=[gold(dense=3, filters={"type": False})]))
    # st-02: run 1 fails at S4, run 2 fails at S5: the question gets S4
    lab.run("single-truth", "st-02", 1, [search_call("b1")], answer([OTHER]))
    lab.run("single-truth", "st-02", 2, [search_call("b2")], answer([OTHER]))
    lab.trace(
        line("b1", candidates=[gold(dense=None, fts=None)]),
        line("b2", candidates=[gold(dense=2, filters={"type": False})]),
    )
    result = lab.attribute()
    assert (result["questions"]["st-01"]["label"], result["questions"]["st-01"]["run"]) == ("S5", 2)
    assert (result["questions"]["st-02"]["label"], result["questions"]["st-02"]["run"]) == ("S4", 1)
    assert result["n"] == 2


def test_attribute_labels_a_negative_question_by_the_tool_of_the_cited_note(lab):
    qs = [{**NEGATIVE, "id": f"ng-{i}"} for i in range(1, 6)]
    lab.questions("negative", *qs)
    found = ("mcp__ragnarok__search", {"query": "moon"}, {"results": [{"path": OTHER}], "trace_id": "n1"})
    lab.run("negative", "ng-1", 1, [found], answer([OTHER], route="none"))
    lab.trace(line("n1", returned=[OTHER]))
    opened = ("mcp__ragnarok__open_note", {"path": OTHER}, {"path": OTHER, "text": "x"})
    lab.run("negative", "ng-2", 1, [opened], answer([OTHER], route="none"))
    entity = ("mcp__ragnarok__find_entity", {"name": "moon"}, {"results": [{"id": "client-acme"}]})
    lab.run("negative", "ng-3", 1, [entity], answer(entities=["client-acme"], route="none"))
    rel = ("mcp__ragnarok__related", {"entity": "x"}, {"edges": [{"target": {"id": "client-beta"}}]})
    lab.run("negative", "ng-4", 1, [rel], answer(entities=["client-beta"], route="none"))
    lab.run("negative", "ng-5", 1, [found], answer(["Kb/made-up.md"], route="none"))  # no tool gave it
    result = lab.attribute()
    got = {i: (r["label"], r["counts"]) for i, r in result["questions"].items()}
    assert got == {
        "ng-1": ("S6", True),
        "ng-2": ("S7", True),
        "ng-3": ("S3", True),
        "ng-4": ("S4", True),
        "ng-5": ("S8", True),
    }
    assert result["questions"]["ng-5"]["evidence"] == "tool_result"


def test_crashed_run_gets_the_run_label_and_is_not_counted_in_condition_3(lab, vault, tmp_path, monkeypatch):
    q2 = {**SINGLE, "id": "st-02"}
    lab.questions("single-truth", SINGLE, q2)
    lab.run("single-truth", "st-01", 1, raw="")  # a crash: no result event
    lab.run("single-truth", "st-01", 2, [search_call("c2")], answer([OTHER]))
    lab.trace(line("c2", candidates=[gold(dense=3, filters={"type": False})]))
    lab.run("single-truth", "st-02", 1, [search_call("d1")], answer([OTHER]))
    lab.trace(line("d1", candidates=[gold(dense=3, filters={"type": False})]))
    result = lab.attribute()
    assert result["questions"]["st-01"]["label"] == "RUN"
    assert (result["counted"], result["n"], result["run"]) == (1, 1, 1)
    assert any("share 1.00 (1 of 1)" in x for x in result["lines"])

    # the runner deletes a RUN result and runs it again one time, and only one time
    monkeypatch.setattr(agent_eval, "prepare", lambda cfg, **kw: cfg)
    outputs = [("", "crash\n", 1), (stream([], answer([GOLD])), "", 0)]
    seen = []

    def runner(cmd, cwd, env):
        seen.append(cmd)
        return outputs[min(len(seen) - 1, 1)]

    out = tmp_path / "out"
    results = agent_eval.run_all(vault, {"s": [SINGLE]}, out, 8, "sonnet", ["mcp-kb"], 1, workers=1, runner=runner)
    assert len(seen) == 2 and results["s"]["st-01"]["mcp-kb"][0]["notes"] == [GOLD]
    assert not (out / "s" / "st-01.mcp-kb.1.stderr").exists()
    outputs[1] = ("", "crash again\n", 1)
    (out / "s" / "st-01.mcp-kb.1.jsonl").write_text("")  # a RUN file that was already run again
    agent_eval.run_all(vault, {"s": [SINGLE]}, out, 8, "sonnet", ["mcp-kb"], 1, workers=1, runner=runner)
    assert len(seen) == 2

    # a run that times out is a RUN run with its partial output, not an error that stops the other runs
    def timeout(*args, **kwargs):
        raise agent_eval.subprocess.TimeoutExpired(args[0], 900, output=b'{"type": "system"}\n')

    monkeypatch.setattr(agent_eval.subprocess, "run", timeout)
    raw, stderr, code = agent_eval.run_claude(["claude"], tmp_path, {})
    assert (raw, code) == ('{"type": "system"}\n', 124) and "timeout" in stderr
    path = tmp_path / "timeout.jsonl"
    path.write_text(raw)
    assert agent_eval.is_run_failure(agent_eval.read_run(path, None))


def test_attribute_reads_the_trace_folder_of_the_evaluation_home(tmp_path, capsys):
    cfg = Config(root=tmp_path / "vaults", data_dir=tmp_path / "data", vaults=["Kb"])
    config = tmp_path / "config.toml"
    config.write_text(dump_config(cfg, root=str(tmp_path / "vaults"), data_dir=str(tmp_path / "data")))
    home = agent_eval.eval_home(cfg)
    home.mkdir(parents=True)
    (home / "config.toml").write_text(dump_config(replace(cfg, vault_paths={}), root="vaults", data_dir="data"))
    lab = Lab(tmp_path / "lab")
    lab.mirror.rename(home / "vaults")
    lab.sets.rename(home / "sets-kb")
    lab.sets, lab.traces, lab.mirror = home / "sets-kb", home / "data" / "trace", home / "vaults"
    lab.traces.mkdir(parents=True)
    lab.questions("single-truth", SINGLE)
    lab.run("single-truth", "st-01", 1, [search_call("t1")], answer([OTHER]))
    lab.trace(line("t1", candidates=[gold(dense=3, filters={"type": False})]))
    assert cli.main(["--config", str(config), "evaluate", "attribute", str(lab.runs)]) == 0
    out = capsys.readouterr().out
    assert "st-01 single-truth S5" in out
    # --trace-dir reads another folder: here an empty one, so no stage can be checked
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--config", str(config), "evaluate", "attribute", str(lab.runs), "--trace-dir", str(empty)]) == 0
    assert "st-01 single-truth UNLABELED" in capsys.readouterr().out
