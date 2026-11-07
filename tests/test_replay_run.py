import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
from test_search import TopicEmbedder

from ragnarok import replay_run as rr
from ragnarok.cli import main
from ragnarok.config import Config, dump_config, load_config
from ragnarok.index import connect, update
from ragnarok.metadata import parse_note

# --- the scorer --------------------------------------------------------------------------------------------------

POSITIVE = {
    "id": "p001",
    "kind": "positive",
    "positive_kind": "content_change",
    "old": [{"path": "V/old-1.md", "expected": "supersede"}, {"path": "V/old-2.md", "expected": "supersede"}],
}
OLDS = ["V/old-1.md", "V/old-2.md"]
MERGE = {**POSITIVE, "id": "p002", "old": [{"path": "V/state.md", "expected": "merge"}]}
DUPLICATE = {
    "id": "d001",
    "kind": "duplicate",
    "positive_kind": None,
    "old": [{"path": "V/dup.md", "expected": "same"}],
}
NEGATIVE = {"id": "n001", "kind": "negative", "positive_kind": None, "old": []}
EMPTY = {"created": [], "changed": [], "deleted": []}


def rec(action, targets=(), written=False, outcome=None, **extra):
    outcome = outcome or ("written" if written else "nothing-written")
    return {
        "record": "kb-write/1",
        "action": action,
        "targets": list(targets),
        "written": written,
        "outcome": outcome,
        "run_id": None,
        **extra,
    }


def diff(created=(), changed=(), deleted=()):
    return {"created": list(created), "changed": list(changed), "deleted": list(deleted)}


def texts(before=None, after=None, as_of=None):
    before, after = before or {}, after or {}
    return rr.NoteTexts(before.get, after.get, as_of)


def note(body="# Note\nThe pump follows the curve.\n", **keys):
    head = "".join(f"{k}: {json.dumps(v)}\n" for k, v in keys.items())
    return f"---\n{head}---\n{body}"


def superseded(olds=OLDS, new="V/new.md"):
    """The note texts of a correct supersede of the olds by the new note."""
    link = new.removesuffix(".md").rsplit("/", 1)[-1]
    before = {o: note(**{"class": "state", "validity": "current"}) for o in olds}
    after = {o: note(**{"class": "state", "validity": "superseded", "superseded_by": f"[[{link}]]"}) for o in olds}
    stems = [o.removesuffix(".md") for o in olds]
    after[new] = note(**{"class": "state", "validity": "current", "supersedes": [f"[[{s}]]" for s in stems]})
    return texts(before, after)


def merged(target="V/state.md", others=(), as_of="2026-08-16"):
    """The note texts of a correct merge into target, with the others superseded by it."""
    link = target.removesuffix(".md")
    before = {t: note(**{"class": "state", "validity": "current", "as_of": "2026-01-01"}) for t in (target, *others)}
    after = {o: note(**{"class": "state", "validity": "superseded", "superseded_by": f"[[{link}]]"}) for o in others}
    keys = {"class": "state", "validity": "current", "as_of": as_of}
    if others:
        keys["supersedes"] = [f"[[{o.removesuffix('.md')}]]" for o in others]
    after[target] = note("# Note\nThe pump follows the curve.\nThe boiler runs in the morning.\n", **keys)
    return texts(before, after, "2026-08-16")


def test_expected_action_of_each_kind():
    assert rr.expected_of(POSITIVE) == ("supersede", OLDS)
    assert rr.expected_of(MERGE) == ("merge", ["V/state.md"])
    assert rr.expected_of(DUPLICATE) == ("same", ["V/dup.md"])
    assert rr.expected_of(NEGATIVE) == ("new", [])


def test_a_supersede_with_the_old_notes_and_exactly_their_change_is_correct():
    s = rr.score(POSITIVE, rec("supersede", OLDS[::-1], written=True), diff(["V/new.md"], OLDS), superseded())
    assert s["action_ok"] and s["target_ok"] and s["written_ok"] and s["correct"]
    assert s["category"] == "correct" and s["reasons"] == []


def test_an_extra_target_is_wrong_and_reported():
    record = rec("supersede", [*OLDS, "V/other.md"], written=True)
    s = rr.score(POSITIVE, record, diff(["V/new.md"], [*OLDS, "V/other.md"]), superseded())
    assert not s["target_ok"] and not s["correct"] and s["category"] == "wrong"
    assert s["extra_targets"] == ["V/other.md"] and "extra-targets" in s["reasons"]


def test_a_supersede_that_misses_an_old_note_is_wrong():
    s = rr.score(POSITIVE, rec("supersede", ["V/old-1.md"], written=True), diff(["V/new.md"], ["V/old-1.md"]))
    assert s["action_ok"] and not s["target_ok"] and not s["correct"]
    assert s["category"] == "wrong" and "missing-targets" in s["reasons"]


def test_extra_writes_are_wrong():
    # an unrelated note changed
    s = rr.score(
        POSITIVE, rec("supersede", OLDS, written=True), diff(["V/new.md"], [*OLDS, "V/unrelated.md"]), superseded()
    )
    assert s["target_ok"] and not s["written_ok"] and s["category"] == "wrong"
    assert s["reasons"] == ["extra-writes"] and s["extra_writes"] == ["changed V/unrelated.md"]
    # two created notes
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), diff(["V/new.md", "V/junk.md"], OLDS))
    assert s["extra_writes"] == ["created V/junk.md"]
    # a merge that also creates a note
    s = rr.score(MERGE, rec("merge", ["V/state.md"], written=True), diff(["V/new.md"], ["V/state.md"]))
    assert not s["correct"] and s["extra_writes"] == ["created V/new.md"]
    # the only created file is not a note
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), diff(["V/.obsidian/workspace.json"], OLDS))
    assert not s["correct"] and s["missing_writes"] == ["no created note"]
    assert s["extra_writes"] == ["created V/.obsidian/workspace.json"]


def test_a_merge_counts_when_the_expected_action_is_merge():
    s = rr.score(MERGE, rec("merge", ["V/state.md"], written=True), diff(changed=["V/state.md"]), merged())
    assert s["correct"] and s["category"] == "correct" and s["effect_errors"] == []


def test_a_merge_for_a_supersede_is_its_own_category():
    s = rr.score(POSITIVE, rec("merge", OLDS, written=True), diff(changed=OLDS), merged(OLDS[0], OLDS[1:]))
    assert not s["action_ok"] and not s["correct"] and s["merge_for_supersede"]
    assert s["category"] == "merge-for-supersede"


def test_same_on_a_duplicate_with_nothing_written_is_correct():
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), EMPTY)
    assert s["correct"] and s["category"] == "correct"
    assert not rr.score(DUPLICATE, rec("same", ["V/dup.md", "V/x.md"]), EMPTY)["correct"]


def test_new_on_a_negative_needs_no_targets_and_one_created_note():
    assert rr.score(NEGATIVE, rec("new", [], written=True), diff(["V/fresh.md"]))["correct"]
    assert not rr.score(NEGATIVE, rec("new", ["V/x.md"], written=True), diff(["V/fresh.md"]))["target_ok"]
    assert not rr.score(NEGATIVE, rec("new", [], written=True), diff(["fresh.md"]))["correct"]  # not in a vault


def test_ask_is_neither_correct_nor_wrong():
    s = rr.score(POSITIVE, rec("ask", ["V/old-1.md"]), EMPTY)
    assert not s["correct"] and s["category"] == "ask"


def test_a_missing_record_is_failed():
    s = rr.score(POSITIVE, None, EMPTY)
    assert s["action"] is None and not s["correct"] and s["category"] == "failed"


def test_a_refused_run_is_refused():
    s = rr.score(NEGATIVE, rec("refused", outcome="refused"), EMPTY)
    assert s["category"] == "refused" and not s["correct"]


def test_the_written_flag_must_match_the_snapshot():
    # the record says written, the snapshot shows no change
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), EMPTY)
    assert s["action_ok"] and s["target_ok"] and not s["written_ok"] and not s["correct"]
    assert "missing-writes" in s["reasons"]
    # same, but a file changed
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), diff(changed=["V/dup.md"]))
    assert not s["written_ok"] and s["category"] == "wrong"
    # supersede written, but an old note is unchanged on disk
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), diff(["V/n.md"], ["V/old-1.md"]))
    assert s["missing_writes"] == ["unchanged V/old-2.md"]
    # the flag is false for a write action
    s = rr.score(NEGATIVE, rec("new", [], written=False), diff(["V/fresh.md"]))
    assert s["reasons"] == ["written-flag"]
    # a deleted file is never consistent
    s = rr.score(NEGATIVE, rec("new", [], written=True), diff(["V/fresh.md"], deleted=["V/gone.md"]))
    assert not s["written_ok"]


def test_a_supersede_in_the_reverse_direction_is_wrong():
    # the probe of review round 2: the new note is marked superseded by the old note
    old = "V/old-1.md"
    case = {**POSITIVE, "old": [{"path": old, "expected": "supersede"}]}
    notes = texts(
        {old: note(**{"class": "state", "validity": "current"})},
        {
            old: note(**{"class": "state", "validity": "current", "supersedes": ["[[new]]"]}),
            "V/new.md": note(**{"class": "state", "validity": "superseded", "superseded_by": "[[old-1]]"}),
        },
    )
    s = rr.score(case, rec("supersede", [old], written=True), diff(["V/new.md"], [old]), notes)
    assert not s["correct"] and s["category"] == "wrong"
    assert {"supersede-direction", "missing-supersedes", "missing-superseded-by", "validity"} <= set(s["reasons"])
    assert s["effect_errors"]


def test_a_no_op_edit_of_an_old_note_is_wrong():
    old = "V/old-1.md"
    case = {**POSITIVE, "old": [{"path": old, "expected": "supersede"}]}
    good = superseded([old])
    start = good.before(old)
    notes = texts(
        {old: start}, {old: start.replace("curve.", "curve.   ") + "\n\n", "V/new.md": good.after("V/new.md")}
    )
    s = rr.score(case, rec("supersede", [old], written=True), diff(["V/new.md"], [old]), notes)
    assert not s["correct"] and "no-op-edit" in s["reasons"] and "missing-superseded-by" in s["reasons"]


def test_a_write_with_no_note_texts_is_not_correct():
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), diff(["V/new.md"], OLDS))
    assert not s["correct"] and s["reasons"] == ["effect-unchecked"]
    # the probe of review round 2 now gives wrong
    rr.score(POSITIVE, rec("ask"), EMPTY)  # an action with no write needs no texts
    assert rr.score(DUPLICATE, rec("same", ["V/dup.md"]), EMPTY)["correct"]


def test_a_superseded_state_needs_validity_superseded_and_a_record_does_not():
    old = "V/old-1.md"
    case = {**POSITIVE, "old": [{"path": old, "expected": "supersede"}]}
    for cls, ok in (("state", False), ("record", True)):
        notes = texts(
            {old: note(**{"class": cls})},
            {old: note(**{"class": cls, "superseded_by": "[[V/new]]"}), "V/new.md": note(supersedes=["[[old-1]]"])},
        )
        s = rr.score(case, rec("supersede", [old], written=True), diff(["V/new.md"], [old]), notes)
        assert s["correct"] is ok and (ok or s["reasons"] == ["validity"])


def test_a_merge_checks_its_target_the_superseded_notes_and_as_of():
    case = {**POSITIVE, "old": [{"path": "V/state.md", "expected": "merge"}, {"path": "V/b.md", "expected": "merge"}]}
    targets = ["V/state.md", "V/b.md"]
    changed = diff(changed=targets)
    assert rr.score(case, rec("merge", targets, written=True), changed, merged("V/state.md", ["V/b.md"]))["correct"]
    # the old as_of stays on the merge target
    s = rr.score(case, rec("merge", targets, written=True), changed, merged("V/state.md", ["V/b.md"], "2026-01-01"))
    assert s["reasons"] == ["as-of"]
    # two targets with no superseded_by: no single merge target
    good = merged("V/state.md", ["V/b.md"])
    both = texts({t: good.before(t) for t in targets}, {t: good.after("V/state.md") for t in targets}, good.as_of)
    s = rr.score(case, rec("merge", targets, written=True), changed, both)
    assert s["reasons"] == ["merge-target"]


def test_a_supersede_that_rewrites_the_body_of_an_old_note_is_wrong():
    # the probe of review round 3, case 3
    old = "V/old-1.md"
    case = {**POSITIVE, "old": [{"path": old, "expected": "supersede"}]}
    good = superseded([old])
    rewritten = good.after(old).replace("The pump follows the curve.", "The boiler runs in the morning.")
    notes = texts({old: good.before(old)}, {old: rewritten, "V/new.md": good.after("V/new.md")})
    s = rr.score(case, rec("supersede", [old], written=True), diff(["V/new.md"], [old]), notes)
    assert not s["correct"] and s["reasons"] == ["old-body-changed"]
    # white space at line ends is not a change of the body
    spaced = good.after(old).replace("curve.", "curve.  ")
    notes = texts({old: good.before(old)}, {old: spaced, "V/new.md": good.after("V/new.md")})
    assert rr.score(case, rec("supersede", [old], written=True), diff(["V/new.md"], [old]), notes)["correct"]


def test_the_superseded_notes_of_a_merge_keep_their_body():
    case = {**POSITIVE, "old": [{"path": "V/state.md", "expected": "merge"}, {"path": "V/b.md", "expected": "merge"}]}
    targets = ["V/state.md", "V/b.md"]
    good = merged("V/state.md", ["V/b.md"])
    after = {t: good.after(t) for t in targets}
    after["V/b.md"] = after["V/b.md"].replace("curve.", "curve and more.")
    notes = texts({t: good.before(t) for t in targets}, after, good.as_of)
    s = rr.score(case, rec("merge", targets, written=True), diff(changed=targets), notes)
    assert s["reasons"] == ["old-body-changed"]


def test_a_merge_into_another_target_than_the_expected_one_is_wrong():
    # the probe of review round 3, case 4: state.md expects the merge, other.md the supersede
    case = {
        **POSITIVE,
        "old": [{"path": "V/state.md", "expected": "merge"}, {"path": "V/other.md", "expected": "supersede"}],
    }
    targets = ["V/state.md", "V/other.md"]
    right = merged("V/state.md", ["V/other.md"])
    s = rr.score(case, rec("merge", targets, written=True), diff(changed=targets), right)
    assert s["correct"] and s["merge_target"] == "V/state.md"
    wrong = merged("V/other.md", ["V/state.md"])
    s = rr.score(case, rec("merge", targets, written=True), diff(changed=targets), wrong)
    assert not s["correct"] and s["reasons"] == ["merge-target"] and s["merge_target"] == "V/other.md"
    # a merge for an expected supersede: any target can be the merge target, and the row names it
    s = rr.score(POSITIVE, rec("merge", OLDS, written=True), diff(changed=OLDS), merged(OLDS[1], OLDS[:1]))
    assert s["category"] == "merge-for-supersede" and s["merge_target"] == OLDS[1]


ENTITY = "V/Entities/Heat pump.md"
ENTITY_TEXT = note("# Heat pump\nA heat pump of the house.\n", type="entity", kind="product")


def test_a_created_entity_note_is_no_extra_write_for_any_action():
    # supersede: the new note and a new entity note
    good = superseded()
    notes = texts({o: good.before(o) for o in OLDS}, {**{k: good.after(k) for k in [*OLDS, "V/new.md"]},
                                                       ENTITY: ENTITY_TEXT})  # fmt: skip
    s = rr.score(POSITIVE, rec("supersede", OLDS, written=True), diff([ENTITY, "V/new.md"], OLDS), notes)
    assert s["correct"] and s["extra_writes"] == [] and s["entity_notes"] == [ENTITY]
    # merge
    good = merged()
    notes = texts({"V/state.md": good.before("V/state.md")}, {"V/state.md": good.after("V/state.md"),
                                                              ENTITY: ENTITY_TEXT}, good.as_of)  # fmt: skip
    s = rr.score(MERGE, rec("merge", ["V/state.md"], written=True), diff([ENTITY], ["V/state.md"]), notes)
    assert s["correct"] and s["entity_notes"] == [ENTITY]
    # new: the entity note sorts before the new note and does not take its place
    notes = texts({}, {ENTITY: ENTITY_TEXT, "V/fresh.md": note()})
    s = rr.score(NEGATIVE, rec("new", [], written=True), diff([ENTITY, "V/fresh.md"]), notes)
    assert s["correct"] and s["entity_notes"] == [ENTITY]
    # same and link: only the entity note is created
    notes = texts({}, {ENTITY: ENTITY_TEXT})
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"], written=True), diff([ENTITY]), notes)
    assert s["correct"] and s["entity_notes"] == [ENTITY]
    s = rr.score(DUPLICATE, rec("link", ["V/dup.md"], written=True), diff([ENTITY]), notes)
    assert s["extra_writes"] == [] and s["entity_notes"] == [ENTITY] and s["reasons"] == ["action"]


def test_a_created_entity_note_needs_the_note_texts_and_type_entity():
    # without the texts: the current rule, an extra write
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), diff([ENTITY]))
    assert s["extra_writes"] == [f"created {ENTITY}"] and s["entity_notes"] == []
    # a created note of another type is still an extra write
    notes = texts({}, {"V/other.md": note(type="guide")})
    s = rr.score(DUPLICATE, rec("same", ["V/dup.md"]), diff(["V/other.md"]), notes)
    assert s["extra_writes"] == ["created V/other.md"] and s["entity_notes"] == []
    # the only created note of a new is an entity (the draft is an entity): it is the new note
    notes = texts({}, {ENTITY: ENTITY_TEXT})
    s = rr.score(NEGATIVE, rec("new", [], written=True), diff([ENTITY]), notes)
    assert s["correct"] and s["entity_notes"] == []


def test_open_conflicts_are_reported_and_do_not_change_the_score():
    conflicts = [{"path": "V/third.md", "draft_claim": "a", "candidate_claim": "b"}, "V/fourth.md"]
    record = rec("same", ["V/dup.md"], open_conflicts=conflicts)
    assert rr.parse_record(f"{rr.PREFIX} {json.dumps(record)}") == (record, None)
    s = rr.score(DUPLICATE, record, EMPTY)
    assert s["correct"] and s["open_conflicts"] == ["V/fourth.md", "V/third.md"]
    assert rr.score(DUPLICATE, rec("same", ["V/dup.md"]), EMPTY)["open_conflicts"] == []
    assert rr.score(DUPLICATE, rec("same", ["V/dup.md"], open_conflicts="bad"), EMPTY)["correct"]


def test_the_summary_counts_entity_notes_and_open_conflicts():
    notes = texts({}, {ENTITY: ENTITY_TEXT})
    record = rec("same", ["V/dup.md"], written=True, open_conflicts=[{"path": "V/a.md"}, {"path": "V/b.md"}])
    rows = [
        {"kind": "duplicate", "positive_kind": None, "score": rr.score(DUPLICATE, record, diff([ENTITY]), notes)},
        {"kind": "duplicate", "positive_kind": None, "score": rr.score(DUPLICATE, None, EMPTY)},
        {"kind": "negative", "positive_kind": None, "score": {"correct": False, "category": "failed"}},
    ]
    s = rr.summarize(rows)
    assert s["entity_notes"] == 1 and s["open_conflicts"] == 2
    assert "entity notes 1" in rr.format_summary(s)


def test_the_bash_sandbox_closes_the_paths_around_the_case_home_and_the_plugin(tmp_path):
    top = tmp_path / "user"
    for rel in ("code/a/runs/r1/case/mirror", "code/a/runs/r0", "code/a/data", "code/plugin/scripts", "code/other",
                "Vaults/V", ".claude/shell-snapshots", "notes.txt"):  # fmt: skip
        path = top / rel
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
        else:
            path.mkdir(parents=True, exist_ok=True)
    case_dir, plugin = top / "code/a/runs/r1/case", top / "code/plugin"
    closed = rr.closed_around([case_dir, plugin], top, [top / ".claude"])
    assert closed == sorted(
        [top / "Vaults", top / "notes.txt", top / "code/other", top / "code/a/data", top / "code/a/runs/r0"]
    )
    assert rr.closed_around([tmp_path / "elsewhere"], top) == []
    settings = rr.case_settings(case_dir / "mirror", case_dir / "tmp", plugin, (), [], closed)
    assert settings["sandbox"]["filesystem"]["denyRead"] == [str(p) for p in closed]
    assert not any(str(p) in settings["permissions"]["deny"] for p in closed)  # the Read rules cover the tools


def test_wiki_links_resolve_by_path_or_basename():
    assert rr.link_names(["[[A/b|alias]]", "[[c#Heading]]"]) == ["A/b", "c"]
    assert rr.link_names("[[x]]") == ["x"] and rr.link_names("V/y.md") == ["V/y.md"] and rr.link_names(None) == []
    assert rr.links_to(["b"], "V/A/b.md") and rr.links_to(["A/b.md"], "V/A/b.md") and rr.links_to(["V/A/b"], "V/A/b.md")
    assert not rr.links_to(["ab"], "V/A/b.md") and not rr.links_to(["A"], "V/A/b.md")


def test_the_summary_counts_the_categories_with_the_wilson_interval():
    rows = [
        {
            "kind": "positive",
            "positive_kind": "content_change",
            "cost_usd": 1.5,
            "score": rr.score(
                POSITIVE, rec("supersede", OLDS, written=True), diff(["V/n.md"], OLDS), superseded(new="V/n.md")
            ),
        },
        {
            "kind": "positive",
            "positive_kind": "moved_copy",
            "cost_usd": 0.5,
            "score": rr.score(POSITIVE, rec("ask"), EMPTY),
        },
        {"kind": "duplicate", "positive_kind": None, "cost_usd": None, "score": rr.score(DUPLICATE, None, EMPTY)},
    ]
    s = rr.summarize(rows)
    assert s["all"]["correct"] == 1 and s["all"]["cases"] == 3
    assert s["all"]["low"] < 1 / 3 < s["all"]["high"]
    assert s["categories"] == {"correct": 1, "ask": 1, "failed": 1}
    assert s["by_kind"]["positive"]["correct"] == 1 and s["by_kind"]["positive"]["cases"] == 2
    assert s["by_positive_kind"]["moved_copy"]["correct"] == 0
    assert s["without_ask"]["cases"] == 2
    assert s["cost_usd"] == 2.0
    assert "correct" in rr.format_summary(s)


# --- the record parser -------------------------------------------------------------------------------------------


def test_the_record_is_the_last_record_line():
    first = json.dumps(rec("ask"))
    last = json.dumps(rec("same", ["V/dup.md"]))
    text = f"Some report.\nKB-WRITE-RECORD {first}\nmore text\nKB-WRITE-RECORD {last}\n\n"
    record, error = rr.parse_record(text)
    assert error is None and record["action"] == "same"


def test_a_malformed_or_absent_record_is_an_error():
    record, error = rr.parse_record("text\nKB-WRITE-RECORD {not json\n")
    assert record is None and "JSON" in error
    record, error = rr.parse_record("KB-WRITE-RECORD [1, 2]")
    assert record is None and "object" in error
    record, error = rr.parse_record("no record here")
    assert record is None and "KB-WRITE-RECORD" in error
    assert rr.parse_record(None)[0] is None


def test_a_record_with_a_wrong_shape_is_an_error():
    def parsed(**fields):
        return rr.parse_record("KB-WRITE-RECORD " + json.dumps({**rec("same", ["V/a.md"]), **fields}))

    assert parsed()[1] is None
    assert parsed(targets="V/a.md")[0] is None and "targets" in parsed(targets="V/a.md")[1]
    assert parsed(targets=[{"path": "V/a.md"}])[0] is None
    assert parsed(action=None)[0] is None
    assert parsed(written="no")[0] is None


def test_the_agent_output_gives_the_result_cost_turns_and_session():
    out = json.dumps(
        {
            "result": "ok",
            "total_cost_usd": 0.7,
            "num_turns": 12,
            "is_error": False,
            "permission_denials": [{"tool_name": "Write"}],
        }
    )
    parsed = rr.parse_agent_output(out)
    assert parsed["result"] == "ok" and parsed["cost_usd"] == 0.7 and parsed["turns"] == 12
    assert parsed["permission_denials"] == 1 and parsed["error"] is None
    stream = "\n".join(
        json.dumps(e)
        for e in [
            {
                "type": "system",
                "subtype": "init",
                "plugins": [{"name": "builtin-x"}],
                "mcp_servers": [{"name": rr.SERVER, "status": "connected"}],
                "tools": ["Read"],
                "permissionMode": "dontAsk",
            },  # fmt: skip
            {"type": "assistant", "message": {}},
            {"type": "result", "result": "done", "total_cost_usd": 0.1, "num_turns": 3, "permission_denials": []},
        ]
    )
    parsed = rr.parse_agent_output(stream)
    assert parsed["result"] == "done" and parsed["turns"] == 3
    assert parsed["session"] == {"plugins": ["builtin-x"], "mcp_servers": [rr.SERVER], "tools": ["Read"],
                                 "permission_mode": "dontAsk"}  # fmt: skip
    assert rr.parse_agent_output("crash")["error"]


# --- snapshots ---------------------------------------------------------------------------------------------------


def test_the_snapshot_diff_gives_created_changed_and_deleted_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "keep.md").write_text("k")
    (tmp_path / "a" / "edit.md").write_text("1")
    (tmp_path / "gone.md").write_text("g")
    before = rr.snapshot(tmp_path)
    (tmp_path / "a" / "edit.md").write_text("2")
    (tmp_path / "gone.md").unlink()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "new.md").write_text("n")
    after = rr.snapshot(tmp_path)
    assert rr.diff_snapshots(before, after) == {
        "created": ["a/b/new.md"],
        "changed": ["a/edit.md"],
        "deleted": ["gone.md"],
    }
    assert rr.diff_snapshots(after, after) == EMPTY


def test_a_symbolic_link_to_a_folder_is_in_the_snapshot(tmp_path):
    (tmp_path / "m").mkdir()
    before = rr.snapshot(tmp_path / "m")
    (tmp_path / "m" / "link").symlink_to(tmp_path)
    after = rr.snapshot(tmp_path / "m")
    assert after == {"link": f"link:{tmp_path}"}
    assert rr.diff_snapshots(before, after)["created"] == ["link"]


def test_changed_since_skips_the_named_folders_and_files(tmp_path):
    (tmp_path / "old.md").write_text("o")
    past = time.time_ns() - 10_000_000_000
    os.utime(tmp_path / "old.md", ns=(past, past))
    start = time.time_ns() - 1_000_000_000
    for rel in ("new.md", "skip/x.md", "own.txt"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("n")
    found = rr.changed_since(tmp_path, start, {tmp_path / "skip"}, {tmp_path / "own.txt"})
    assert found == [str(tmp_path / "new.md")]
    assert rr.changed_since(tmp_path / "nothing", start, set(), set()) == []


def test_strip_keys_removes_only_the_named_keys():
    text = (
        "---\ntitle: T\nsuperseded_by: '[[x]]'\nsupersedes:\n  - '[[a]]'\n  - '[[b]]'\nvalidity: superseded\n"
        "type: guide\n---\n# Body\nvalidity: stays\n"
    )
    out = rr.strip_keys(text, ["supersedes", "superseded_by", "validity"])
    assert out == "---\ntitle: T\ntype: guide\n---\n# Body\nvalidity: stays\n"
    assert rr.strip_keys(text, []) == text
    assert rr.strip_keys("# no frontmatter\n", ["validity"]) == "# no frontmatter\n"


def test_as_of_falls_back_to_updated_created_and_date():
    assert rr.note_as_of({"as_of": "2026-01-02", "updated": "2025-01-01"}) == "2026-01-02"
    assert rr.note_as_of({"updated": datetime.date(2026, 3, 4), "created": "2020-01-01"}) == "2026-03-04"
    assert rr.note_as_of({"created": "2026-05-06"}) == "2026-05-06"
    assert rr.note_as_of({"date": datetime.datetime(2026, 7, 8, 9, 0)}) == "2026-07-08"
    assert rr.note_as_of({}) is None


# --- the case home -----------------------------------------------------------------------------------------------

OLD = "V/Docs/old.md"
NEWER = "V/Docs/newer.md"
OTHER = "V/Misc/other.md"
SRC_NOTES = {
    OLD: (
        "---\ntype: guide\nclass: state\nvalidity: superseded\nsuperseded_by: '[[newer]]'\n---\n"
        "# Old\nThe boiler runs at night and the pump follows the curve.\n"
    ),
    NEWER: (
        "---\ntype: guide\nclass: state\nvalidity: current\nas_of: 2026-08-16\nsupersedes: ['[[old]]']\n---\n"
        "# Newer\nThe boiler runs in the morning and the pump follows the curve.\n"
    ),
    OTHER: "# Other\nThe eyepiece field stop.\n",
}
CASE = {
    "id": "p001",
    "kind": "positive",
    "positive_kind": "content_change",
    "newer": NEWER,
    "old": [{"path": OLD, "expected": "supersede", "similarity": 0.8}],
    "drafts": {"bare": "# Newer\nThe boiler runs in the morning.\n", "typed": "---\ntype: guide\n---\n# Newer\n"},
    "start_state": {"exclude": NEWER, "remove_keys": {OLD: ["superseded_by", "validity"]}},
}
CASE2 = {**CASE, "id": "p002"}


@pytest.fixture
def skill(tmp_path):
    """A plugin folder with the skill file and the helper script."""
    plugin = tmp_path / "plugin"
    (plugin / "skills" / "kb-write").mkdir(parents=True)
    (plugin / "scripts").mkdir()
    (plugin / "scripts" / "kb-write.sh").write_text("#!/usr/bin/env bash\n")
    path = plugin / "skills" / "kb-write" / "SKILL.md"
    path.write_text("# skill\n")
    past = time.time_ns() - 10_000_000_000
    for p in (path, plugin / "scripts" / "kb-write.sh"):
        os.utime(p, ns=(past, past))
    return path


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A source bench home with a mirror, an index made with a test embedder, and a set folder."""
    mirror = tmp_path / "mirror"
    for rel, text in SRC_NOTES.items():
        path = mirror / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    home = tmp_path / "home"
    (home / "data").mkdir(parents=True)
    (home / rr.MARKER).touch()
    cfg = Config(
        root=mirror,
        index_path=home / "data" / "index.db",
        vaults=["V"],
        dimension=len(TopicEmbedder.TOPICS),
        min_words=3,
    )
    (home / "config.toml").write_text(dump_config(cfg, root=mirror, data_dir="data"))
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    cfg = load_config(home / "config.toml")
    conn = connect(cfg)
    update(cfg, conn, TopicEmbedder())
    conn.close()
    set_dir = home / "replay" / "set-v1"
    set_dir.mkdir(parents=True)
    (set_dir / "cases.jsonl").write_text(json.dumps(CASE) + "\n" + json.dumps(CASE2) + "\n")
    (set_dir / "manifest.json").write_text("{}\n")
    # the files made before the run are older than its start
    past = time.time_ns() - 10_000_000_000
    for top, _, files in os.walk(tmp_path):
        for name in files:
            os.utime(Path(top) / name, ns=(past, past))
    return cfg, home, mirror, set_dir


def test_the_case_home_has_the_start_state_the_marker_and_its_own_root(bench, skill):
    cfg, home, mirror, set_dir = bench
    before = rr.snapshot(mirror)
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "bare", skill.parent.parent.parent)
    h = case_home.home
    assert h == set_dir / "runs" / "r1" / "p001"
    assert (h / rr.MARKER).is_file() and case_home.tmp == h / "tmp" and case_home.tmp.is_dir()
    assert not (case_home.mirror / NEWER).exists()
    fm, body = parse_note((case_home.mirror / OLD).read_text())
    assert "superseded_by" not in fm and "validity" not in fm and fm["class"] == "state"
    assert (case_home.mirror / OTHER).read_text() == SRC_NOTES[OTHER]
    case_cfg = load_config(h / "config.toml")
    assert case_cfg.root == case_home.mirror
    assert case_cfg.index_path == h / "data" / "index.db"
    assert case_home.draft.read_text() == CASE["drafts"]["bare"]
    assert case_home.as_of == "2026-08-16"
    assert case_home.index["embedded"] == 0
    conn = connect(case_cfg)
    paths = {p for (p,) in conn.execute("SELECT path FROM notes")}
    conn.close()
    assert paths == {OLD, OTHER}
    assert rr.snapshot(mirror) == before
    config = json.loads(case_home.mcp_config.read_text())
    server = config["mcpServers"][rr.SERVER]
    assert server["env"] == {"RAGNAROK_HOME": str(h)}
    assert server["command"] == sys.executable and server["args"] == ["-m", "ragnarok.cli", "serve"]


def test_the_typed_form_writes_the_typed_draft(bench, skill):
    cfg, home, mirror, set_dir = bench
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "typed", skill.parent.parent.parent)
    assert case_home.draft.read_text() == CASE["drafts"]["typed"]


def test_the_agent_loads_no_setting_source_and_only_exact_rules(bench, skill, tmp_path):
    cfg, home, mirror, set_dir = bench
    plugin = skill.parent.parent.parent
    guard = tmp_path / "vaults"
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "bare", plugin, (guard, tmp_path))
    argv = rr.agent_command("claude", case_home, budget=2.0, model="opus")
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--settings") + 1] == str(case_home.settings)
    assert argv[argv.index("--tools") + 1] == "Bash,Read,Write,Edit,Glob,Grep,ToolSearch"
    assert "--strict-mcp-config" in argv and argv[argv.index("--max-budget-usd") + 1] == "2.0"
    assert argv[argv.index("--model") + 1] == "opus"
    assert "--plugin-dir" not in argv and "--allowedTools" not in argv and "--add-dir" not in argv
    settings = json.loads(case_home.settings.read_text())
    allow = settings["permissions"]["allow"]
    script = plugin / "scripts" / "kb-write.sh"
    m = case_home.mirror
    assert [a for a in allow if a.startswith("Bash(")] == [
        f"Bash(bash {script} bench:*)",
        f'Bash(bash "{script}" bench:*)',
        f"Bash(KB_WRITE_WITHIN={m} bash {script}:*)",
        f'Bash(KB_WRITE_WITHIN="{m}" bash "{script}":*)',
        "Bash(test:*)",
        "Bash(diff:*)",
        "Bash(mktemp:*)",
        "Bash(ls:*)",
        "Bash(cat:*)",
        "Bash(date:*)",
    ]
    assert [a for a in allow if a.startswith(("Write", "Edit"))] == [f"Write(/{m}/**)", f"Edit(/{m}/**)"]
    # reads only in the case home and the plugin folder; no bare Read, Glob or Grep rule
    assert [a for a in allow if a.startswith(("Read", "Glob", "Grep"))] == [
        f"Read(/{case_home.home}/**)",
        f"Read(/{plugin}/**)",
    ]
    sandbox = settings["sandbox"]
    assert sandbox["enabled"] and sandbox["failIfUnavailable"] and not sandbox["allowUnsandboxedCommands"]
    assert not sandbox["autoAllowBashIfSandboxed"]
    assert sandbox["filesystem"]["allowWrite"] == [str(m), str(case_home.tmp)]
    # a guard that holds the case home cannot be closed to reads; the other guard is closed
    assert (
        sandbox["filesystem"]["denyRead"][-1] == str(guard) and str(tmp_path) not in sandbox["filesystem"]["denyRead"]
    )
    assert settings["permissions"]["deny"][-2:] == [f"Read(/{guard})", f"Read(/{guard}/**)"]
    prompt = rr.agent_prompt(case_home, skill, plugin)
    assert f"mirror={m}" in prompt and "2026-08-16" in prompt and str(case_home.draft) in prompt
    assert f"TMPDIR is {case_home.tmp}" in prompt
    env = rr.agent_env(case_home)
    assert env["TMPDIR"] == str(case_home.tmp) and env["RAGNAROK_HOME"] == str(case_home.home)
    assert set(env) <= set(rr.ENV_KEYS) | set(rr.AUTH_ENV_KEYS) | {"TMPDIR", "RAGNAROK_HOME"}


def test_the_agent_gets_the_auth_variables_that_are_set(bench, skill, monkeypatch):
    cfg, home, mirror, set_dir = bench
    case_home = rr.build_case_home(CASE, cfg, set_dir / "runs" / "r1", "bare", skill.parent.parent.parent)
    for key in rr.AUTH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/cfg")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("SOME_OTHER_TOKEN", "x")
    env = rr.agent_env(case_home)
    assert env["ANTHROPIC_API_KEY"] == "test-key" and env["CLAUDE_CONFIG_DIR"] == "/cfg"
    assert env["AWS_REGION"] == "eu-west-1" and "SOME_OTHER_TOKEN" not in env and "ANTHROPIC_BASE_URL" not in env
    assert {"ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"} <= set(rr.AUTH_ENV_KEYS)


def test_the_agent_cannot_read_the_answer_key_the_source_or_other_cases(bench, skill):
    cfg, home, mirror, set_dir = bench
    other_run = set_dir / "runs" / "20260101T000000Z-aaaaaa"
    other_run.mkdir(parents=True)
    (set_dir / "score-x.json").write_text("{}")
    (home / "replay" / "set-v0").mkdir()
    run_dir = set_dir / "runs" / "r1"
    case_home = rr.build_case_home(
        CASE, cfg, run_dir, "bare", skill.parent.parent.parent, bench_home=home, case_ids=["p001", "p002"]
    )
    settings = json.loads(case_home.settings.read_text())
    closed = [Path(p) for p in settings["sandbox"]["filesystem"]["denyRead"]]
    assert set(closed) == {
        mirror,
        home / "data",
        home / "config.toml",
        home / rr.MARKER,
        home / "replay" / "set-v0",
        set_dir / "cases.jsonl",
        set_dir / "manifest.json",
        set_dir / "score-x.json",
        other_run,
        run_dir / "p002",
        run_dir / "records.jsonl",
        run_dir / "summary.json",
    }
    assert list(case_home.closed) == closed
    deny = settings["permissions"]["deny"]
    for path in closed:
        assert f"Read(/{path})" in deny and f"Read(/{path}/**)" in deny
    # the case home and its parent folders stay readable
    assert not any(case_home.home.is_relative_to(p) for p in closed)


# --- the run -----------------------------------------------------------------------------------------------------


def init_event(**changes):
    """The init event of an isolated case session."""
    tools = [*rr.BUILTIN_TOOLS.split(","), f"mcp__{rr.SERVER}__search", f"mcp__{rr.SERVER}__find_collisions"]
    event = {
        "type": "system",
        "subtype": "init",
        "plugins": [{"name": "cc-plugin-telemetry", "path": "builtin", "source": "cc-plugin-telemetry@builtin"}],
        "mcp_servers": [{"name": rr.SERVER, "status": "connected"}],
        "tools": tools,
        "permissionMode": "dontAsk",
    }
    return {**event, **changes}


def tool_use(name, n=0, **args):
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "id": f"t{n}", "name": name, "input": args}]},
    }


def stream(*events):
    return "\n".join(json.dumps(e) for e in events)


def fake_runner(action, write=True, seen=None, calls=(), init=None, denied=()):
    """A runner that acts like the skill: it writes into the mirror of the prompt (the new note with `supersedes`,
    the old note with `superseded_by` and `validity`) and prints a transcript and a record."""

    def run(argv, prompt, cwd, env, timeout):
        if seen is not None:
            seen.append({"argv": argv, "cwd": cwd, "env": env})
        mirror = Path(re.search(r"mirror=(\S+)", prompt).group(1))
        script = re.search(r"The helper script is (\S+)\.\n", prompt).group(1)
        targets, written = [], False
        events = [init or init_event(), tool_use("Bash", 1, command=f"bash {script} bench {mirror}")]
        if write:
            fresh = mirror / "V" / "Docs" / "fresh.md"
            fresh.write_text("---\nclass: state\nvalidity: current\nsupersedes: ['[[old]]']\n---\n# Fresh\n")
            old = (mirror / OLD).read_text()
            new = old.replace("class: state", "class: state\nvalidity: superseded\nsuperseded_by: '[[fresh]]'")
            (mirror / OLD).write_text(new)
            targets, written = [OLD], True
            backup = cwd / "tmp" / "kb-write.abc123"
            events += [
                tool_use("Bash", 2, command=f"mktemp -d {cwd}/tmp/kb-write.XXXXXX"),
                tool_use(
                    "Bash", 3, command=f"KB_WRITE_WITHIN={mirror} bash {script} save {backup} {fresh} {mirror / OLD}"
                ),
                tool_use("Write", 4, file_path=str(fresh), content="..."),
                tool_use("Edit", 5, file_path=str(mirror / OLD), old_string="a", new_string="b"),
            ]
        events += [tool_use(name, 10 + i, **args) for i, (name, args) in enumerate(calls)]
        record = rec(action, targets, written=written)
        events.append(
            {
                "type": "result",
                "result": f"done\nKB-WRITE-RECORD {json.dumps(record)}",
                "total_cost_usd": 0.25,
                "num_turns": 9,
                "is_error": False,
                "permission_denials": [{"tool_use_id": f"t{10 + i}"} for i in denied],
            }
        )
        return rr.AgentOutput(exit_code=0, stdout=stream(*events), stderr="", duration_s=1.0)

    return run


def test_a_run_records_each_case_and_leaves_the_source_unchanged(bench, skill):
    cfg, home, mirror, set_dir = bench
    before = rr.snapshot(mirror)
    seen = []
    rows, summary, run_dir = rr.run_cases(
        cfg, home, set_dir, [CASE, CASE2], skill=skill, runner=fake_runner("supersede", seen=seen), parallel=2
    )
    assert rr.snapshot(mirror) == before
    assert run_dir.parent == (set_dir / "runs").resolve()
    assert [r["case"] for r in rows] == ["p001", "p002"]
    row = rows[0]
    assert row["form"] == "bare" and row["as_of"] == "2026-08-16"
    assert row["diff"] == {"created": ["V/Docs/fresh.md"], "changed": [OLD], "deleted": []}
    assert row["record"]["action"] == "supersede" and row["record_error"] is None
    assert row["cost_usd"] == 0.25 and row["turns"] == 9 and row["exit_code"] == 0
    assert row["session"]["mcp_servers"] == [rr.SERVER] and row["session"]["plugins"] == ["cc-plugin-telemetry@builtin"]
    assert row["score"]["correct"] and row["home_changes"] is None and row["score"]["effect_errors"] == []
    assert row["audit"] == {"tool_calls": 5, "denied": 0, "violations": []}
    assert summary["all"]["correct"] == 2 and summary["clean"] and summary["outside_warnings"] == []
    assert summary["audit_failures"] == {} and summary["home_changes"] == {}
    assert {s["cwd"] for s in seen} == {run_dir / "p001", run_dir / "p002"}
    assert all(s["env"]["TMPDIR"] == str(s["cwd"] / "tmp") for s in seen)
    assert not (run_dir / "p001" / "data").exists()  # the index copy is deleted after the case
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    assert sorted(json.loads(line)["case"] for line in lines) == ["p001", "p002"]
    assert json.loads((run_dir / "summary.json").read_text())["all"]["cases"] == 2


def test_keep_homes_keeps_the_index_copy(bench, skill):
    cfg, home, mirror, set_dir = bench
    _, _, run_dir = rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=fake_runner("same", False),
                                 keep_homes=True)  # fmt: skip
    assert (run_dir / "p001" / "data" / "index.db").is_file()


def test_a_crash_in_one_case_keeps_the_others_and_the_summary(bench, skill):
    cfg, home, mirror, set_dir = bench
    good = fake_runner("supersede")

    def runner(argv, prompt, cwd, env, timeout):
        if cwd.name == "p001":
            raise RuntimeError("boom")
        return good(argv, prompt, cwd, env, timeout)

    rows, summary, run_dir = rr.run_cases(cfg, home, set_dir, [CASE, CASE2], skill=skill, runner=runner, parallel=2)
    assert rows[0]["error"] == "RuntimeError: boom" and rows[0]["score"]["category"] == "failed"
    assert rows[1]["score"]["correct"]
    assert summary["failed_cases"] == ["p001"] and summary["categories"] == {"failed": 1, "correct": 1}
    assert len((run_dir / "records.jsonl").read_text().splitlines()) == 2
    assert not (run_dir / "p001" / "data").exists()


def test_a_bad_record_is_a_record_error(bench, skill):
    cfg, home, mirror, set_dir = bench

    def runner(argv, prompt, cwd, env, timeout):
        record = {**rec("same"), "targets": [{"path": OLD}]}
        out = stream(init_event(), {"type": "result", "result": f"KB-WRITE-RECORD {json.dumps(record)}"})
        return rr.AgentOutput(0, out, "", 1.0)

    rows, summary, _ = rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=runner)
    assert rows[0]["record"] is None and "targets" in rows[0]["record_error"]
    assert rows[0]["score"]["category"] == "failed" and rows[0]["error"] is None


def test_a_timeout_gives_a_failed_record(bench, skill):
    cfg, home, mirror, set_dir = bench

    def slow(argv, prompt, cwd, env, timeout):
        return rr.AgentOutput(exit_code=None, stdout="", stderr="", duration_s=5.0, timed_out=True)

    rows, summary, _ = rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=slow)
    assert rows[0]["timed_out"] and rows[0]["record"] is None and rows[0]["score"]["category"] == "failed"
    assert summary["categories"] == {"failed": 1} and summary["failed_cases"] == ["p001"]


def test_a_changed_source_mirror_fails_the_run(bench, skill):
    cfg, home, mirror, set_dir = bench

    def vandal(argv, prompt, cwd, env, timeout):
        (mirror / OTHER).write_text("changed\n")
        return fake_runner("same", write=False)(argv, prompt, cwd, env, timeout)

    with pytest.raises(rr.SourceChanged, match="other.md"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=vandal)


@pytest.mark.parametrize("where", ["case", "sibling"])
def test_a_change_of_a_case_home_file_fails_the_run(bench, skill, where):
    cfg, home, mirror, set_dir = bench

    def vandal(argv, prompt, cwd, env, timeout):
        {"case": cwd / "config.toml", "sibling": cwd / "evil.txt"}[where].write_text("changed\n")
        return fake_runner("same", write=False)(argv, prompt, cwd, env, timeout)

    with pytest.raises(rr.SourceChanged, match="case homes"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=vandal)
    (run_dir,) = (set_dir / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert not summary["clean"] and list(summary["home_changes"]) == ["p001"]


@pytest.mark.parametrize("where", ["home", "other-run", "plugin", "plugin-deleted", "guard"])
def test_a_change_outside_the_run_is_a_warning(bench, skill, tmp_path, where):
    cfg, home, mirror, set_dir = bench
    guard = tmp_path / "vaults"
    guard.mkdir()
    plugin = skill.parent.parent.parent

    def vandal(argv, prompt, cwd, env, timeout):
        if where == "plugin-deleted":
            skill.unlink()
        else:
            target = {
                "home": home / "replay" / "note.txt",
                "other-run": set_dir / "runs" / "other" / "records.jsonl",
                "plugin": plugin / "scripts" / "kb-write.sh",
                "guard": guard / "x.md",
            }[where]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("changed\n")
        return fake_runner("same", write=False)(argv, prompt, cwd, env, timeout)

    rows, summary, _ = rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=vandal, guards=(guard,))
    assert summary["clean"] and not rows[0]["error"] and rows[0]["audit"]["violations"] == []
    if where == "other-run":
        assert summary["outside_warnings"] == []  # another replay run of the same home
    else:
        assert len(summary["outside_warnings"]) == 1
    if where == "plugin-deleted":
        assert summary["outside_warnings"] == [f"deleted {skill}"]


def test_an_audit_violation_fails_the_case_and_the_run(bench, skill, tmp_path):
    cfg, home, mirror, set_dir = bench
    outside = tmp_path / "elsewhere.md"
    runner = fake_runner("supersede", calls=[("Write", {"file_path": str(outside), "content": "x"})])
    with pytest.raises(rr.SourceChanged, match="audit"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=runner)
    (run_dir,) = (set_dir / "runs").iterdir()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["audit_failures"] == {"p001": [f"Write outside the case mirror: {outside}"]}
    assert summary["failed_cases"] == ["p001"] and summary["categories"] == {"failed": 1}
    row = json.loads((run_dir / "records.jsonl").read_text())
    assert row["score"]["category"] == "failed" and "audit" in row["score"]["reasons"]
    # a call that the permission rules denied did not run: no violation
    runner = fake_runner("supersede", calls=[("Write", {"file_path": str(outside), "content": "x"})], denied=[0])
    rows, summary, _ = rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=runner)
    assert summary["clean"] and rows[0]["audit"]["denied"] == 1 and rows[0]["score"]["correct"]


def test_a_session_with_a_user_plugin_fails_the_run(bench, skill):
    cfg, home, mirror, set_dir = bench
    plugins = [{"name": "ragnarok", "source": "ragnarok@ragnarok"}]
    runner = fake_runner("same", write=False, init=init_event(plugins=plugins))
    with pytest.raises(rr.SourceChanged, match="ragnarok@ragnarok"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=runner)


def audit_home(tmp_path):
    home = tmp_path / "case"
    plugin = tmp_path / "plugin"
    closed = (tmp_path / "src", tmp_path / "set" / "cases.jsonl")
    return rr.CaseHome(home, home / "mirror", home / "tmp", home / "draft.md", home / "mcp.json",
                       home / "settings.json", None, {}, closed), plugin  # fmt: skip


def test_the_audit_allows_the_helper_and_shell_commands_of_the_skill(tmp_path):
    case, plugin = audit_home(tmp_path)
    m, t, script = case.mirror, case.tmp, plugin / "scripts" / "kb-write.sh"
    for command in (
        f"bash {script} bench {m}",
        f'bash "{script}" bench "{m}"',
        f"mktemp -d {t}/kb-write.XXXXXX",
        f"KB_WRITE_WITHIN={m} bash {script} save {t}/kb-write.a {m}/V/new.md {m}/V/old.md",
        f'KB_WRITE_WITHIN="{m}" bash "{script}" restore "{t}/kb-write.a"',
        f'test -e "{m}/V/A note.md"',
        f"diff -u /dev/null {t}/draft.md",
        "date -u +%Y%m%dT%H%M%SZ",
        "ls mirror/V",
        f"cat {plugin}/skills/kb-write/SKILL.md",
    ):
        assert rr.bash_errors(command, case, plugin) == [], command


def test_the_audit_finds_other_commands_and_outside_paths(tmp_path):
    case, plugin = audit_home(tmp_path)
    m, script = case.mirror, plugin / "scripts" / "kb-write.sh"
    for command, word in (
        (f"bash {script} save /tmp/b {m}/V/a.md", "KB_WRITE_WITHIN"),
        (f"KB_WRITE_WITHIN=/tmp bash {script} save /tmp/b /tmp/a.md", "KB_WRITE_WITHIN"),
        (f"rm -r {m}", "not an allowed"),
        ("cat /etc/hosts", "outside"),
        (f"cat {m}/../../secret.md", "outside"),
        ("cat ~/notes.md", "~"),
        (f"ls {m} ; touch x", "compound"),
        (f"cat {m}/a.md | tee x", "compound"),
        (f"cat {m}/a.md > {m}/b.md", "compound"),
        ('mktemp -d "${TMPDIR}/x.XXXXXX"', "expansion"),
        ("cat 'unclosed", "parsed"),
    ):
        errors = rr.bash_errors(command, case, plugin)
        assert errors and word in errors[0], command


def test_the_audit_checks_writes_reads_and_tools(tmp_path):
    case, plugin = audit_home(tmp_path)
    out = stream(
        init_event(),
        tool_use("Write", 1, file_path=str(case.mirror / "V" / "a.md")),
        tool_use("Edit", 2, file_path=str(case.home / "settings.json")),
        tool_use("Read", 3, file_path=str(tmp_path / "set" / "cases.jsonl")),
        tool_use("Grep", 4, pattern="x", path=str(tmp_path / "src" / "V")),
        tool_use("Glob", 5, pattern=str(tmp_path / "src") + "/**/*.md"),
        tool_use("Read", 6, file_path=str(case.draft)),
        tool_use("WebFetch", 7, url="https://example.com"),
        tool_use(f"mcp__{rr.SERVER}__search", 8, query="x"),
        tool_use("Write", 9, file_path="/etc/x"),
        tool_use("Read", 10, file_path=str(plugin / "skills" / "kb-write" / "SKILL.md")),
        tool_use("Grep", 11, pattern="x", path="mirror/V"),
        tool_use("Grep", 12, pattern="x"),
        tool_use("Glob", 13, pattern="**/*.md"),
        tool_use("Read", 14, file_path=str(tmp_path / "other-data" / "newer.md")),
        tool_use("Grep", 15, pattern="x", path="../.."),
        tool_use("Glob", 16, pattern="/elsewhere/**/*.md"),
        {"type": "result", "result": "", "permission_denials": [{"tool_use_id": "t9"}]},
    )
    audit = rr.audit_agent(out, case, plugin)
    assert audit["tool_calls"] == 16 and audit["denied"] == 1
    assert audit["violations"] == [
        f"Edit outside the case mirror: {case.home / 'settings.json'}",
        f"Read of a closed path: {tmp_path / 'set' / 'cases.jsonl'}",
        f"Grep of a closed path: {tmp_path / 'src' / 'V'}",
        f"Glob of a closed path: {tmp_path / 'src'}/**/*.md",
        "a tool that the runner did not give: WebFetch",
        f"Read outside the case home and the plugin folder: {tmp_path / 'other-data' / 'newer.md'}",
        "Grep outside the case home and the plugin folder: ../..",
        "Glob outside the case home and the plugin folder: /elsewhere/**/*.md",
    ]
    assert rr.audit_agent("", case, plugin)["violations"] == []  # no output: nothing ran
    no_init = rr.audit_agent(stream({"type": "result"}), case, plugin)
    assert no_init["violations"] == ["the agent output has no init event"]


def test_the_session_must_be_the_one_that_the_runner_gave():
    good = rr.parse_agent_output(stream(init_event(), {"type": "result"}))["session"]
    assert rr.session_errors(good) == []
    bad = {
        **good,
        "plugins": [*good["plugins"], "ha@ha"],
        "mcp_servers": [rr.SERVER, "home-assistant"],
        "permission_mode": "default",
        "tools": [*good["tools"], "WebFetch", "mcp__home-assistant__ha_restart"],
    }
    errors = rr.session_errors(bad)
    assert len(errors) == 5 and "ha@ha" in errors[0] and "home-assistant" in errors[1] and "default" in errors[2]
    assert rr.session_errors(None) == ["the agent output has no init event"]
    assert rr.session_errors({**good, "plugins": ["cc-plugin-x"]})  # a built-in name with no @builtin source


def test_the_run_options_are_checked(bench, skill, tmp_path):
    cfg, home, mirror, set_dir = bench
    run = fake_runner("same", False)
    for bad in ({"parallel": 0}, {"timeout": 0}, {"budget": -1}):
        with pytest.raises(ValueError):
            rr.run_cases(cfg, home, set_dir, [CASE], skill=skill, runner=run, **bad)
    with pytest.raises(ValueError, match="repeated"):
        rr.run_cases(cfg, home, set_dir, [CASE, CASE], skill=skill, runner=run)
    loose = tmp_path / "SKILL.md"
    loose.write_text("# copy\n")
    with pytest.raises(ValueError, match="kb-write.sh"):
        rr.run_cases(cfg, home, set_dir, [CASE], skill=loose, runner=run)
    assert not (set_dir / "runs").exists()


def test_the_subprocess_runner_kills_the_agent_and_its_children(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "agent.py"
    script.write_text(
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen(['sleep', '60'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    out = rr.subprocess_runner([sys.executable, str(script)], "", tmp_path, dict(os.environ), timeout=2)
    assert out.timed_out and out.exit_code is None
    pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the child of the agent is still running")
    done = rr.subprocess_runner([sys.executable, "-c", "import sys; print(sys.stdin.read())"], "hi", tmp_path,
                                dict(os.environ), timeout=30)  # fmt: skip
    assert done.exit_code == 0 and done.stdout.strip() == "hi" and not done.timed_out


def test_the_run_refuses_a_home_with_no_bench_marker(bench, skill, capsys):
    cfg, home, mirror, set_dir = bench
    (home / rr.MARKER).unlink()
    assert main(["replay", "run", "--set", str(set_dir), "--skill", str(skill)]) == 1
    assert rr.MARKER in capsys.readouterr().err


def test_the_run_refuses_a_set_outside_the_home(bench, skill, tmp_path, capsys):
    outside = tmp_path / "set"
    outside.mkdir()
    (outside / "cases.jsonl").write_text(json.dumps(CASE) + "\n")
    (outside / "manifest.json").write_text("{}\n")
    assert main(["replay", "run", "--set", str(outside), "--skill", str(skill)]) == 1
    assert "inside" in capsys.readouterr().err


def test_the_cli_runs_the_chosen_cases(bench, skill, monkeypatch, capsys):
    cfg, home, mirror, set_dir = bench
    monkeypatch.setattr(rr, "subprocess_runner", fake_runner("supersede"))
    args = ["replay", "run", "--set", str(set_dir), "--skill", str(skill)]
    assert main([*args, "--cases", "p001"]) == 0
    assert "correct" in capsys.readouterr().out
    assert main([*args, "--cases", "p999"]) == 1
    assert "p999" in capsys.readouterr().err
    assert main([*args, "--cases", "p001,p001"]) == 1
    assert "repeated" in capsys.readouterr().err
    assert main([*args, "--cases", "p001", "--parallel", "0"]) == 1
    assert "--parallel" in capsys.readouterr().err


def test_the_cli_exits_non_zero_when_a_case_fails(bench, skill, monkeypatch, capsys):
    cfg, home, mirror, set_dir = bench

    def crash(argv, prompt, cwd, env, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(rr, "subprocess_runner", crash)
    assert main(["replay", "run", "--set", str(set_dir), "--skill", str(skill), "--cases", "p001"]) == 1
    assert "p001" in capsys.readouterr().err
