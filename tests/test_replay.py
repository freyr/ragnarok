import hashlib
import inspect
import json

import pytest
from test_search import TopicEmbedder

from ragnarok import collisions, replay
from ragnarok.cli import main
from ragnarok.config import Config, dump_config
from ragnarok.index import connect, update
from ragnarok.metadata import parse_note
from ragnarok.tools import Tools

PUMP = "V/Entities/Pump.md"
FAN = "V/Entities/Fan.md"
NEW_A = "V/Docs/new-a.md"
OLD_A1 = "V/Docs/old-a1.md"
OLD_A2 = "V/Docs/old-a2.md"
OLD_A3 = "V/Docs/old-a3.md"
OLD_X = "V/Docs/old-x.md"
OLD_Y = "V/Docs/old-y.md"
NEW_F = "V/Docs/free.md"
OLD_Z = "V/Docs/old-z.md"
NEW_M = "V/Docs/new-m.md"
OLD_M = "V/Archive/old-m.md"
NEW_D = "V/Docs/new-d.md"
OLD_D1 = "V/Archive/old-d1.md"
OLD_D2 = "V/Archive/old-d2.md"
NEW_E = "V/Docs/new-e.md"
OLD_E = "V/Archive/old-e.md"
LONE = "V/Misc/lone.md"
PARTIAL = "V/Misc/partial.md"
SOLO = "V/Misc/solo.md"
SHARED = "V/Misc/shared.md"
LINK_OUT = "V/Misc/link-out.md"
BUDGET = "V/Misc/budget.md"
DB1 = "V/Misc/db-1.md"
DB2 = "V/Misc/db-2.md"

NEW_A_BODY = "# New A\nThe boiler defrost cycle runs at night. See [[old-a1]] for the history.\n"
# 20 words; the old copy changes one word (night -> morning): ratio 2 * 19 / 40 = 0.95
MOVED = (
    "# Moved\nThe boiler pump runs every {} and the flow follows the outdoor weather curve of the installer today.\n"
)
DUP_BODY = "# Dup\nThe boiler   valve list. Keep `[[old-d2]]` aside.\n"

NOTES = {
    PUMP: "---\ntype: entity\nkind: device\n---\n# Pump\n",
    FAN: "---\ntype: entity\nkind: device\n---\n# Fan\n",
    NEW_A: (
        "---\ntype: guide\nclass: state\nvalidity: current\nsubject: ['[[Pump]]']\ntask: defrost\n"
        "description: The new defrost guide.\nsupersedes: ['[[old-a1]]', 'old notes about the pump']\n---\n"
        + NEW_A_BODY
    ),
    # a state note with the same identity key (the task differs only in case): merge
    OLD_A1: (
        "---\ntype: guide\nclass: state\nvalidity: superseded\nsubject: ['[[Pump]]']\ntask: Defrost\n"
        "superseded_by: '[[new-a]]'\n---\n# Old A1\nThe boiler defrost cycle ran in the morning.\n"
    ),
    # a record with the same key: supersede
    OLD_A2: (
        "---\ntype: guide\nclass: record\nsubject: ['[[Pump]]']\ntask: defrost\nsuperseded_by: '[[new-a]]'\n---\n"
        "# Old A2\nThe boiler log of the first winter.\n"
    ),
    # a bare path that resolves; no subject: supersede
    OLD_A3: "---\ntype: guide\nsuperseded_by: V/Docs/new-a.md\n---\n# Old A3\nThe boiler notes.\n",
    # a bare path that resolves to no note
    OLD_X: "---\nsuperseded_by: Docs/nothing-here\n---\n# Old X\nThe boiler draft.\n",
    # a wiki-link that resolves to no note
    OLD_Y: "---\nsuperseded_by: '[[missing-note]]'\n---\n# Old Y\nThe boiler sketch.\n",
    # supersedes alone (free text that names an entity alias, and a link) makes no pair
    NEW_F: "---\nsupersedes: ['Pump', '[[old-z]]']\n---\n# Free\nThe boiler fan.\n",
    OLD_Z: "# Old Z\nThe boiler plan.\n",
    NEW_M: "---\ntype: guide\nproject: heat\n---\n" + MOVED.format("night"),
    OLD_M: "---\ntype: guide\nsuperseded_by: '[[new-m]]'\n---\n" + MOVED.format("morning"),
    NEW_D: "---\ntype: reference\n---\n" + DUP_BODY,
    OLD_D1: (
        "---\ntype: reference\nsuperseded_by: '[[new-d]]'\nvalidity: superseded\n---\n"
        "# Dup\n\nThe boiler valve list.  Keep `[[old-d2]]` aside.\n"
    ),
    OLD_D2: (
        "---\nsuperseded_by: '[[new-d]]'\nsupersedes: []\n---\n# Valves\nThe boiler valve list of the old house.\n"
    ),
    NEW_E: "# Echo\nThe boiler echo.\n",
    OLD_E: "---\nsuperseded_by: '[[new-e]]'\n---\n# Echo\nThe boiler echo.\n",
    # the negative pool
    LONE: "# Lone\nThe queue grows.\n",
    PARTIAL: "# Partial\nThe queue and the person.\n",
    SOLO: "---\nsubject: ['[[Fan]]']\n---\n# Solo\nThe eyepiece cap.\n",
    SHARED: "---\nsubject: ['[[Pump]]']\n---\n# Shared\nThe incident report.\n",
    LINK_OUT: "# Link out\nThe password list. See [[budget]].\n",
    BUDGET: "# Budget\nThe budget sheet.\n",
    DB1: "# Index\nThe database index.\n",
    DB2: "# Backup\nThe database backup.\n",
}

REMOVED = ("supersedes", "superseded_by", "validity")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    return Config(
        root=tmp_path,
        index_path=home / "data" / "index.db",
        vaults=["V"],
        dimension=len(TopicEmbedder.TOPICS),
    )


@pytest.fixture
def conn(cfg):
    c = connect(cfg)
    update(cfg, c, TopicEmbedder())
    return c


@pytest.fixture
def built(cfg, conn):
    return replay.build(cfg, conn)


def cases_of(result, kind):
    return [c for c in result.cases if c["kind"] == kind]


def by_newer(result, kind="positive"):
    return {c["newer"]: c for c in cases_of(result, kind)}


def old_paths(case):
    return [o["path"] for o in case["old"]]


# --- the positive and duplicate cases ----------------------------------------------------------------------------


def test_positive_cases_are_grouped_by_the_newer_note(built):
    cases = by_newer(built)
    assert sorted(cases) == [NEW_A, NEW_D, NEW_M]
    assert sorted(old_paths(cases[NEW_A])) == [OLD_A1, OLD_A2, OLD_A3]
    assert [c["id"] for c in cases_of(built, "positive")] == ["p001", "p002", "p003"]
    assert [c["newer"] for c in cases_of(built, "positive")] == [NEW_A, NEW_D, NEW_M]


def test_a_bare_path_counts_only_when_it_resolves(built):
    assert OLD_A3 in old_paths(by_newer(built)[NEW_A])
    every_old = [p for c in built.cases for p in old_paths(c)]
    assert OLD_X not in every_old
    assert OLD_Y not in every_old
    assert built.counts["superseded_by_refs"] == 9
    assert built.counts["bare_resolved"] == 1
    assert built.counts["bare_dropped"] == 1
    assert built.counts["unresolved_links"] == 1


def test_supersedes_gives_no_pair(built):
    assert NEW_F not in by_newer(built)
    every_old = [p for c in built.cases for p in old_paths(c)]
    assert OLD_Z not in every_old and PUMP not in every_old
    # the supersedes link of NEW_A repeats a superseded_by pair: one old note, listed once
    assert old_paths(by_newer(built)[NEW_A]).count(OLD_A1) == 1


def test_an_equal_body_pair_is_a_duplicate_case_only(built):
    dups = cases_of(built, "duplicate")
    assert [(c["id"], c["newer"], old_paths(c)) for c in dups] == [
        ("d001", NEW_D, [OLD_D1]),
        ("d002", NEW_E, [OLD_E]),
    ]
    assert all(c["old"][0]["expected"] == "same" for c in dups)
    assert old_paths(by_newer(built)[NEW_D]) == [OLD_D2]  # OLD_D1 left the positive case
    assert NEW_E not in by_newer(built)  # no old note left: no positive case


def test_moved_copy_or_content_change_by_the_similarity(cfg, conn, built):
    cases = by_newer(built)
    assert cases[NEW_M]["positive_kind"] == "moved_copy"
    assert cases[NEW_M]["old"][0]["similarity"] == pytest.approx(0.95)
    assert cases[NEW_A]["positive_kind"] == "content_change"
    assert cases[NEW_D]["positive_kind"] == "content_change"
    strict = replay.build(cfg, conn, copy_similarity=0.96)
    assert by_newer(strict)[NEW_M]["positive_kind"] == "content_change"
    assert built.counts["moved_copy"] == 1
    assert built.counts["content_change"] == 2


def test_the_bare_draft_has_no_frontmatter(built):
    for case in [*cases_of(built, "positive"), *cases_of(built, "duplicate")]:
        fm, _ = parse_note(case["drafts"]["bare"])
        assert fm == {}
        assert not case["drafts"]["bare"].startswith("---")
    assert by_newer(built)[NEW_A]["drafts"]["bare"].strip() == NEW_A_BODY.strip()


def test_the_typed_draft_has_only_type_subject_and_task(built):
    cases = by_newer(built)
    fm, body = parse_note(cases[NEW_A]["drafts"]["typed"])
    assert fm == {"type": "guide", "subject": ["[[Pump]]"], "task": "defrost"}
    assert body.strip() == NEW_A_BODY.strip()
    assert parse_note(cases[NEW_M]["drafts"]["typed"])[0] == {"type": "guide"}  # project is not kept
    echo = by_newer(built, "duplicate")[NEW_E]
    assert parse_note(echo["drafts"]["typed"])[0] == {}


def test_the_leak_flag(built):
    cases = by_newer(built)
    assert cases[NEW_A]["leak"] is True  # [[old-a1]] in the body
    assert cases[NEW_D]["leak"] is False  # [[old-d2]] only in inline code
    assert cases[NEW_M]["leak"] is False
    assert built.counts["leak"] == 1


def test_merge_or_supersede(built):
    expected = {o["path"]: o["expected"] for o in by_newer(built)[NEW_A]["old"]}
    assert expected == {OLD_A1: "merge", OLD_A2: "supersede", OLD_A3: "supersede"}
    assert by_newer(built)[NEW_M]["old"][0]["expected"] == "supersede"  # no subject
    assert built.counts["merge"] == 1
    assert built.counts["supersede"] == 4
    assert built.counts["old_notes"] == 5


def test_the_start_state_record(built):
    case = by_newer(built)[NEW_D]
    assert case["start_state"]["exclude"] == NEW_D
    assert case["start_state"]["remove_keys"] == {OLD_D2: ["supersedes", "superseded_by"]}
    dup = by_newer(built, "duplicate")[NEW_D]
    assert dup["start_state"]["remove_keys"] == {OLD_D1: ["superseded_by", "validity"]}
    for case in built.cases:
        assert all(set(keys) <= set(REMOVED) for keys in case["start_state"]["remove_keys"].values())


# --- the negative cases ------------------------------------------------------------------------------------------


def test_the_negative_pool_follows_the_rules(cfg, conn, built):
    pool = replay.negative_pool(cfg, conn, built.cases)
    # in a case, a shared subject, a link out, a link in: not in the pool
    for path in (NEW_A, OLD_A1, OLD_M, NEW_E, SHARED, LINK_OUT, BUDGET):
        assert path not in pool
    # 1 - cos of (queue) and (queue, person) = 1 - 1/sqrt(2)
    assert pool[LONE] == pytest.approx(0.29289, abs=1e-4)
    assert pool[PARTIAL] == pytest.approx(0.29289, abs=1e-4)
    # (eyepiece) and an entity with no topic (all ones): 1 - 1/sqrt(8)
    assert pool[SOLO] == pytest.approx(0.64645, abs=1e-4)
    assert pool[DB1] == pytest.approx(0.0, abs=1e-6)
    assert built.qualifying == {"0.15": 3, "0.20": 3, "0.25": 3, "0.30": 1}


def test_negative_cases_are_sampled_with_the_seed(cfg, conn):
    first = replay.build(cfg, conn, neg_count=2, neg_distance=0.25, seed=3)
    again = replay.build(cfg, conn, neg_count=2, neg_distance=0.25, seed=3)
    negatives = cases_of(first, "negative")
    assert [c["newer"] for c in negatives] == [c["newer"] for c in cases_of(again, "negative")]
    assert len(negatives) == 2
    assert {c["newer"] for c in negatives} <= {LONE, PARTIAL, SOLO}
    for i, case in enumerate(negatives, start=1):
        assert case["id"] == f"n{i:03d}"
        assert case["old"] == [] and case["expected"] == "new"
        assert case["start_state"] == {"exclude": case["newer"], "remove_keys": {}}
        assert parse_note(case["drafts"]["bare"])[0] == {}
    solo = replay.build(cfg, conn, neg_count=1, neg_distance=0.30)
    assert [c["newer"] for c in cases_of(solo, "negative")] == [SOLO]
    assert parse_note(cases_of(solo, "negative")[0]["drafts"]["typed"])[0] == {"subject": ["[[Fan]]"]}


def test_no_negative_cases_by_default_and_too_many_fails(cfg, conn, built):
    assert cases_of(built, "negative") == []
    with pytest.raises(ValueError):
        replay.build(cfg, conn, neg_count=4, neg_distance=0.25)
    with pytest.raises(ValueError):
        replay.build(cfg, conn, neg_count=1)  # no distance


# --- the Wilson interval and the summary -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hits", "n", "low", "high"),
    [
        (9, 10, 0.59585, 0.98212),
        (0, 5, 0.0, 0.43449),
        (25, 25, 0.86680, 1.0),
        (4, 5, 0.37553, 0.96378),
    ],
)
def test_the_wilson_interval(hits, n, low, high):
    lo, hi = replay.wilson(hits, n)
    assert lo == pytest.approx(low, abs=1e-4)
    assert hi == pytest.approx(high, abs=1e-4)


def test_the_wilson_interval_of_no_cases():
    assert replay.wilson(0, 0) == (None, None)


def _record(i, form, old):
    return {
        "id": f"c{i}",
        "form": form,
        "leak": i in (0, 1),
        "positive_kind": "moved_copy" if i < 5 else "content_change",
        "old": [{"path": f"o{i}-{j}", "position": p, "reasons": r} for j, (p, r) in enumerate(old)],
    }


def test_the_summary_for_each_form_and_subset():
    records = []
    for i in range(10):
        if i < 5:
            bare = [(1, ["search"])]
        elif i < 7:
            bare = [(2, ["search", "links_out"])]
        elif i < 9:
            bare = [(4, ["identity", "search"])]
        else:
            bare = [(3, ["search"]), (None, [])]  # one of two old notes is missing: no hit
        records.append(_record(i, "bare", bare))
        typed = bare if i < 5 else [(None, [])] * len(bare)
        records.append(_record(i, "typed", typed))
    summary = replay.summarize(records)
    bare, typed = summary["bare"], summary["typed"]
    assert (bare["all"]["hits"], bare["all"]["cases"]) == (9, 10)
    assert bare["all"]["r1"] == pytest.approx(0.9)
    assert bare["all"]["low"] == pytest.approx(0.59585, abs=1e-4)
    assert bare["all"]["high"] == pytest.approx(0.98212, abs=1e-4)
    assert (bare["no_leak"]["hits"], bare["no_leak"]["cases"]) == (7, 8)
    assert (bare["moved_copy"]["hits"], bare["moved_copy"]["cases"]) == (5, 5)
    assert (bare["content_change"]["hits"], bare["content_change"]["cases"]) == (4, 5)
    sources = bare["sources"]
    assert (sources["search"]["found"], sources["search"]["old_notes"]) == (10, 11)
    assert sources["links_out"]["found"] == 2
    assert sources["identity"]["found"] == 2
    assert sources["links_to_supersedes"]["found"] == 0
    assert sources["links_out"]["share"] == pytest.approx(2 / 11)
    assert (typed["all"]["hits"], typed["all"]["cases"]) == (5, 10)
    assert (typed["no_leak"]["hits"], typed["no_leak"]["cases"]) == (3, 8)
    assert (typed["content_change"]["hits"], typed["content_change"]["cases"]) == (0, 5)
    assert typed["content_change"]["low"] == pytest.approx(0.0, abs=1e-6)
    assert typed["content_change"]["high"] == pytest.approx(0.43449, abs=1e-4)


# --- the score ---------------------------------------------------------------------------------------------------


def test_find_collisions_excludes_a_note_before_the_limit(tmp_path):
    notes = {f"V/n{i:02d}.md": f"# N{i}\nThe boiler note number {i} with more words.\n" for i in range(1, 12)}
    notes["V/n00.md"] = "# Boiler note\nThe boiler note.\n"
    for rel, text in notes.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    c = Config(root=tmp_path, index_path=tmp_path / "index.db", vaults=["V"], dimension=len(TopicEmbedder.TOPICS))
    db = connect(c)
    update(c, db, TopicEmbedder())
    draft = "# Boiler note\nThe boiler note.\n"
    plain = [x["path"] for x in collisions.find_collisions(c, db, TopicEmbedder(), draft=draft)]
    assert "V/n00.md" in plain and len(plain) == 10
    found = [x["path"] for x in collisions.find_collisions(c, db, TopicEmbedder(), draft=draft, exclude="V/n00.md")]
    assert "V/n00.md" not in found
    assert len(found) == 10  # the pool grows by the excluded note
    assert set(plain) - {"V/n00.md"} < set(found)


def test_the_mcp_tool_has_no_exclude_parameter():
    assert "exclude" not in inspect.signature(Tools.find_collisions).parameters


def test_score_runs_each_positive_case_and_form_without_the_newer_note(cfg, conn, built):
    records = replay.score_cases(cfg, conn, TopicEmbedder(), built.cases)
    positives = cases_of(built, "positive")
    assert len(records) == 2 * len(positives)  # duplicate and negative cases are skipped
    assert {(r["id"], r["form"]) for r in records} == {(c["id"], f) for c in positives for f in ("bare", "typed")}
    newer = {c["id"]: c["newer"] for c in positives}
    for r in records:
        assert newer[r["id"]] not in r["candidates"]
        assert len(r["candidates"]) <= 10
        for o in r["old"]:
            if o["path"] in r["candidates"]:
                assert o["position"] == r["candidates"].index(o["path"]) + 1
            else:
                assert o["position"] is None and o["reasons"] == []


# --- the commands ------------------------------------------------------------------------------------------------


@pytest.fixture
def cli(cfg, conn, tmp_path, monkeypatch):
    path = tmp_path / "home" / "config.toml"
    path.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path))
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda c: TopicEmbedder())
    return str(path)


def test_build_writes_the_set_the_manifest_and_the_marker(cli, cfg, tmp_path, capsys):
    out = tmp_path / "home" / "replay" / "set-v1"
    assert main(["--config", cli, "replay", "build", "--out", str(out)]) == 0
    assert (tmp_path / "home" / ".ragnarok-bench").is_file()
    lines = (out / "cases.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["p001", "p002", "p003", "d001", "d002"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["cases_sha256"] == hashlib.sha256((out / "cases.jsonl").read_bytes()).hexdigest()
    mirror = "".join(f"{rel}\t{hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest()}\n" for rel in sorted(NOTES))
    assert manifest["mirror_sha256"] == hashlib.sha256(mirror.encode()).hexdigest()
    assert manifest["mirror_notes"] == len(NOTES)
    assert manifest["counts"]["positive"] == 3 and manifest["counts"]["duplicate"] == 2
    assert manifest["parameters"]["copy_similarity"] == 0.9
    assert manifest["parameters"]["neg_count"] == 0
    assert "commit" in manifest
    printed = capsys.readouterr().out
    assert "0.30" in printed and "positive" in printed


def test_build_refuses_a_folder_outside_the_home_or_with_a_set(cli, tmp_path, capsys):
    assert main(["--config", cli, "replay", "build", "--out", str(tmp_path / "elsewhere")]) == 1
    assert "inside" in capsys.readouterr().err
    out = tmp_path / "home" / "replay" / "set-v1"
    assert main(["--config", cli, "replay", "build", "--out", str(out)]) == 0
    assert main(["--config", cli, "replay", "build", "--out", str(out)]) == 1
    assert "cases.jsonl" in capsys.readouterr().err


def test_score_writes_the_result_and_prints_a_table(cli, tmp_path, capsys):
    out = tmp_path / "home" / "replay" / "set-v1"
    assert main(["--config", cli, "replay", "build", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["--config", cli, "replay", "score", "--set", str(out)]) == 0
    files = list(out.glob("score-*.json"))
    assert len(files) == 1
    result = json.loads(files[0].read_text())
    assert set(result["summary"]) == {"bare", "typed"}
    assert len(result["cases"]) == 6
    assert result["manifest"]["cases_sha256"] == json.loads((out / "manifest.json").read_text())["cases_sha256"]
    printed = capsys.readouterr().out
    assert "bare" in printed and "typed" in printed and "no_leak" in printed
