"""The set format v2, `bench run` and `bench compare`."""

import json
import os

import numpy as np
import pytest

from ragnarok.bench import (
    cluster_bootstrap,
    convert,
    holm,
    load_set,
    measures,
    set_hash,
    write_set,
)
from ragnarok.config import Config

# --- measures ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranks", "recall", "rr"),
    [
        ([1], 1, 1.0),
        ([3], 1, 1 / 3),
        ([5], 1, 1 / 5),
        ([7], 0, 1 / 7),
        ([None], 0, 0.0),
        # multi-gold: recall@5 needs all expected notes (up to 5) in the first 5, MRR takes the first expected note
        ([2, 6], 0, 1 / 2),
        ([4, 1], 1, 1.0),
        ([None, 9], 0, 1 / 9),
        ([None, None], 0, 0.0),
        ([5, 3], 1, 1 / 3),
        # more than 5 expected notes: recall@5 needs 5 of them in the first 5
        ([1, 2, 3, 4, 5, 9, None], 1, 1.0),
        ([2, 1, 5, 3, 4, None], 1, 1.0),
        ([1, 2, 3, 4, 6, 7], 0, 1.0),
        ([1, 2, 3, 4, None, None], 0, 1.0),
    ],
)
def test_measures_of_a_single_gold_and_a_multi_gold_question(ranks, recall, rr):
    assert measures(ranks) == (recall, pytest.approx(rr))


# --- cluster bootstrap and Holm ------------------------------------------------------------------------------------


def test_cluster_bootstrap_resamples_clusters_not_questions():
    # cluster X: 9 questions with d = 1; cluster Y: 1 question with d = 0. Two clusters drawn with replacement give
    # XX (mean 1.0, p 1/4), XY (mean 0.9, p 1/2) or YY (mean 0.0, p 1/4), so the 95% interval is [0, 1].
    # A bootstrap of questions would give a lower end near 0.7.
    d = [1.0] * 9 + [0.0]
    clusters = ["X"] * 9 + ["Y"]
    mean, lo, hi, p = cluster_bootstrap(d, clusters)
    assert mean == pytest.approx(0.9)
    assert (lo, hi) == (0.0, 1.0)
    # 1/4 of the samples are YY with a mean of 0, so P(d* <= 0) is about 0.25 and p = 2 * 0.25
    assert p == pytest.approx(0.5, abs=0.03)


def test_cluster_bootstrap_is_repeatable_and_a_constant_difference_has_no_spread():
    rng = np.random.default_rng(7)
    d = list(rng.choice([-1.0, 0.0, 1.0], 40))
    clusters = [f"c{i % 13}" for i in range(40)]
    assert cluster_bootstrap(d, clusters) == cluster_bootstrap(d, clusters)
    assert cluster_bootstrap([0.5] * 6, list("aabbcc"))[:3] == (0.5, 0.5, 0.5)
    mean, lo, hi, p = cluster_bootstrap([0.0] * 6, list("aabbcc"))
    assert (mean, lo, hi, p) == (0.0, 0.0, 0.0, 1.0)


def test_cluster_bootstrap_level_widens_the_interval():
    rng = np.random.default_rng(3)
    d = list(rng.normal(0.1, 1.0, 60))
    clusters = [f"c{i}" for i in range(60)]
    _, lo95, hi95, _ = cluster_bootstrap(d, clusters, alpha=0.05)
    _, lo99, hi99, _ = cluster_bootstrap(d, clusters, alpha=0.01)
    assert lo99 < lo95
    assert hi99 > hi95


def test_holm_levels_of_two_variants():
    # sorted: 0.01 gets 0.05 / 2 = 0.025 (rejected), 0.04 gets 0.05 / 1 (rejected)
    levels, rejected = holm([0.04, 0.01])
    assert levels == pytest.approx([0.05, 0.025])
    assert rejected == [True, True]


def test_holm_levels_of_three_variants_stop_at_the_first_kept_hypothesis():
    # sorted: 0.01 -> 0.05/3 = 0.0167 rejected; 0.03 -> 0.05/2 = 0.025 kept, so 0.04 -> 0.05 is also kept
    levels, rejected = holm([0.01, 0.04, 0.03])
    assert levels == pytest.approx([0.05 / 3, 0.05, 0.025])
    assert rejected == [True, False, False]


def test_holm_of_one_variant_is_the_plain_level():
    assert holm([0.03]) == ([0.05], [True])
    assert holm([0.2]) == ([0.05], [False])


# --- the set format ------------------------------------------------------------------------------------------------

QUESTIONS = [
    {
        "id": "q1",
        "question": "boiler flow",
        "expected": ["V/a.md"],
        "stratum": "lookup",
        "area": "v",
        "cluster": "V/a.md",
    },
    {
        "id": "q2",
        "question": "pump speed",
        "expected": ["V/b.md", "V/a.md"],
        "stratum": "fact",
        "area": "v",
        "cluster": "V/b.md",
        "notes": "two notes",
    },
]


def test_a_set_keeps_the_header_and_the_hash_of_the_questions(tmp_path):
    path = tmp_path / "set.json"
    write_set(path, "pilot", "real", "real", QUESTIONS)
    data = json.loads(path.read_text())
    assert (data["name"], data["kind"], data["version"], data["corpus"]) == ("pilot", "real", 2, "real")
    canonical = json.dumps(QUESTIONS, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    import hashlib

    assert data["hash"] == hashlib.sha256(canonical).hexdigest() == set_hash(QUESTIONS)
    assert load_set(path)["questions"] == QUESTIONS


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["questions"][0].update(question="changed"),  # the hash does not agree
        lambda d: d.update(kind="other"),
        lambda d: d.update(version=1),
        lambda d: d["questions"][0].update(expected=[]),
        lambda d: d["questions"][0].pop("cluster"),
        lambda d: d["questions"][1].update(cluster="V/c.md"),  # the cluster is not an expected path
    ],
)
def test_a_wrong_set_is_refused(tmp_path, change):
    path = tmp_path / "set.json"
    write_set(path, "pilot", "real", "real", QUESTIONS)
    data = json.loads(path.read_text())
    change(data)
    if "hash" in data and data["questions"][0].get("question") != "changed":
        data["hash"] = set_hash(data["questions"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        load_set(path)


def test_convert_the_draft_format(tmp_path):
    (tmp_path / "V/Heating").mkdir(parents=True)
    (tmp_path / "V/Heating/boiler.md").write_text("# Boiler\n")
    cfg = Config(root=tmp_path, vaults=["V"], index_path=tmp_path / "i.db")
    old = [
        {
            "id": "q01",
            "question": "boiler?",
            "type": "lookup",
            "expected": ["V/Heating/boiler.md", "V/x.md"],
            "note": "n",
        },
        {"id": "q02", "question": "absent?", "type": "fact", "expected": ["V/Gone/x.md"]},
        {"id": "q03", "question": "nothing", "type": "negative", "expected": []},
        {"id": "h01", "question": "resume", "type": "continuation", "subtype": "resume", "expected": ["V/Heating"]},
    ]
    questions, skipped = convert(old, cfg)
    assert questions == [
        {
            "id": "q01",
            "question": "boiler?",
            "expected": ["V/Heating/boiler.md", "V/x.md"],
            "stratum": "lookup",
            "area": "heating",
            "cluster": "V/Heating/boiler.md",
            "notes": "n",
        },
        {
            "id": "q02",
            "question": "absent?",
            "expected": ["V/Gone/x.md"],
            "stratum": "fact",
            "area": "unknown",
            "cluster": "V/Gone/x.md",
        },
    ]
    assert [qid for qid, _ in skipped] == ["q03", "h01"]


def gold_vault(tmp_path):
    for rel in ("V/Heating/boiler.md", "V/Heating/pump.md", "V/Astro/mount.md", "V/Astro/New/eyepiece.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("# Note\n")
    return Config(root=tmp_path, vaults=["V"], index_path=tmp_path / "i.db")


def test_convert_the_health_format(tmp_path):
    cfg = gold_vault(tmp_path)
    old = [
        {
            "id": "st-01",
            "class": "now",
            "when": "now",
            "question": "boiler now?",
            "question_alt": "the boiler today?",
            "answer": "a",
            "gold": [
                {"path": "V/Heating/boiler.md", "grade": 1},
                {"path": "V/Heating/old.md", "grade": 0},
                {"path": "V/Heating/pump.md"},
            ],
            "wrong": ["V/Heating/old.md"],
        },
        {"id": "fu-01", "class": "future", "when": "future", "question": "plan?", "gold": [], "wrong": []},
    ]
    questions, skipped = convert(old, cfg)
    assert questions == [
        {
            "id": "st-01",
            "question": "boiler now?",
            "question_alt": "the boiler today?",
            "answer": "a",
            "wrong": ["V/Heating/old.md"],
            "expected": ["V/Heating/boiler.md", "V/Heating/pump.md"],
            "stratum": "now",
            "area": "heating",
            "cluster": "V/Heating/boiler.md",
        }
    ]
    assert skipped == [("fu-01", "no expected path")]


def test_convert_the_kb_format_drops_the_kb_keys(tmp_path):
    cfg = gold_vault(tmp_path)
    old = [
        {
            "id": "st-02",
            "class": "single-truth",
            "question": "mount load?",
            "question_alt": "how much does the mount carry?",
            "gold": [{"path": "V/Astro/mount.md", "grade": 1}],
            "gold_eligibility": {"type": ["entity"]},
            "gold_route": "kb",
            "gold_entities": ["mount"],
            "gold_claims": ["20 kg"],
            "distractors": [],
            "origin": "builder",
        }
    ]
    questions, skipped = convert(old, cfg)
    assert questions == [
        {
            "id": "st-02",
            "question": "mount load?",
            "question_alt": "how much does the mount carry?",
            "gold_entities": ["mount"],
            "distractors": [],
            "origin": "builder",
            "expected": ["V/Astro/mount.md"],
            "stratum": "single-truth",
            "area": "astro",
            "cluster": "V/Astro/mount.md",
        }
    ]
    assert skipped == []


def test_convert_applies_the_path_map_and_takes_the_cluster_from_an_existing_note(tmp_path):
    cfg = gold_vault(tmp_path)
    path_map = {"V/Astro/Old/eyepiece.md": "V/Astro/New/eyepiece.md", "V/Astro/mount-copy.md": "V/Astro/mount.md"}
    old = [
        {"id": "q01", "question": "eyepiece?", "type": "lookup", "expected": ["V/Astro/Old/eyepiece.md"]},
        {
            "id": "hi-01",
            "class": "history",
            "question": "old mount?",
            "gold": [
                {"path": "V/Astro/gone.md", "grade": 1},
                {"path": "V/Astro/mount-copy.md", "grade": 1},
                {"path": "V/Astro/mount.md", "grade": 1},
            ],
        },
    ]
    questions, _ = convert(old, cfg, path_map=path_map)
    assert [(q["expected"], q["cluster"], q["area"]) for q in questions] == [
        (["V/Astro/New/eyepiece.md"], "V/Astro/New/eyepiece.md", "astro"),
        # the gone path stays, so the gold check reports it; the map makes a duplicate, which is removed
        (["V/Astro/gone.md", "V/Astro/mount.md"], "V/Astro/mount.md", "astro"),
    ]


def test_convert_keeps_one_question_of_each_text_and_makes_the_ids_unique(tmp_path):
    cfg = gold_vault(tmp_path)
    gold = [{"path": "V/Astro/mount.md", "grade": 1}]
    old = [
        {"id": "st-20", "class": "now", "question": "Mount load?", "gold": gold},
        {"id": "st-20", "class": "history", "question": "mount load before?", "gold": gold},
        {"id": "q05", "question": " mount LOAD? ", "type": "fact", "expected": ["V/Astro/mount.md"]},
    ]
    questions, skipped = convert(old, cfg)
    assert [q["id"] for q in questions] == ["st-20", "history-st-20"]
    assert skipped == [("q05", "the same text as st-20")]


def test_convert_keeps_the_selected_areas(tmp_path):
    cfg = gold_vault(tmp_path)
    old = [
        {"id": "q01", "question": "boiler?", "type": "lookup", "expected": ["V/Heating/boiler.md"]},
        {"id": "q02", "question": "mount?", "type": "fact", "expected": ["V/Astro/mount.md"]},
        {"id": "q03", "question": "gone?", "type": "fact", "expected": ["V/Gone/x.md"]},
    ]
    questions, skipped = convert(old, cfg, areas=["astro"])
    assert [q["id"] for q in questions] == ["q02"]
    assert skipped == [("q01", "the area heating is not selected"), ("q03", "the area unknown is not selected")]


def test_bench_convert_reads_several_files_with_a_path_map(tmp_path, capsys, monkeypatch):
    gold_vault(tmp_path / "vaults")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        f'root = "{tmp_path / "vaults"}"\nvaults = ["V"]\n\n[corpora.real]\nroot = "{tmp_path / "vaults"}"\n'
        'vaults = ["V"]\n'
    )
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    health = tmp_path / "health.json"
    health.write_text(
        json.dumps([{"id": "st-01", "class": "now", "question": "mount?", "gold": [{"path": "V/Astro/mount.md"}]}])
    )
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            [
                {"id": "q01", "question": "eyepiece?", "type": "lookup", "expected": ["V/Astro/Old/eyepiece.md"]},
                {"id": "q02", "question": "boiler?", "type": "lookup", "expected": ["V/Heating/boiler.md"]},
            ]
        )
    )
    path_map = tmp_path / "map.tsv"
    path_map.write_text("old\tnew\tsource\nV/Astro/Old/eyepiece.md\tV/Astro/New/eyepiece.md\tcleanup move\n")
    out = tmp_path / "real.json"
    code, printed = run(
        capsys, "bench", "convert", health, draft, "--out", out, "--name", "real-v1", "--kind", "real",
        "--corpus", "real", "--path-map", path_map, "--area", "astro",
    )  # fmt: skip
    assert code == 0, printed.err
    assert "not converted q02: the area heating is not selected" in printed.err
    data = load_set(out)
    assert (data["name"], data["kind"], data["corpus"]) == ("real-v1", "real", "real")
    assert [(q["id"], q["expected"]) for q in data["questions"]] == [
        ("st-01", ["V/Astro/mount.md"]),
        ("q01", ["V/Astro/New/eyepiece.md"]),
    ]


# --- bench run and bench compare on a fixture vault ----------------------------------------------------------------


class FixedEmbedder:
    """The same vector for each text, so that BM25 gives the order."""

    def embed_docs(self, items):
        return [[1.0, 0.0, 0.0] for _ in items]

    def embed_query(self, query):
        return [1.0, 0.0, 0.0]


NOTES = {
    "V/Heating/boiler.md": "# Boiler\nThe boiler flow temperature is 55 degrees.\n",
    "V/Heating/pump.md": "# Pump\nThe pump speed is level two.\n",
    "W/Astro/eyepiece.md": "# Eyepiece\nThe field stop of the eyepiece is 27 mm.\n",
    "W/Astro/mount.md": "# Mount\nThe mount carries 20 kg.\n",
}

BENCH = (
    '\n[variants.baseline]\n\n[variants.other]\nmin_words = 2\n\n[corpora.real]\nroot = "vaults"\nvaults = ["V", "W"]\n'
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for rel, text in NOTES.items():
        path = home / "vaults" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (home / "config.toml").write_text(
        f'root = "vaults"\nvaults = ["V", "W"]\ndata_dir = "data"\ndimension = 3\nmin_words = 1\n{BENCH}'
    )
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda cfg: FixedEmbedder())
    return home


def run(capsys, *argv):
    from ragnarok.cli import main

    code = main([str(a) for a in argv])
    return code, capsys.readouterr()


def fixture_questions():
    q = [
        ("q1", "boiler flow temperature", ["V/Heating/boiler.md"], "lookup"),
        ("q2", "pump speed level", ["V/Heating/pump.md"], "fact"),
        ("q3", "eyepiece field stop", ["W/Astro/eyepiece.md"], "lookup"),
        ("q4", "mount carries kg", ["W/Astro/mount.md", "W/Astro/eyepiece.md"], "fact"),
    ]
    return [
        {
            "id": i,
            "question": text,
            "expected": exp,
            "stratum": s,
            "area": exp[0].split("/")[1].lower(),
            "cluster": exp[0],
        }
        for i, text, exp, s in q
    ]


def test_bench_run_refuses_a_set_with_a_missing_gold_path(home, capsys):
    questions = fixture_questions()
    questions[1]["expected"] = ["V/Heating/pump.md", "V/Heating/gone.md"]
    questions[3]["expected"] = ["W/Astro/mount.md"]
    questions[3]["cluster"] = "W/Astro/mount.md"
    path = home / "set.json"
    write_set(path, "pilot", "real", "real", questions)
    config = home / "config.toml"
    config.write_text('private_patterns = ["W/Astro/mount.md"]\n' + config.read_text())
    assert run(capsys, "bench", "index", "--variant", "baseline", "--corpus", "real")[0] == 0
    code, out = run(capsys, "bench", "run", "--set", path, "--variant", "baseline")
    assert code == 1
    assert "q2: V/Heating/gone.md (not in the index)" in out.err
    assert "q4: W/Astro/mount.md (private)" in out.err
    assert "2 expected paths cannot be found" in out.err
    assert not (home / "data/bench").exists()


def test_bench_run_brings_an_index_far_behind_current_before_the_gold_check(home, capsys):
    from ragnarok.index import SCAN_LIMIT

    path = home / "set.json"
    questions = fixture_questions()
    questions[0]["expected"] = ["V/New/n00.md"]
    questions[0]["cluster"] = "V/New/n00.md"
    questions[0]["question"] = "note zero"
    write_set(path, "pilot", "real", "real", questions)
    assert run(capsys, "bench", "index", "--variant", "baseline", "--corpus", "real")[0] == 0
    # more new notes than one change scan takes; the gold note is the oldest, so a scan with the limit defers it
    for i in range(SCAN_LIMIT + 5):
        note = home / "vaults/V/New" / f"n{i:02d}.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(f"# N{i}\nnote {'zero' if i == 0 else i}\n")
        os.utime(note, (1_000_000 + i, 1_000_000 + i))
    code, out = run(capsys, "bench", "run", "--set", path, "--variant", "baseline")
    assert code == 0, out.err
    result = json.loads(next((home / "data/bench").glob("*.json")).read_text())
    # the index was current before the first question, so no timed scan changed a note
    assert [r["scan_changed"] for r in result["questions"]] == [0, 0, 0, 0]
    assert result["questions"][0]["ranks"][0] is not None


def test_bench_run_writes_the_ranks_and_the_times(home, capsys):
    path = home / "set.json"
    write_set(path, "pilot", "real", "real", fixture_questions())
    assert run(capsys, "bench", "index", "--variant", "baseline", "--corpus", "real")[0] == 0
    code, out = run(capsys, "bench", "run", "--set", path, "--variant", "baseline")
    assert code == 0, out.err
    from ragnarok.config import load_config, variant_hash

    cfg = load_config()
    name = f"{set_hash(fixture_questions())[:12]}-{variant_hash(cfg.variant('baseline', 'real'))}.json"
    result = json.loads((home / "data/bench" / name).read_text())
    assert (result["variant"], result["corpus"], result["set_hash"]) == (
        "baseline",
        "real",
        set_hash(fixture_questions()),
    )
    rows = {r["id"]: r for r in result["questions"]}
    assert rows["q1"]["ranks"] == [1]
    assert rows["q3"]["ranks"][0] == 1
    assert len(rows["q4"]["ranks"]) == 2
    for r in rows.values():
        assert {"scan_ms", "search_ms", "total_ms", "overlap", "bm25_rank"} <= set(r)
        assert r["total_ms"] >= r["search_ms"] >= 0
        assert r["scan_ms"] >= 0
    assert rows["q1"]["overlap"] == 1.0
    assert rows["q1"]["bm25_rank"] == 1
    # the other variant writes its own file
    assert run(capsys, "bench", "index", "--variant", "other", "--corpus", "real")[0] == 0
    assert run(capsys, "bench", "run", "--set", path, "--variant", "other")[0] == 0
    assert len(list((home / "data/bench").glob("*.json"))) == 2


@pytest.mark.parametrize("kind", ["synthetic", "real"])
def test_bench_compare_of_the_baseline_against_itself_is_no_win(home, capsys, kind):
    path = home / "set.json"
    write_set(path, "pilot", kind, "real", fixture_questions())
    assert run(capsys, "bench", "index", "--variant", "baseline", "--corpus", "real")[0] == 0
    assert run(capsys, "bench", "run", "--set", path, "--variant", "baseline")[0] == 0
    code, out = run(capsys, "bench", "compare", "--set", path, "baseline", "baseline")
    assert code == 0, out.err
    text = out.out
    assert "recall@5 B-A +0.000 [+0.000, +0.000]" in text
    assert "MRR B-A +0.000 [+0.000, +0.000]" in text
    assert "discordant 0.00" in text
    assert "stratum lookup" in text
    assert "area heating" in text
    last = text.strip().splitlines()[-1]
    assert last.startswith("verdict baseline: no win")
    if kind == "synthetic":
        assert "vault V: recall@5 0 MRR 0" in text
    else:
        assert "no regression: pass" in last


def test_bench_compare_needs_a_run_of_each_variant(home, capsys):
    path = home / "set.json"
    write_set(path, "pilot", "real", "real", fixture_questions())
    code, out = run(capsys, "bench", "compare", "--set", path, "baseline", "other")
    assert code == 1
    assert "bench run" in out.err
