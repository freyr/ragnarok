import json

import pytest

from ragnarok import cli
from ragnarok.config import Config, dump_config
from ragnarok.evaluate import METHODS, bootstrap, compare, inside, pool_markdown, read_pool_markdown, score_hard

ROWS = [
    {
        "set": "draft-long",
        "id": "q01",
        "question": "What is a field stop?",
        "expected": ["A/fs.md", "A/missing.md"],
        "pool": [{"path": "A/fs.md", "methods": ["fts"]}, {"path": "A/other.md", "methods": ["dense"]}],
    },
    {
        "set": "draft-short",
        "id": "q01",
        "question": "field stop",
        "expected": ["A/fs.md", "A/missing.md"],
        "pool": [{"path": "A/fs.md", "methods": ["grep"]}],
    },
    {
        "set": "hard",
        "id": "h01",
        "question": "vague",
        "expected": ["Home/Heating"],
        "pool": [{"path": "Home/Heating/x.md", "methods": ["hybrid"]}, {"path": "Home/Bike/y.md", "methods": []}],
    },
]


def test_pool_markdown_round_trip():
    md = pool_markdown(ROWS)
    assert "> What is a field stop?\n> field stop" in md
    assert "- [x] `A/fs.md` (grep, fts)" in md
    assert "- [x] `A/missing.md` (expected only)" in md
    assert read_pool_markdown(md) == {"q01": ["A/fs.md", "A/missing.md"], "h01": ["Home/Heating/x.md"]}


def test_user_marks_are_read():
    md = (
        pool_markdown(ROWS)
        .replace("- [ ] `A/other.md`", "- [x] `A/other.md`")
        .replace("- [x] `A/missing.md`", "- [ ] `A/missing.md`")
    )
    assert read_pool_markdown(md)["q01"] == ["A/fs.md", "A/other.md"]


def test_inside_prefix():
    assert inside("Home/Heating/x.md", "Home/Heating")
    assert not inside("Home/HeatingOld/x.md", "Home/Heating")


def test_bootstrap_interval():
    d, lo, hi = bootstrap([1, 1, 1, 0], [0, 0, 0, 0])
    assert d == 0.75 and lo <= d <= hi


def test_note_precision_uses_checked_notes():
    q = {"id": "h01", "type": "category", "expected": ["P/A"], "relevant": ["P/A/good.md"]}
    ranked = {m: ["P/A/good.md", "P/A/off.md"] for m in METHODS}
    r = score_hard(q, ranked)
    assert r["hybrid_prec@5"] == 0.4
    assert r["hybrid_nprec@5"] == 0.2


MEASURES = {"answer_first": ["answer-first", 1], "wrong": ["wrong@5", -1]}


def row(qid, cls, answer_first, wrong):
    return {"id": qid, "set": cls, "class": cls, "answer_first": answer_first, "wrong": wrong}


# the first file: system C; the second file: system A, in another order, with one question of its own
FIRST = {
    "runner": "mechanical",
    "measures": MEASURES,
    "systems": {
        "kb": [row("a1", "now", 1.0, 0.0), row("a2", "now", 0.5, 0.0), row("a3", "now", 1.0, 0.5)]
        + [row("a4", "future", 0.0, 0.0)]
    },
}
SECOND = {
    "runner": "mechanical",
    "measures": MEASURES,
    "systems": {
        "base": [row("a3", "now", 0.0, 1.0), row("a5", "now", 1.0, 0.0), row("a1", "now", 0.0, 1.0)]
        + [row("a2", "now", 0.5, 0.0), row("a4", "future", 1.0, 0.0)]
    },
}


def test_compare_pairs_by_question_id_and_gives_the_in_run_bootstrap():
    lines = compare(FIRST, SECOND)
    # now, in the order of the first file (a1, a2, a3): answer-first C [1, 0.5, 1], A [0, 0.5, 0]: C - A [1, 0, 1]
    _, lo, hi = bootstrap([1.0, 0.5, 1.0], [0.0, 0.5, 0.0])
    assert f"  answer-first (n=3): first 0.83, second 0.17; first - second +0.67 [{lo:+.2f}, {hi:+.2f}]" in lines
    # wrong@5: A [1, 0, 1], C [0, 0, 0.5]: A - C [1, 0, 0.5], mean 0.5
    _, lo, hi = bootstrap([1.0, 0.0, 1.0], [0.0, 0.0, 0.5])
    assert f"  wrong@5 (n=3): first 0.17, second 0.67; second - first +0.50 [{lo:+.2f}, {hi:+.2f}]" in lines
    assert "## now (n=3)" in lines and "## future (n=1)" in lines
    assert "  answer-first (n=1): first 0.00, second 1.00; first - second -1.00 [-1.00, -1.00]" in lines
    assert "unpaired: a5 (second only)" in lines


def test_compare_needs_one_system_of_each_file_and_the_same_runner():
    two = {**FIRST, "systems": {**FIRST["systems"], "kb-when": FIRST["systems"]["kb"]}}
    with pytest.raises(ValueError, match="kb, kb-when"):
        compare(two, SECOND)
    assert compare(two, SECOND, first_system="kb-when") == compare(FIRST, SECOND)
    with pytest.raises(ValueError, match="runner"):
        compare(FIRST, {**SECOND, "runner": "agent"})


def test_evaluate_compare_command(tmp_path, capsys):
    cfg = Config(root=tmp_path / "vaults", data_dir=tmp_path / "data", vaults=["Kb"])
    config = tmp_path / "config.toml"
    config.write_text(dump_config(cfg, root=str(tmp_path / "vaults"), data_dir=str(tmp_path / "data")))
    first, second = tmp_path / "c.json", tmp_path / "a.json"
    first.write_text(json.dumps(FIRST))
    second.write_text(json.dumps(SECOND))
    assert cli.main(["--config", str(config), "evaluate", "compare", str(first), str(second)]) == 0
    out = capsys.readouterr().out
    _, lo, hi = bootstrap([1.0, 0.5, 1.0], [0.0, 0.5, 0.0])
    assert f"first - second +0.67 [{lo:+.2f}, {hi:+.2f}]" in out
    assert f"# compare, first {first} (kb), second {second} (base)" in out
    assert cli.main(["--config", str(config), "evaluate", "compare", str(first)]) == 1
    assert "two per-question files" in capsys.readouterr().err
