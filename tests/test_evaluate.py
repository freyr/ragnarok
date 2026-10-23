from ragnarok.evaluate import METHODS, bootstrap, inside, pool_markdown, read_pool_markdown, score_hard

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
