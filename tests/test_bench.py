"""`ragnarok bench`: the index state check, the B cache, the report with the deciders and the diagnostics, on a
synthetic home with a fake baseline and a fake new-code search."""

import json
from dataclasses import replace

import pytest
from test_kb_eval import GOLD, NOW, NOW_EMPTY, OTHER, STALE, WRONG, hits
from test_search import TopicEmbedder as FakeEmbedder

from ragnarok import agent_eval, bench, cli, kb_eval
from ragnarok.config import Config, dump_config
from ragnarok.index import connect, update

NOTES = {
    GOLD: "# Deploy\nDeploy with the pipeline.\n",
    WRONG: "---\nclass: proposal\nvalidity: rejected\n---\n# Deploy by hand\nA rejected way.\n",
    OTHER: "---\nclass: proposal\nvalidity: open\n---\n# Other\nSomething else.\n",
    STALE: "# Deploy 2024\nThe old way.\n",
}
# A, the frozen file of the before home: nw-01 answer-first 1 (C loses against it), wrong 1 for both
A_ROWS = [
    {"id": "nw-01", "set": "now", "class": "now", "answer_first": 1.0, "wrong": 1.0},
    {"id": "nw-02", "set": "now", "class": "now", "answer_first": 0.0, "wrong": 1.0},
]
DECIDERS = """
[[decider]]
set = "now"
measure = "answer_first"
first = "C"
second = "A"
lower = 0.0
[[decider]]
set = "now"
measure = "wrong"
first = "C"
second = "A"
lower = 0.0
[[decider]]
set = "now"
measure = "wrong"
first = "C"
second = "B"
lower = 0.0
[[decider]]
kind = "filtered"
max_regressed = 0
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home with an evaluation home (mirror, index), the sets, the regression set, the A file and bench.toml."""
    cfg = Config(
        root=tmp_path / "vaults",
        data_dir=tmp_path / "data",
        vaults=["Kb"],
        dimension=8,
        baseline_checkout=str(tmp_path / "bc"),
        baseline_home=str(tmp_path / "bh"),
    )
    (tmp_path / "config.toml").write_text(
        dump_config(cfg, root=str(tmp_path / "vaults"), data_dir=str(tmp_path / "data"))
    )
    ehome = agent_eval.eval_home(cfg)
    for rel, text in NOTES.items():
        (ehome / "vaults" / rel).parent.mkdir(parents=True, exist_ok=True)
        (ehome / "vaults" / rel).write_text(text)
    (ehome / "config.toml").write_text(dump_config(replace(cfg, vault_paths={}), root="vaults", data_dir="data"))
    ecfg = agent_eval.eval_config(cfg)
    conn = connect(ecfg)
    update(ecfg, conn, FakeEmbedder())
    conn.close()
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "now.json").write_text(json.dumps([NOW, NOW_EMPTY]))
    (sets / "join.json").write_text(json.dumps([{"id": "jn-01", "class": "join", "question": "Who?", "gold": []}]))
    (tmp_path / "regression").mkdir()
    (tmp_path / "regression" / "draft-long.json").write_text("[]")
    a_file = tmp_path / "per-question-A.json"
    kb_eval.write_per_question(a_file, "mechanical", kb_eval.MECHANICAL, {"base": A_ROWS})
    (tmp_path / "bench.toml").write_text(
        f'sets = "sets"\nregression_sets = "regression"\na_file = "{a_file}"\n' + DECIDERS
    )
    stamps = iter(f"20261106-1200{i:02d}" for i in range(60))
    monkeypatch.setattr(bench, "_stamp", lambda: next(stamps))
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda cfg: FakeEmbedder())
    return tmp_path


class Fakes:
    """The fake baseline (base: [WRONG, GOLD] for each search) and the fake new-code search (kb: form 1
    [GOLD, WRONG], form 2 `second`; kb-when: [GOLD]). `base_searches` counts the searches of the question forms."""

    def __init__(self, monkeypatch):
        self.base_searches = 0
        self.second = [OTHER, STALE]
        forms = {NOW["question"], NOW["question_alt"]}

        def base(batch):
            self.base_searches += sum(t == "search" and a["query"] in forms for t, a in batch)
            return [{"type": "note"} if t == "open_note" else {"results": [{"path": WRONG}, {"path": GOLD}]}
                    for t, a in batch]  # fmt: skip

        def search(cfg, conn, embedder, query, limit=None, **kw):
            if kw.get("when") == "now":
                return hits(GOLD)
            return hits(GOLD, WRONG) if query == NOW["question"] else hits(*self.second)

        monkeypatch.setattr(kb_eval, "mcp_caller", lambda checkout, home: base)
        monkeypatch.setattr(kb_eval, "search", search)


def bench_cli(home, *args):
    return cli.main(["--config", str(home / "config.toml"), "bench", *args])


def runs(home):
    return sorted((home / "bench" / "runs").iterdir())


def test_bench_run_stops_when_the_fixture_is_not_the_indexed_one(home, monkeypatch, capsys):
    Fakes(monkeypatch)
    assert bench_cli(home, "run") == 1
    assert capsys.readouterr().err.strip() == "the fixture is not the indexed one: run `ragnarok bench index`"
    assert bench_cli(home, "index", "--record-only") == 0
    fixture = agent_eval.mirror_hash(home / "data" / "eval" / "vaults")
    assert capsys.readouterr().out.strip() == fixture
    assert json.loads((home / "bench" / "index-state.json").read_text()) == {"fixture_hash": fixture}
    (home / "data" / "eval" / "vaults" / "Kb" / "new.md").write_text("# New\n")
    assert bench_cli(home, "run") == 1
    assert "run `ragnarok bench index`" in capsys.readouterr().err


def test_bench_index_rebuilds_both_indexes_and_records_the_hash(home, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bench.subprocess, "run", lambda cmd, env, check: calls.append((cmd, env["RAGNAROK_HOME"])))
    assert bench_cli(home, "index") == 0
    assert calls == [(["uv", "--directory", str(home / "bc"), "run", "ragnarok", "index"], str(home / "bh"))]
    conn = connect(agent_eval.eval_config(Config(data_dir=home / "data")))
    assert {p for (p,) in conn.execute("SELECT path FROM doc")} == set(NOTES)
    fixture = agent_eval.mirror_hash(home / "data" / "eval" / "vaults")
    assert json.loads((home / "bench" / "index-state.json").read_text()) == {"fixture_hash": fixture}


def test_bench_run_reports_values_gains_deciders_and_power(home, monkeypatch, capsys):
    fakes = Fakes(monkeypatch)
    bench_cli(home, "index", "--record-only")
    capsys.readouterr()
    assert bench_cli(home, "run", "--label", "first") == 0
    out = capsys.readouterr().out
    (run,) = runs(home)
    assert run.name == "20261106-120000-first"
    assert (run / "report.md").read_text().strip() == out.strip()
    # A: af [1, 0], wrong [1, 1]; B (base, [WRONG, GOLD]): af [0, 0], wrong [1, 0];
    # C (kb): nw-01 af 0.5 wrong 0.5, nw-02 af 0 wrong 0.5; Cwhen: af [1, 0], wrong [0, 0]
    assert "| now | answer-first | 2 | 0.50 | 0.00 | 0.25 | 0.50 | n/a |" in out
    assert "| now | wrong@5 | 2 | 1.00 | 0.50 | 0.50 | 0.00 | n/a |" in out
    assert "join" not in out.split("## Values")[1].split("## Deciders")[0]  # n=0: no row
    # gains, paired by id: C-A af [-0.5, 0]; wrong is A - C = [0.5, 0.5]; C-B wrong is B - C = [0.5, -0.5]
    assert "| now | answer-first | -0.25 [-0.50, +0.00] | +0.25 [+0.00, +0.50] | n/a | +0.00 [+0.00, +0.00] |" in out
    assert "| now | wrong@5 | +0.50 [+0.50, +0.50] | +0.00 [-0.50, +0.50] | n/a | +1.00 [+1.00, +1.00] |" in out
    # the deciders: the sign of wrong@5 makes A - C the gain of C over A
    assert "| now answer-first C-A | -0.25 [-0.50, +0.00] | > +0.00 | fail |" in out
    assert "| now wrong@5 C-A | +0.50 [+0.50, +0.50] | > +0.00 | pass |" in out
    assert "| now wrong@5 C-B | +0.00 [-0.50, +0.50] | > +0.00 | fail |" in out
    assert "| filtered regressed | 0 | <= 0 | pass |" in out
    assert "**Verdict: no-go**" in out
    # the power line: 2.8 * sd([-0.5, 0], ddof 1) / sqrt(2) = 0.70; sd 0 gives n/a
    assert "- now answer-first C-A: smallest detectable difference 0.70 (n=2)" in out
    assert "- now wrong@5 C-A: smallest detectable difference n/a (n=2)" in out
    data = json.loads((run / "per-question.json").read_text())
    assert data["runner"] == "mechanical" and list(data["systems"]) == ["base", "kb", "kb-when"]
    assert data["fixture_hash"] == agent_eval.mirror_hash(home / "data" / "eval" / "vaults")
    assert set(data["set_hashes"]) == {"now.json", "join.json", "draft-long.json"}
    assert data["label"] == "first" and data["commit"]
    assert fakes.base_searches == 4  # 2 questions, 2 forms


def test_bench_run_uses_the_b_cache_and_compares_with_the_last_run(home, monkeypatch, capsys):
    fakes = Fakes(monkeypatch)
    bench_cli(home, "index", "--record-only")
    bench_cli(home, "run", "--label", "first")
    fakes.second = [GOLD]  # kb form 2: nw-01 af 0.5 -> 1.0, nw-02 wrong 0.5 -> 0.0
    capsys.readouterr()
    assert bench_cli(home, "run", "--label", "second") == 0
    out = capsys.readouterr().out
    assert fakes.base_searches == 4  # the second run took B from the cache
    assert "| now | answer-first | 2 | 0.50 | 0.00 | 0.50 | 0.50 | 0.25 |" in out
    assert "| now | answer-first | +0.00 [+0.00, +0.00] | +0.50 [+0.00, +1.00] | +0.25 [+0.00, +0.50] |" in out
    assert "- nw-01 answer-first 0.50 → 1.00" in out
    assert "- nw-02 wrong@5 0.50 → 0.00" in out
    assert "differs from the last run" not in out
    # a changed set file: the report says so on top, and B runs again
    sets = home / "sets" / "now.json"
    sets.write_text(sets.read_text() + "\n")
    capsys.readouterr()
    assert bench_cli(home, "run", "--label", "third") == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "**The set hashes differ from the last run (now.json).**"
    assert fakes.base_searches == 8


def test_bench_run_writes_the_diagnostics_of_each_query_form(home, monkeypatch, capsys):
    Fakes(monkeypatch)
    bench_cli(home, "index", "--record-only")
    capsys.readouterr()
    assert bench_cli(home, "run") == 0
    out = capsys.readouterr().out
    (run,) = runs(home)
    rows = json.loads((run / "diagnostics.json").read_text())
    first = [r for r in rows if r["id"] == "nw-01"]
    assert [r["form"] for r in first] == [NOW["question"], NOW["question_alt"]]
    assert first[0]["kind"] == "now" and first[0]["word"] is None
    assert first[0]["plain"] == {"path": GOLD, "type": "note", "class": "none", "validity": None, "gold_rank": 1}
    assert first[1]["plain"] == {
        "path": OTHER,
        "type": "note",
        "class": "proposal",
        "validity": "open",
        "gold_rank": None,
    }
    assert first[1]["when"]["path"] == GOLD and first[1]["when"]["gold_rank"] == 1
    # the summaries: the forms of each rule kind, and the non-gold rank-1 notes by type/class/validity
    assert "| now | now 4 |" in out
    assert "| note/proposal/open | 2 | 0 |" in out
    # nw-01 loses against A on answer-first: its non-gold rank-1 row
    assert f"| nw-01 | {NOW['question_alt']} | now | {OTHER} | note/proposal/open | - | {GOLD} | 1 |" in out
