import json

import pytest
from test_index import FakeEmbedder

from ragnarok import validate
from ragnarok.cli import main
from ragnarok.config import AcceptRule, Config, HealthConfig, dump_config, load_config
from ragnarok.index import connect, update

ENTITY = "V/Entities/Heat pump.md"
GUIDE = "V/Heating/guide.md"
BOILER = "V/Heating/boiler.md"
LOG = "V/Heating/log.md"
TOP = "V/top.md"
BAD = "V/Fixtures/bad.md"
LINK = "V/Fixtures/link.md"

NOTES = {
    ENTITY: "---\ntype: entity\nkind: device\n---\n# Heat pump\n",
    GUIDE: "---\nclass: state\nvalidity: current\nas_of: 2026-01-05\nsubject: ['[[Heat pump]]']\n---\n# Guide\n",
    BOILER: "---\ntype: guide\n---\n# Boiler\n",  # type guide gives a default class in the index, not here
    LOG: "---\nclass: record\nsubject: ['[[Heat pump]]']\n---\n# Log\nSee [[nowhere]].\n",
    TOP: "---\nclass: proposal\n---\n# Top\n",
    BAD: "---\nclass: [state\n---\n# Bad\n",
    LINK: "---\nclass: record\nas_of: 2026-02-01\nsubject: ['[[Heat pump]]']\n---\n# Link\nSee [[gone]].\n",
}

HEALTH_TOML = """
[health]
accept = [
  { glob = "V/Fixtures/**", kinds = ["broken-link", "bad-yaml"], reason = "fixtures" },
]
[health.targets]
structural_errors = 0
coverage = 0.5
correctness = 0.9
"""


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / "vaults"
    for rel, text in NOTES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Config(root=root, index_path=tmp_path / "index.db", vaults=["V"], dimension=8)


def indexed(cfg):
    conn = connect(cfg)
    update(cfg, conn, FakeEmbedder())
    conn.close()


def config_file(cfg, tmp_path, health=HEALTH_TOML):
    path = tmp_path / "config.toml"
    path.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path) + health)
    return path


def run(cfg, tmp_path, capsys, *args, health=HEALTH_TOML):
    code = main(["--config", str(config_file(cfg, tmp_path, health)), "health", *args])
    return code, capsys.readouterr()


def cov(have, need):
    return {"have": have, "need": need, "share": round(have / need, 4) if need else None}


def test_health_config_is_read_from_toml(cfg, tmp_path):
    loaded = load_config(config_file(cfg, tmp_path))
    assert loaded.health == HealthConfig(
        accept=[AcceptRule(glob="V/Fixtures/**", kinds=["broken-link", "bad-yaml"], reason="fixtures")],
        targets={"structural_errors": 0, "coverage": 0.5, "correctness": 0.9},
    )
    assert Config().health == HealthConfig()


def test_health_gives_errors_and_coverage_for_each_area_and_the_totals(cfg, tmp_path, capsys):
    indexed(cfg)
    code, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json")
    assert code == 0  # the gate fails, the run completes
    data = json.loads(out.out)
    areas = {a["area"]: a for a in data["areas"]}
    assert sorted(areas) == ["(top)", "Entities", "Fixtures", "Heating"]
    heating = areas["Heating"]
    assert heating["notes"] == 3
    assert heating["structural_errors"] == {"errors": 1, "accepted": 0, "not_accepted": 1}
    assert heating["coverage"] == {
        "class": cov(2, 3),  # the boiler has no explicit class
        "validity": cov(1, 1),
        "as_of": cov(0, 1),
        "subject": cov(2, 3),
    }
    fixtures = areas["Fixtures"]
    assert fixtures["notes"] == 2
    assert fixtures["structural_errors"] == {"errors": 2, "accepted": 2, "not_accepted": 0}
    # the fixture with bad YAML is out of the coverage count; the accepted link fixture counts
    assert fixtures["coverage"] == {
        "class": cov(1, 1),
        "validity": cov(0, 0),
        "as_of": cov(1, 1),
        "subject": cov(1, 1),
    }
    assert areas["Entities"]["coverage"]["subject"] == cov(0, 0)  # an entity needs no subject
    total = data["total"]
    assert total["notes"] == 7
    assert total["structural_errors"] == {"errors": 3, "accepted": 2, "not_accepted": 1}
    assert total["coverage"] == {
        "class": cov(4, 6),
        "validity": cov(1, 2),
        "as_of": cov(1, 2),
        "subject": cov(3, 5),
    }
    assert data["gate"] == {
        "structural_errors": {"value": 1, "target": 0, "result": "fail"},
        "coverage.class": {"value": 0.6667, "target": 0.5, "result": "pass"},
        "coverage.validity": {"value": 0.5, "target": 0.5, "result": "pass"},
        "coverage.as_of": {"value": 0.5, "target": 0.5, "result": "pass"},
        "coverage.subject": {"value": 0.6, "target": 0.5, "result": "pass"},
        "result": "fail",
    }
    issues = {(i["path"], i["kind"]): i for i in data["issues"]}
    assert set(issues) == {(LOG, "broken-link"), (BAD, "bad-yaml"), (LINK, "broken-link")}
    assert issues[(LINK, "broken-link")]["accepted"] is True
    assert issues[(LINK, "broken-link")]["reason"] == "fixtures"
    assert issues[(LOG, "broken-link")]["accepted"] is False


def test_the_structural_count_equals_the_errors_of_validate(cfg, tmp_path, capsys):
    indexed(cfg)
    conn = connect(cfg)
    errors = validate.check_index(cfg, conn).errors
    conn.close()
    _, out = run(cfg, tmp_path, capsys, "--json", health="")
    assert json.loads(out.out)["total"]["structural_errors"]["errors"] == errors


def test_without_targets_and_accept_list_all_errors_count_and_no_target_shows(cfg, tmp_path, capsys):
    indexed(cfg)
    code, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json", health="")
    data = json.loads(out.out)
    assert code == 0
    assert data["total"]["structural_errors"] == {"errors": 3, "accepted": 0, "not_accepted": 3}
    assert data["total"]["coverage"]["class"] == cov(4, 7)  # the bad YAML fixture counts and fails class
    assert data["gate"]["structural_errors"] == {"value": 3, "target": None, "result": "no target"}
    assert data["gate"]["coverage.subject"]["result"] == "no target"
    assert data["gate"]["result"] == "no target"  # with nothing to judge, the gate neither passes nor fails


def test_one_area(cfg, tmp_path, capsys):
    indexed(cfg)
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--area", "Heating", "--json")
    data = json.loads(out.out)
    assert [a["area"] for a in data["areas"]] == ["Heating"]
    assert data["total"]["notes"] == 3
    assert data["gate"]["structural_errors"]["result"] == "fail"


def test_text_output_marks_accepted_issues_and_the_gate(cfg, tmp_path, capsys):
    indexed(cfg)
    code, out = run(cfg, tmp_path, capsys, "--vault", "V")
    assert code == 0
    assert f"{LINK}: error [4] link [[gone]] resolves to no note (accepted: fixtures)" in out.out
    assert f"{LOG}: error [4] link [[nowhere]] resolves to no note\n" in out.out
    assert "result: fail" in out.out
    assert "no target" not in out.out


def test_health_scans_the_vault_before_it_reads(cfg, tmp_path, capsys, monkeypatch):
    indexed(cfg)
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda c: FakeEmbedder())
    (cfg.root / "V/Heating/new.md").write_text("---\nclass: record\nas_of: 2026-03-01\n---\n# New\n")
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json")
    heating = {a["area"]: a for a in json.loads(out.out)["areas"]}["Heating"]
    assert heating["notes"] == 4
    assert heating["coverage"]["as_of"] == cov(1, 2)


def test_save_adds_one_line_for_each_area_and_the_totals(cfg, tmp_path, capsys):
    indexed(cfg)
    run(cfg, tmp_path, capsys, "--vault", "V", "--save")
    run(cfg, tmp_path, capsys, "--vault", "V", "--save")
    lines = [json.loads(line) for line in (cfg.data_dir / "health" / "history.jsonl").read_text().splitlines()]
    assert len(lines) == 10
    first = lines[:5]
    assert [line["area"] for line in first] == ["(top)", "Entities", "Fixtures", "Heating", "(total)"]
    assert len({line["time"] for line in first}) == 1
    assert first[-1]["structural_errors"]["not_accepted"] == 1
    assert first[-1]["coverage"]["subject"] == cov(3, 5)


def test_an_unknown_vault_fails(cfg, tmp_path, capsys):
    indexed(cfg)
    code, out = run(cfg, tmp_path, capsys, "--vault", "Nope")
    assert code == 1
    assert "Unknown vault Nope" in out.err


def test_health_changes_no_note(cfg, tmp_path, capsys):
    indexed(cfg)
    before = {rel: (cfg.root / rel).read_bytes() for rel in NOTES}
    run(cfg, tmp_path, capsys, "--vault", "V", "--save")
    assert {rel: (cfg.root / rel).read_bytes() for rel in NOTES} == before
