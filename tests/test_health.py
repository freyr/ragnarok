import json

import pytest
from test_index import FakeEmbedder

from ragnarok import health as health_module
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


def test_health_counts_a_batch_of_new_notes_larger_than_the_scan_limit(cfg, tmp_path, capsys, monkeypatch):
    indexed(cfg)
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda c: FakeEmbedder())
    for n in range(25):
        (cfg.root / f"V/Heating/new-{n}.md").write_text(f"---\nclass: record\nas_of: 2026-03-01\n---\n# New {n}\n")
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json")
    heating = {a["area"]: a for a in json.loads(out.out)["areas"]}["Heating"]
    assert heating["notes"] == 28
    assert heating["coverage"]["as_of"] == cov(25, 26)


def test_an_accepted_bad_validity_fixture_is_out_of_the_coverage_count(cfg, tmp_path, capsys):
    (cfg.root / "V/Fixtures/validity.md").write_text("---\nclass: record\nvalidity: current\n---\n# Validity\n")
    indexed(cfg)
    health = HEALTH_TOML.replace('"bad-yaml"]', '"bad-yaml", "bad-validity"]')
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json", health=health)
    fixtures = {a["area"]: a for a in json.loads(out.out)["areas"]}["Fixtures"]
    assert fixtures["notes"] == 3
    assert fixtures["structural_errors"] == {"errors": 3, "accepted": 3, "not_accepted": 0}
    # only the link fixture counts; the record with a validity would add a class and a missing as_of
    assert fixtures["coverage"] == {
        "class": cov(1, 1),
        "validity": cov(0, 0),
        "as_of": cov(1, 1),
        "subject": cov(1, 1),
    }


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


# --- the worklist ------------------------------------------------------------------------------------------------

REC = "---\nclass: record\nas_of: 2026-01-01\n"
STATE = "---\nclass: state\nvalidity: current\nas_of: 2026-01-01\nsubject: ['[[Heat pump]]']\n---\n# State\n"

WORK_NOTES = {
    "V/Entities/Heat pump.md": "---\ntype: entity\nkind: device\n---\n# Heat pump\n",
    # the targets
    "V/Guides/Heat pump guide.md": REC + "---\n",
    "V/_archive/Old guide.md": REC + "---\n",
    "V/Docs/README.md": REC + "---\n",
    "V/A/Twin.md": REC + "---\n",
    "V/_archive/Twin.md": REC + "---\n",
    "V/Astro/Sub/Dashboard.md": REC + "---\n",
    "V/Folder/inner.md": REC + "---\n",
    "W/Notes/Only in W.md": REC + "---\n",
    # one link of each class of the research
    "V/Src/links.md": REC
    + "---\n[[Old/Heat pump guide]] [[Old/Old guide]] [[Old/README]] [[Old/Twin]] [[Old/Heat pump gide]]"
    + " [[Old/Folder]] [[Old/Nowhere]] [[Old/Only in W]] ![[Old/Heat pump guide.md#Top|Guide]]\n",
    "V/Astro/src.md": REC + "---\n[[Old/Dashboard]]\n",
    "V/Other/src.md": REC + "---\n[[Old/Dashboard]]\n",
    "V/Fixtures/assets/fx.md": REC + "---\n[[Old/Heat pump guide]]\n",
    "V/Skip/src.md": REC + "---\n[[Old/Heat pump guide]]\n",
    # the other kinds
    "V/Y/fixable.md": REC + "title: Foo: bar\n---\n",
    "V/Y/quoted.md": REC + 'title: "a": b\n---\n',
    "V/Y/multiline.md": REC + "title: Foo: bar\n  more\n---\n",
    "V/Y/nested.md": REC + "meta:\n  title: Foo: bar\n---\n",
    "V/Y/bracket.md": "---\nclass: [state\n---\n",
    "V/K/one.md": STATE,
    "V/K/two.md": STATE,
    "V/K/supersedes.md": REC + "supersedes: nothing\n---\n",
    "V/K/ghost.md": REC + "subject: ['[[Ghost]]']\n---\n",
}
WORK_TOML = """
[health]
accept = [
  { glob = "V/Skip/**", kinds = ["broken-link"], reason = "skipped" },
]
"""


@pytest.fixture
def wcfg(tmp_path):
    root = tmp_path / "vaults"
    for rel, text in WORK_NOTES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    cfg = Config(root=root, index_path=tmp_path / "index.db", vaults=["V", "W"], dimension=8)
    indexed(cfg)
    return cfg


def work(wcfg, tmp_path, capsys, *args):
    code, out = run(wcfg, tmp_path, capsys, "--worklist", "--json", *args, health=WORK_TOML)
    assert code == 0
    return json.loads(out.out)["worklist"]


def test_worklist_gives_one_fix_for_each_link_class(wcfg, tmp_path, capsys):
    lines = work(wcfg, tmp_path, capsys)
    links = {(w["detail"].split("]]")[0] + "]]", w["path"]): w for w in lines if w["kind"] == "broken-link"}
    src, astro = "V/Src/links.md", "V/Astro/src.md"
    fix, user = "ragnarok fix links", "the user"
    expected = {
        ("[[Old/Heat pump guide]]", src): (fix, "V/Guides/Heat pump guide.md"),  # stale prefix, unique, specific
        ("[[Old/Old guide]]", src): (fix, "V/_archive/Old guide.md"),  # _archive counts
        ("[[Old/Dashboard]]", astro): (fix, "V/Astro/Sub/Dashboard.md"),  # Dashboard in the same area
        ("[[Old/Dashboard]]", "V/Other/src.md"): (user, "generic-name"),  # Dashboard in another area
        ("[[Old/README]]", src): (user, "generic-name"),
        ("[[Old/Twin]]", src): (user, "several-candidates"),
        ("[[Old/Heat pump gide]]", src): (user, "no-candidate"),  # similar name only
        ("[[Old/Folder]]", src): (user, "no-candidate"),  # a folder
        ("[[Old/Nowhere]]", src): (user, "no-candidate"),
        ("[[Old/Only in W]]", src): (user, "cross-vault-only"),
        ("[[Old/Heat pump guide]]", "V/Fixtures/assets/fx.md"): (user, "accepted-or-assets"),
    }
    assert {k: (v["fix"], v["target"] or v["reason"]) for k, v in links.items()} == expected
    assert not [w for w in lines if w["path"] == "V/Skip/src.md"]  # accepted: no line


def test_worklist_maps_the_other_kinds(wcfg, tmp_path, capsys):
    lines = work(wcfg, tmp_path, capsys)
    other = {(w["path"], w["kind"]): w["fix"] for w in lines if w["kind"] != "broken-link"}
    assert other == {
        ("V/Y/fixable.md", "bad-yaml"): "ragnarok fix yaml",
        ("V/Y/quoted.md", "bad-yaml"): "the user",
        ("V/Y/multiline.md", "bad-yaml"): "the user",
        ("V/Y/nested.md", "bad-yaml"): "the user",
        ("V/Y/bracket.md", "bad-yaml"): "the user",
        ("V/K/one.md", "double-identity"): "kb-write",
        ("V/K/two.md", "double-identity"): "kb-write",
        ("V/K/supersedes.md", "bad-supersedes"): "kb-write",
        ("V/K/supersedes.md", "missing-target"): "the user",
        ("V/K/ghost.md", "missing-target"): "the user",
    }


def test_worklist_has_one_line_for_each_not_accepted_error_sorted_by_area_then_fix(wcfg, tmp_path, capsys):
    _, out = run(wcfg, tmp_path, capsys, "--worklist", "--json", health=WORK_TOML)
    data = json.loads(out.out)
    assert len(data["worklist"]) == data["total"]["structural_errors"]["not_accepted"]
    keys = [(w["vault"], w["area"], w["fix"]) for w in data["worklist"]]
    assert keys == sorted(keys)
    assert "worklist" not in json.loads(run(wcfg, tmp_path, capsys, "--json", health=WORK_TOML)[1].out)


def test_worklist_text_has_one_line_for_each_defect(wcfg, tmp_path, capsys):
    _, out = run(wcfg, tmp_path, capsys, "--worklist", "--vault", "V", health=WORK_TOML)
    line = "V/Src/links.md: broken-link [[Old/Heat pump guide]] -> V/Guides/Heat pump guide.md  fix: ragnarok fix links"
    assert line in out.out.splitlines()
    assert "V/Y/quoted.md: bad-yaml" in out.out
    assert "fix: " not in run(wcfg, tmp_path, capsys, "--vault", "V", health=WORK_TOML)[1].out


def test_link_decision_is_the_function_of_the_worklist_and_the_fixer(wcfg):
    conn = connect(wcfg)
    names = health_module.note_names(conn)
    conn.close()
    auto = health_module.link_decision(wcfg, names, "V/Src/links.md", "Old/Heat pump guide.md")
    assert (auto.auto, auto.target, auto.reason) == (True, "V/Guides/Heat pump guide.md", None)
    # a case difference and the extension do not matter
    assert health_module.link_decision(wcfg, names, "V/Src/links.md", "old/HEAT PUMP GUIDE").auto
    wcfg.health = HealthConfig(accept=[AcceptRule(glob="V/Skip/**", kinds=["broken-link"], reason="skipped")])
    skip = health_module.link_decision(wcfg, names, "V/Skip/src.md", "Old/Heat pump guide")
    assert (skip.auto, skip.reason) == (False, "accepted-or-assets")
