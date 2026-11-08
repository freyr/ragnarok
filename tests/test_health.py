import hashlib
import json
import os
import re
import time

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
        "drift": {"value": None, "target": None, "result": "no baseline"},
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
    assert data["gate"]["drift"] == {"value": None, "target": None, "result": "no baseline"}
    assert data["total"]["drift"] is None and data["areas"][0]["drift"] is None  # no baseline file: no failure
    assert data["gate"]["result"] == "no target"  # with nothing to judge, the gate neither passes nor fails


def test_one_area(cfg, tmp_path, capsys):
    indexed(cfg)
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--area", "Heating", "--json")
    data = json.loads(out.out)
    assert [a["area"] for a in data["areas"]] == ["Heating"]
    assert data["total"]["notes"] == 3
    assert data["gate"]["structural_errors"]["result"] == "fail"


def test_text_output_gives_only_the_table_and_the_gate(cfg, tmp_path, capsys):
    indexed(cfg)
    code, out = run(cfg, tmp_path, capsys, "--vault", "V")
    assert code == 0
    assert out.out.startswith("area ")  # no issue lines before the table
    assert LOG not in out.out and LINK not in out.out
    assert "result: fail" in out.out
    assert "  drift: no baseline" in out.out
    assert "no target" not in out.out
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--worklist")
    assert out.out.startswith("area ")
    assert f"{LOG}: broken-link [[nowhere]] (no-candidate)  fix: the user" in out.out  # the issue is in the worklist


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


def test_subject_none_with_a_reason_counts_as_a_subject(cfg, tmp_path, capsys):
    (cfg.root / "V/Index").mkdir()
    (cfg.root / "V/Index/i.md").write_text(
        "---\nclass: record\nas_of: 2026-02-01\nsubject: none\nsubject_reason: index note\n---\n# Index\n"
    )
    indexed(cfg)
    _, out = run(cfg, tmp_path, capsys, "--vault", "V", "--json")
    index = {a["area"]: a for a in json.loads(out.out)["areas"]}["Index"]
    assert index["structural_errors"] == {"errors": 0, "accepted": 0, "not_accepted": 0}
    assert index["coverage"]["subject"] == cov(1, 1)


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
    "V/Src/cut.md": REC
    + "---\n[[Old/Heat pump gu\n- class: x]] [[@name]] [[1-on-1 {{date}}]] [[Board.base]] [[Only in W!]]\n",
    "V/Fixtures/assets/cut.md": REC + "---\n[[Old/Heat pump gu\n- class: x]]\n",
    "V/Fixtures/assets/yaml.md": REC + "title: Foo: bar\n---\n",
    "V/Labels/bare.md": "---\ntype: guide\n---\n# Bare\n",
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
    assert len(links) == len([w for w in lines if w["kind"] == "broken-link"])
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
        ("[[Old/Heat pump gu\n- class: x]]", "V/Src/cut.md"): (user, "no-candidate"),  # cut link text
        ("[[Old/Heat pump gu\n- class: x]]", "V/Fixtures/assets/cut.md"): (user, "accepted-or-assets"),
        ("[[@name]]", "V/Src/cut.md"): (user, "no-candidate"),  # placeholders
        ("[[1-on-1 {{date}}]]", "V/Src/cut.md"): (user, "no-candidate"),
        ("[[Board.base]]", "V/Src/cut.md"): (user, "no-candidate"),  # the validator false positive
        ("[[Only in W!]]", "V/Src/cut.md"): (user, "no-candidate"),  # punctuation differs, note in another vault
    }
    assert {k: (v["fix"], v["target"] or v["reason"]) for k, v in links.items()} == expected
    assert not [w for w in lines if w["path"] == "V/Skip/src.md" and not w["label"]]  # accepted: no line


def test_worklist_maps_the_other_kinds(wcfg, tmp_path, capsys):
    lines = work(wcfg, tmp_path, capsys)
    other = {(w["path"], w["kind"]): w["fix"] for w in lines if w["kind"] != "broken-link" and not w["label"]}
    assert other == {
        ("V/Fixtures/assets/yaml.md", "bad-yaml"): "the user",  # a fixture is never fixed
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
    errors = [w for w in data["worklist"] if not w["label"]]
    assert len(errors) == data["total"]["structural_errors"]["not_accepted"]
    keys = [(w["vault"], w["area"], w["fix"]) for w in data["worklist"]]
    assert keys == sorted(keys)
    assert "worklist" not in json.loads(run(wcfg, tmp_path, capsys, "--json", health=WORK_TOML)[1].out)


def test_worklist_has_a_label_line_for_each_missing_required_key(wcfg, tmp_path, capsys):
    lines = work(wcfg, tmp_path, capsys)
    bare = [(w["kind"], w["fix"], w["detail"]) for w in lines if w["path"] == "V/Labels/bare.md"]
    assert sorted(bare) == [
        ("no-class", "kb-convert Labels", "class is missing"),
        ("no-subject", "kb-convert Labels", "subject is missing"),
    ]
    assert all(w["label"] for w in lines if w["path"] == "V/Labels/bare.md")
    # an entity needs no subject (only its class is missing); a record with as_of needs no validity
    kinds = {
        p: {w["kind"] for w in lines if w["label"] and w["path"] == p}
        for p in ("V/Entities/Heat pump.md", "V/Guides/Heat pump guide.md")
    }
    assert kinds == {"V/Entities/Heat pump.md": {"no-class"}, "V/Guides/Heat pump guide.md": {"no-subject"}}
    # a note with bad YAML that is not accepted counts, as in the coverage
    assert {w["kind"] for w in lines if w["label"] and w["path"] == "V/Y/bracket.md"} == {"no-class", "no-subject"}


def test_worklist_text_has_one_line_for_each_defect(wcfg, tmp_path, capsys):
    _, out = run(wcfg, tmp_path, capsys, "--worklist", "--vault", "V", health=WORK_TOML)
    line = "V/Src/links.md: broken-link [[Old/Heat pump guide]] -> V/Guides/Heat pump guide.md  fix: ragnarok fix links"
    assert line in out.out.splitlines()
    assert "V/Y/quoted.md: bad-yaml" in out.out
    assert "V/Labels/bare.md: no-class class is missing  fix: kb-convert Labels" in out.out.splitlines()
    cut = [x for x in out.out.splitlines() if x.startswith("V/Src/cut.md: broken-link [[Old/Heat pump gu")]
    assert cut == ["V/Src/cut.md: broken-link [[Old/Heat pump gu - class: x]] (no-candidate)  fix: the user"]
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


# --- the drift baseline ------------------------------------------------------------------------------------------

DATED = "V/Heating/dated.md"
CHANGED = "V/Heating/changed.md"
NEWER = "V/Heating/newer.md"
OLD_GUIDE = "---\nclass: state\nvalidity: current\nas_of: 2025-12-01\nsubject: ['[[Heat pump]]']\n---\n# Old guide\n"
LIVE_DATED = "---\nclass: record\nas_of: 2026-03-01\nsubject: ['[[Heat pump]]']\n---\n# Dated\nCorrection 2026-03-01.\n"
TEMPLATE_DATED = LIVE_DATED.replace("2026-03-01", "{{apply-date}}")
LIVE_CHANGED = "---\nclass: record\nas_of: 2026-03-02\nsubject: ['[[Heat pump]]']\n---\n# Changed by hand\n"
TEMPLATE_CHANGED = LIVE_CHANGED.replace("2026-03-02", "{{apply-date}}").replace("by hand", "by the conversion")
PROPOSED_LOG = NOTES[LOG].replace("# Log", "# Log before the edit")
DRIFT_TOML = HEALTH_TOML + "drift = 0.6\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_proposed(w10, area, key, text, folder="proposed"):
    path = w10 / area / folder / f"{key}.proposed"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def dcfg(cfg, tmp_path):
    """The live notes of NOTES plus three notes of the area Heating; the W10 folder is tmp_path / "w10"."""
    for key, text in {DATED: LIVE_DATED, CHANGED: LIVE_CHANGED, NEWER: LIVE_DATED}.items():
        (cfg.root / key).write_text(text)
    w10 = tmp_path / "w10"
    write_proposed(w10, "Heating", GUIDE, NOTES[GUIDE])  # equal to the live note
    write_proposed(w10, "Heating", LOG, PROPOSED_LOG)  # the live note changed after the conversion
    write_proposed(w10, "Heating", DATED, TEMPLATE_DATED)  # equal with the date read as any date
    write_proposed(w10, "Heating", CHANGED, TEMPLATE_CHANGED)  # the live note differs and has another date
    write_proposed(w10, "Heating", "V/Heating/skipped.md", "skipped\n", folder="skipped/proposed")  # not applied
    (w10 / "Heating" / "real-before.txt").write_text(
        f"{sha(NOTES[BOILER])}  ./boiler.md\n"  # a note that the conversion did not change
        f"{sha(OLD_GUIDE)}  ./guide.md\n"  # the text before the conversion: the proposed text wins
        f"{sha('gone')}  ./gone.md\n"  # a note that no longer exists: kept, harmless
    )
    return cfg


def seed(cfg, tmp_path, capsys, *args):
    path = config_file(cfg, tmp_path, DRIFT_TOML)
    cutoff = [] if "--cutoff" in args else ["--cutoff", "2000-01-01T00:00:00"]  # before every file: no live hash
    code = main(["--config", str(path), "health", "baseline", "seed", *args, *cutoff])
    return code, capsys.readouterr()


def baseline(cfg):
    return json.loads((cfg.data_dir / "health" / "baseline.json").read_text())


def test_seed_writes_the_hash_and_the_date_of_each_note(dcfg, tmp_path, capsys):
    code, out = seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    assert code == 0
    data = baseline(dcfg)
    assert {k: (v["sha256"], v["source"]) for k, v in data.items()} == {
        GUIDE: (sha(NOTES[GUIDE]), "proposed"),
        LOG: (sha(PROPOSED_LOG), "proposed"),
        # the date is read as any date: the live note matches, so its hash is the baseline
        DATED: (sha(LIVE_DATED), "proposed-live"),
        # the live note does not match the template: the template hash never equals a live note
        CHANGED: (sha(TEMPLATE_CHANGED), "proposed-unmatched"),
        "V/Heating/boiler.md": (sha(NOTES[BOILER]), "real-before"),
        "V/Heating/gone.md": (sha("gone"), "real-before"),
    }
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", v["date"]) for v in data.values())
    assert "proposed texts: 4 in 1 areas, equal to the live note: 2, changed: 2" in out.out
    assert f"  {LOG}\n" in out.out and f"  {CHANGED}\n" in out.out
    assert "real-before only: 2, equal to the live note: 1, changed: 1" in out.out
    assert "pre-cutoff-live: 0" in out.out
    # the baseline stays fixed: a second seed does not overwrite it
    before = (dcfg.data_dir / "health" / "baseline.json").read_text()
    code, out = seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    assert code == 1
    assert "baseline exists" in out.err
    assert (dcfg.data_dir / "health" / "baseline.json").read_text() == before


def test_seed_needs_the_w10_folder(dcfg, tmp_path, capsys):
    with pytest.raises(SystemExit):
        seed(dcfg, tmp_path, capsys)
    capsys.readouterr()
    (tmp_path / "empty").mkdir()
    code, out = seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "empty"))
    assert code == 1
    assert "no area with a proposed folder" in out.err
    assert not (dcfg.data_dir / "health" / "baseline.json").exists()


def drift(data, area):
    return {a["area"]: a for a in data["areas"]}[area]["drift"]


def test_health_reports_the_changed_notes_and_the_drift_share(dcfg, tmp_path, capsys):
    seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    indexed(dcfg)
    code, out = run(dcfg, tmp_path, capsys, "--vault", "V", "--json", health=DRIFT_TOML)
    assert code == 0
    data = json.loads(out.out)
    # Heating: log (broken link), changed (valid), newer (new, valid); dated and guide equal the baseline
    assert drift(data, "Heating") == {"changed": 3, "passing": 2, "share": 0.6667}
    # not in the baseline: entity (no class), top (no validity); link (accepted error) passes; bad (accepted bad YAML)
    # is out of the count, as in coverage
    assert drift(data, "Entities") == {"changed": 1, "passing": 0, "share": 0.0}
    assert drift(data, "Fixtures") == {"changed": 1, "passing": 1, "share": 1.0}
    assert drift(data, "(top)") == {"changed": 1, "passing": 0, "share": 0.0}
    assert data["total"]["drift"] == {"changed": 6, "passing": 3, "share": 0.5}
    assert data["gate"]["drift"] == {"value": 0.5, "target": 0.6, "result": "fail"}
    assert data["gate"]["result"] == "fail"
    _, out = run(dcfg, tmp_path, capsys, "--vault", "V", health=DRIFT_TOML)
    assert "  drift: 3/6 50.0%  target 60.0%  fail" in out.out


def test_drift_is_no_target_without_the_target_and_counts_only_baseline_vaults(dcfg, tmp_path, capsys):
    seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    (dcfg.root / "W").mkdir()
    (dcfg.root / "W/other.md").write_text(NOTES[GUIDE])
    dcfg.vaults.append("W")
    indexed(dcfg)
    _, out = run(dcfg, tmp_path, capsys, "--json", health=HEALTH_TOML)
    data = json.loads(out.out)
    assert data["gate"]["drift"] == {"value": 0.5, "target": None, "result": "no target"}
    assert data["total"]["drift"]["changed"] == 6  # the note of W is not counted: W has no baseline entry


def test_an_edit_after_the_seed_makes_a_changed_note_and_update_baseline_takes_it_back(dcfg, tmp_path, capsys):
    seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    (dcfg.root / GUIDE).write_text(NOTES[GUIDE] + "An edit.\n")
    (dcfg.root / LOG).write_text(PROPOSED_LOG)  # back to the baseline text
    indexed(dcfg)
    _, out = run(dcfg, tmp_path, capsys, "--vault", "V", "--area", "Heating", "--json")
    assert drift(json.loads(out.out), "Heating") == {"changed": 3, "passing": 3, "share": 1.0}  # guide, changed, newer
    assert health_module.update_baseline(dcfg, [GUIDE, NEWER, "V/Heating/nothing.md"]) == 2
    data = baseline(dcfg)
    assert (data[GUIDE]["sha256"], data[GUIDE]["source"]) == (sha(NOTES[GUIDE] + "An edit.\n"), "update")
    assert data[NEWER]["sha256"] == sha(LIVE_DATED)
    assert "V/Heating/nothing.md" not in data
    _, out = run(dcfg, tmp_path, capsys, "--vault", "V", "--area", "Heating", "--json")
    assert drift(json.loads(out.out), "Heating") == {"changed": 1, "passing": 1, "share": 1.0}  # only `changed`


def test_update_baseline_does_nothing_without_a_baseline(dcfg):
    assert health_module.update_baseline(dcfg, [GUIDE]) == 0
    assert not (dcfg.data_dir / "health").exists()


def age(cfg, key, iso):
    stamp = time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%S"))
    os.utime(cfg.root / key, (stamp, stamp))


def test_seed_takes_the_live_hash_of_a_note_at_or_before_the_cutoff(dcfg, tmp_path, capsys):
    for key in (LOG, ENTITY, GUIDE):
        age(dcfg, key, "2026-01-01T00:00:00")
    age(dcfg, TOP, "2026-06-01T00:00:00")  # at the cutoff
    code, out = seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"), "--cutoff", "2026-06-01T00:00:00")
    assert code == 0
    data = {k: (v["sha256"], v["source"]) for k, v in baseline(dcfg).items()}
    # differs from its W10 entry, or has none, and is not newer than the cutoff
    assert data[LOG] == (sha(NOTES[LOG]), "pre-cutoff-live")
    assert data[ENTITY] == (sha(NOTES[ENTITY]), "pre-cutoff-live")
    assert data[TOP] == (sha(NOTES[TOP]), "pre-cutoff-live")
    assert data[GUIDE] == (sha(NOTES[GUIDE]), "proposed")  # equal to its W10 entry: unchanged
    # newer than the cutoff: the current rules (a changed text stays changed, a new path stays new)
    assert data[CHANGED] == (sha(TEMPLATE_CHANGED), "proposed-unmatched")
    assert BAD not in data and LINK not in data
    assert "pre-cutoff-live: 3" in out.out
    # the counts of the written file: the entries by source and the entries that differ from the live note
    assert (
        "entries by source: pre-cutoff-live 3, proposed 1, proposed-live 1, proposed-unmatched 1, real-before 2"
        in out.out
    )
    assert "entries that differ from the live note: 2\n" in out.out
    assert out.out.endswith(f"  {CHANGED}\n  V/Heating/gone.md\n")
    indexed(dcfg)
    _, out = run(dcfg, tmp_path, capsys, "--vault", "V", "--json", health=DRIFT_TOML)
    # changed, newer and link pass; bad has an accepted frontmatter defect and is out of the count
    assert json.loads(out.out)["total"]["drift"] == {"changed": 3, "passing": 3, "share": 1.0}


def test_seed_needs_a_cutoff(dcfg, tmp_path, capsys):
    path = config_file(dcfg, tmp_path, DRIFT_TOML)
    with pytest.raises(SystemExit):
        main(["--config", str(path), "health", "baseline", "seed", "--from", str(tmp_path / "w10")])
    assert "--cutoff" in capsys.readouterr().err


def test_the_baseline_is_written_through_a_temp_file(dcfg, tmp_path, capsys, monkeypatch):
    seed(dcfg, tmp_path, capsys, "--from", str(tmp_path / "w10"))
    before = (dcfg.data_dir / "health" / "baseline.json").read_text()

    def stop(src, dst):
        raise OSError("stop")

    monkeypatch.setattr("ragnarok.health.os.replace", stop)
    with pytest.raises(OSError):
        health_module.update_baseline(dcfg, [LOG])
    assert (dcfg.data_dir / "health" / "baseline.json").read_text() == before  # never half written
    monkeypatch.undo()
    assert health_module.update_baseline(dcfg, [LOG]) == 1
    assert sorted(p.name for p in (dcfg.data_dir / "health").iterdir()) == ["baseline.json"]  # no temp file is left
