from pathlib import Path

import pytest

from ragnarok.config import Config, home_dir, load_config
from ragnarok.metadata import iter_notes, resolve, resolve_file


def meta(cfg, rel):
    return resolve_file(cfg, rel)


def test_walker_skips_dot_folders_and_node_modules(vault):
    notes = set(iter_notes(vault))
    assert "Home/Heating/SUMMARY.md" in notes
    assert not any("node_modules" in n or "/.claude/" in n or "/.obsidian/" in n for n in notes)


@pytest.mark.parametrize(
    "rel, area, project, status, type_",
    [
        ("Work/Devops/db-resize/SUMMARY.md", "devops", "db-resize", "active", "summary"),
        ("Work/Devops/db-resize/research-cpu.md", "devops", "db-resize", "active", "research"),
        (
            "Work/Devops/_archive/db-migration/Q7-cost.md",
            "devops",
            "db-migration",
            "archived",
            "note",
        ),
        ("Work/Sales/Answers/Security.md", "sales", "Answers", "active", "note"),
        ("Work/SLA/2026-Q1 review.md", "sla", "none", "active", "note"),
        ("Clients/Acme/01-Projects/fsm/notes.md", "acme", "fsm", "active", "note"),
        ("Clients/Acme/99-Templates/Meeting.md", "acme", "Acme", "active", "template"),
        (
            "Clients/Queue-Research/Block-04.md",
            "queue-research",
            "Queue-Research",
            "active",
            "reference",
        ),
        ("Clients/docs/plans/2026-03-13-feat-agents-plan.md", "clients", "none", "active", "plan"),
        ("Clients/Root guide.md", "clients", "none", "active", "note"),
        (
            "Studio/api-gateway/docs/brainstorms/2026-03-01-x.md",
            "studio",
            "api-gateway",
            "active",
            "brainstorm",
        ),
        ("Home/Heating/notes/boiler.md", "heating", "Heating", "active", "note"),
        ("Home/Astro/Theory/Field Stop.md", "astro", "Theory", "active", "note"),
    ],
)
def test_path_rules(vault, rel, area, project, status, type_):
    m = meta(vault, rel)
    assert (m.area, m.project, m.status, m.type) == (area, project, status, type_)


def test_frontmatter_overrides_path_rules(vault):
    m = meta(vault, "Home/Bike/shortlist.md")
    assert (m.project, m.area, m.status) == ("bike-2026", "bikes", "archived")
    assert m.sources["project"] == "frontmatter"


@pytest.mark.parametrize(
    "value, project",
    [
        ("[[Devops/tenant-split/SUMMARY]]", "tenant-split"),
        ("[[Devops/_archive/db-migration/SUMMARY|old]]", "db-migration"),
        ("[[homelab]]", "homelab"),
        ("homelab", "homelab"),
    ],
)
def test_project_wikilink_becomes_folder_name(vault, value, project):
    assert resolve(vault, "Home/Bike/x.md", {"project": value}, "", lambda f: False).project == project


def test_description_from_frontmatter(vault):
    assert meta(vault, "Work/Devops/db-resize/SUMMARY.md").description == "Resize the database servers."


@pytest.mark.parametrize(
    "rel, private",
    [
        ("Work/Personal/People/Someone.md", True),
        ("Work/SLA/Incidents/2026-01-01.md", True),
        ("Work/SLA/2026-Q1 review.md", False),
        ("Home/Password/bank.md", True),
        ("Home/Budget/2026.md", True),
        ("Home/Heating/SUMMARY.md", False),
    ],
)
def test_sensitivity_from_path_patterns(vault, rel, private):
    assert (meta(vault, rel).sensitivity == "private") is private


def test_frontmatter_cannot_lower_sensitivity(vault):
    m = resolve(vault, "Home/Password/bank.md", {"sensitivity": "normal"}, "", lambda f: False)
    assert m.sensitivity == "private"


def test_jira_rule(vault):
    assert meta(vault, "Work/Devops/db-resize/research-cpu.md").jira == ["OPS-77", "TS-9"]
    assert meta(vault, "Clients/docs/plans/2026-03-13-feat-agents-plan.md").jira == []  # FR is a prefix of Work only
    assert meta(vault, "Home/Astro/Theory/Field Stop.md").jira == []  # ES is not a prefix of Home
    assert meta(vault, "Studio/api-gateway/docs/brainstorms/2026-03-01-x.md").jira == ["CORE-123"]


def test_load_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('root = "/tmp/v"\nallow_private = true\nmin_words = 50\n')
    cfg = load_config(path)
    assert (str(cfg.root), cfg.allow_private, cfg.min_words) == ("/tmp/v", True, 50)
    assert load_config(tmp_path / "missing.toml") == Config()


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("allow_privat = true\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_config_paths_are_relative_to_the_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGNAROK_HOME", str(tmp_path / "home"))
    path = tmp_path / "config.toml"
    path.write_text('root = "vaults"\ndata_dir = "store"\n')
    cfg = load_config(path)
    assert (cfg.root, cfg.data_dir, cfg.index_path) == (
        tmp_path / "vaults",
        tmp_path / "store",
        tmp_path / "store/index.db",
    )
    assert Config().index_path == tmp_path / "home/data/index.db"


def test_vault_paths_give_the_folder_of_a_vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    path = tmp_path / "config.toml"
    path.write_text('root = "vaults"\nvaults = ["A", "B", "C"]\n\n[vault_paths]\nB = "~/code/b"\nC = "c-kb"\n')
    cfg = load_config(path)
    assert [cfg.vault_dir(v) for v in cfg.vaults] == [
        tmp_path / "vaults/A",
        tmp_path / "user/code/b",
        tmp_path / "c-kb",
    ]
    assert cfg.note_path("B/x/y.md") == tmp_path / "user/code/b/x/y.md"
    assert cfg.note_path("A/y.md") == tmp_path / "vaults/A/y.md"


def test_a_config_without_vault_paths_keeps_its_rules_hash():
    from ragnarok.config import rules_hash

    # the hash of 2d18ddf, before vault_paths: an existing index must not re-derive
    assert rules_hash(Config(root=Path("/vaults"), vaults=["Notes"])) == "d0b71fa70ee3f689f3d3bcdcea69ec1612e3c144"


def test_the_default_home_is_the_repo_checkout(monkeypatch):
    monkeypatch.delenv("RAGNAROK_HOME", raising=False)
    assert (home_dir() / "pyproject.toml").exists()
    assert Config().data_dir == home_dir() / "data"


def test_a_vault_without_rules_uses_the_generic_rules(vault):
    vault.vault_rules = {}
    m = meta(vault, "Work/Sales/Answers/Security.md")
    assert (m.area, m.project) == ("sales", "Answers")
    assert meta(vault, "Work/SLA/2026-Q1 review.md").project == "SLA"


def test_no_jira_prefix_means_no_jira_key(vault):
    vault.jira_prefixes, vault.vault_rules = [], {}
    assert meta(vault, "Work/Devops/db-resize/research-cpu.md").jira == []


def test_vault_rules_round_trip_through_toml(vault, tmp_path):
    from ragnarok.config import dump_config

    vault.vault_paths = {"Studio": tmp_path / "studio-kb"}
    path = tmp_path / "copy.toml"
    path.write_text(dump_config(vault, root="vaults", data_dir="data"))
    copy = load_config(path)
    assert (copy.root, copy.data_dir) == (tmp_path / "vaults", tmp_path / "data")
    assert copy.vault_paths == vault.vault_paths
    assert (copy.vault_rules, copy.jira_prefixes, copy.vaults) == (vault.vault_rules, vault.jira_prefixes, vault.vaults)


def test_unknown_rule_key_is_an_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[vault_rules.Work]\nproject_level = 2\n")
    with pytest.raises(TypeError):
        load_config(path)


def test_frontmatter_area_is_lower_case(vault):
    assert resolve(vault, "Home/Bike/x.md", {"area": "Devops"}, "", lambda f: False).area == "devops"


@pytest.mark.parametrize(
    "raw, type_",
    [
        ("design", "spec"),
        ("brief", "spec"),
        ("incident-rca", "review"),
        ("sla-review", "review"),
        ("one-pager", "brainstorm"),
        ("content-analysis", "research"),
        ("facilitator-runbook", "guide"),
        ("digest", "log"),
    ],
)
def test_type_aliases(vault, raw, type_):
    m = resolve(vault, "Home/Bike/x.md", {"type": raw}, "", lambda f: False)
    assert (m.type, m.sources["type"]) == (type_, "frontmatter-map")


@pytest.mark.parametrize("stage, status", [("DONE", "active"), ("done", "active"), ("DELIVER", "active")])
def test_summary_stage_done_does_not_archive_the_project(vault, stage, status):
    (vault.root / "Home/Heating/SUMMARY.md").write_text(f"---\nstage: {stage}\n---\n# Heating\n")
    m = meta(vault, "Home/Heating/notes/boiler.md")
    assert m.status == status
    assert m.sources["status"] == ("summary" if status == "archived" else "path")


@pytest.mark.parametrize(
    "state, status",
    [
        ("done", "active"),
        ("superseded", "archived"),
        ("abandoned", "archived"),
        ("paused", "active"),
        ("in-progress", "active"),
    ],
)
def test_summary_state_archives_the_project(vault, state, status):
    (vault.root / "Home/Heating/SUMMARY.md").write_text(f"---\nstage: DESIGN\nstate: {state}\n---\n# Heating\n")
    assert meta(vault, "Home/Heating/notes/boiler.md").status == status


def test_done_project_under_archive_folder_is_archived_by_path(vault):
    (vault.root / "Work/Devops/_archive/db-migration/SUMMARY.md").write_text("---\nstate: done\n---\n# Old\n")
    m = meta(vault, "Work/Devops/_archive/db-migration/Q7-cost.md")
    assert (m.status, m.sources["status"]) == ("archived", "path")


def test_kb_labels_come_from_frontmatter(vault):
    fm = {"class": "State", "validity": "current", "as_of": "2026-11-01"}
    m = resolve(vault, "Home/Heating/notes/boiler.md", fm)
    assert (m.note_class, m.validity, m.as_of, m.superseded_by) == ("state", "current", "2026-11-01", None)


def test_a_successor_link_makes_the_note_superseded(vault):
    m = resolve(vault, "Home/Heating/notes/boiler.md", {"class": "state", "superseded_by": "[[boiler-2026|new]]"})
    assert (m.validity, m.superseded_by) == ("superseded", "boiler-2026")


def test_a_note_without_kb_keys_has_no_labels(vault):
    m = resolve(vault, "Home/Heating/notes/boiler.md", {})
    assert (m.note_class, m.validity, m.as_of, m.superseded_by) == (None, None, None, None)


@pytest.mark.parametrize(
    "raw_type, expected",
    [
        ("research", "record"),
        ("meeting", "record"),
        ("log", "record"),
        ("review", "record"),
        ("decision", "record"),
        ("plan", "proposal"),
        ("brainstorm", "proposal"),
        ("spec", "proposal"),
        ("guide", "state"),
        ("reference", "state"),
        ("summary", "state"),
        ("drivers", "state"),
        ("note", None),
        ("template", None),
        ("entity", None),
        ("something-else", None),
        ("investigation", "record"),  # a raw type: the default follows the mapped type research
        ("how-to", "state"),  # mapped to guide
    ],
)
def test_class_defaults_from_the_normalized_type(vault, raw_type, expected):
    m = resolve(vault, "Home/Heating/notes/boiler.md", {"type": raw_type})
    assert m.note_class == expected
    assert m.validity is None  # a default class gives no validity


def test_class_defaults_from_a_path_type(vault):
    assert resolve(vault, "Home/Heating/SUMMARY.md", {}).note_class == "state"
    assert resolve(vault, "Home/Heating/notes/boiler.md", {}).note_class is None


def test_an_explicit_class_wins_over_the_default(vault):
    m = resolve(vault, "Home/Heating/notes/boiler.md", {"type": "research", "class": "State"})
    assert m.note_class == "state"


@pytest.mark.parametrize(
    "value, expected",
    [
        (["[[boiler|the boiler]]", "[[Home/Heating/pump#Flow]]"], ["boiler", "Home/Heating/pump"]),
        ("[[boiler]]", ["boiler"]),
        ("plain topic", ["plain topic"]),
        ([["boiler"], ["pump"]], ["boiler", "pump"]),  # unquoted wiki-links are nested lists in YAML
        (None, []),
        ([], []),
        ("none", []),  # no subject: the value none is not a link
        ("None", ["None"]),  # only the lower case none
    ],
)
def test_subject_targets_come_from_frontmatter(vault, value, expected):
    fm = {} if value is None else {"subject": value}
    assert resolve(vault, "Home/Heating/notes/boiler.md", fm).subject == expected


def test_summary_stage_does_not_reach_notes_outside_the_project(vault):
    (vault.root / "Home/Heating/SUMMARY.md").write_text("---\nstage: DONE\n---\n# Heating\n")
    assert meta(vault, "Home/Astro/Theory/Field Stop.md").status == "active"


def test_an_entity_note_gives_its_kind_and_aliases(vault):
    fm = {"type": "Entity", "kind": " Device ", "aliases": ["HP", "heat pump", " hp ", "Wärmepumpe"]}
    m = resolve(vault, "Home/Entities/Heat pump.md", fm)
    assert (m.entity, m.kind, m.aliases) == (True, "device", ["HP", "heat pump", "Wärmepumpe"])
    assert m.type == "note"  # the type label does not change


def test_a_single_alias_can_be_a_string(vault):
    m = resolve(vault, "Home/Entities/Heat pump.md", {"type": "entity", "aliases": "HP"})
    assert (m.entity, m.kind, m.aliases) == (True, None, ["HP"])


def test_a_note_that_is_not_an_entity_has_no_kind_or_aliases(vault):
    m = resolve(vault, "Home/Heating/notes/boiler.md", {"kind": "device", "aliases": ["b"]})
    assert (m.entity, m.kind, m.aliases) == (False, None, [])


def test_owner_and_supersedes_targets_come_from_frontmatter(vault):
    fm = {"owner": "[[Platform team|team]]", "supersedes": ["[[boiler-2025]]", "[[old/boiler-2024.md]]"]}
    m = resolve(vault, "Home/Heating/notes/boiler.md", fm)
    assert (m.owner, m.supersedes) == (["Platform team"], ["boiler-2025", "old/boiler-2024"])
