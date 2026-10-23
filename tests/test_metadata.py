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

    path = tmp_path / "copy.toml"
    path.write_text(dump_config(vault, root="vaults", data_dir="data"))
    copy = load_config(path)
    assert (copy.root, copy.data_dir) == (tmp_path / "vaults", tmp_path / "data")
    assert (copy.vault_rules, copy.jira_prefixes, copy.vaults) == (vault.vault_rules, vault.jira_prefixes, vault.vaults)


def test_unknown_rule_key_is_an_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[vault_rules.Work]\nproject_level = 2\n")
    with pytest.raises(TypeError):
        load_config(path)
