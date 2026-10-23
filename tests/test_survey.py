import json
from dataclasses import replace

import pytest

from ragnarok.cli import main
from ragnarok.config import dump_config
from ragnarok.survey import survey

EXTRA = {
    "Work/Top.md": "---\nkey: [unclosed\n---\n# Bad\n",
    "Work/Sales/Answers/a.md": "---\ntype: guide\nstatus: active\nproject: p1\n---\n# A\nRFC-2119, UTF-8 and OPS-5.\n",
    "Work/Sales/Answers/b.md": "---\ntype: guide\n---\n# B\nOPS-6 and OPS-5. SECRETBODY\n",
    "Work/Personal/People/Other.md": "---\ntype: secret-type\ncolor: red\n---\n# P\n",
    "Work/Sales/Answers/README.md": "# r\n",
    "Work/Sales/README.md": "# r\n",
    "Work/Devops/README.md": "# r\n",
}


@pytest.fixture
def cfg(vault):
    for rel, text in EXTRA.items():
        (vault.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault.root / rel).write_text(text)
    return vault


@pytest.fixture
def facts(cfg):
    return survey(cfg, "Work")


def test_counts(facts):
    assert facts["notes"] == 15
    assert facts["with_frontmatter"] == 4
    assert facts["malformed_frontmatter"] == ["Work/Top.md"]


def test_depth_and_folders(facts):
    assert facts["by_depth"] == {"0": 1, "1": 3, "2": 9, "3": 2}
    assert facts["level1"] == {"folders": {"Devops": 5, "Sales": 5, "Personal": 2, "SLA": 2}, "more": 0}
    assert facts["level2"]["folders"] == {
        "Sales/Answers": 4,
        "Devops/db-resize": 2,
        "Devops/_archive": 2,
        "Personal/People": 2,
        "SLA/Incidents": 1,
    }


def test_top_n_and_more(cfg):
    facts = survey(cfg, "Work", top=2)
    assert len(facts["level1"]["folders"]) == 2
    assert facts["level1"]["more"] == 2


def test_frontmatter_keys_and_values_skip_private(facts):
    assert facts["keys"] == {"type": 3, "description": 1, "status": 1, "project": 1, "color": 1}
    assert facts["values"]["type"] == {"guide": 2}
    assert facts["values"]["status"] == {"active": 1}
    assert facts["values"]["project"] == {"p1": 1}
    assert facts["values"]["area"] == {}
    assert "secret-type" not in json.dumps(facts)


def test_marker_candidates(facts):
    assert facts["marker_candidates"] == {"README.md": 3}
    assert facts["project_markers"] == {"DRIVERS.md": 0, "SUMMARY.md": 2}


def test_name_patterns(facts):
    assert facts["prefixes"] == {"research-": 1, "Q7-": 1}
    assert facts["date_prefixed"] == 1


def test_archive_folders(facts):
    assert facts["archive_folders"] == {"_archive": 2}


def test_ticket_prefixes_skip_non_tickets(facts):
    assert facts["tickets"] == {
        "OPS": {"keys": 3, "notes": 3},
        "FOO": {"keys": 1, "notes": 1},
        "TS": {"keys": 1, "notes": 1},
    }


def test_excluded_folders_are_not_walked(cfg):
    (cfg.root / "Work/node_modules/x").mkdir(parents=True)
    (cfg.root / "Work/node_modules/x/a.md").write_text("# a\n")
    (cfg.root / "Work/.obsidian").mkdir()
    (cfg.root / "Work/.obsidian/b.md").write_text("# b\n")
    assert survey(cfg, "Work")["notes"] == 15


def write_config(cfg, tmp_path, vaults):
    path = tmp_path / "config.toml"
    path.write_text(dump_config(replace(cfg, vaults=vaults), root=cfg.root, index_path=tmp_path / "i.db"))
    return str(path)


def test_cli_text_for_vault_not_in_config(cfg, tmp_path, capsys):
    config = write_config(cfg, tmp_path, [])
    assert main(["--config", config, "survey", "Work"]) == 0
    out = capsys.readouterr().out
    assert "15" in out
    assert "README.md" in out
    assert "_archive" in out
    assert "OPS" in out
    assert "SECRETBODY" not in out
    assert "secret-type" not in out


def test_cli_json(cfg, tmp_path, capsys):
    config = write_config(cfg, tmp_path, ["Work"])
    assert main(["--config", config, "survey", "Work", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["notes"] == 15


def test_cli_unknown_folder(cfg, tmp_path, capsys):
    config = write_config(cfg, tmp_path, ["Work"])
    assert main(["--config", config, "survey", "Nowhere"]) == 1
    assert "Nowhere" in capsys.readouterr().err


def test_level3_folders(facts):
    assert facts["level3"] == {"folders": {"Devops/_archive/db-migration": 2}, "more": 0}


def test_private_note_names_are_not_listed(cfg, tmp_path, capsys):
    (cfg.root / "Work/Personal/People/Jane-Doe.md").write_text("---\nkey: [unclosed\n---\n# J\n")
    (cfg.root / "Work/Personal/People/Zed-Quux.md").write_text("# Z\n")
    facts = survey(cfg, "Work")
    assert facts["notes"] == 17
    assert facts["malformed_frontmatter"] == ["Work/Top.md"]
    assert facts["malformed_private"] == 1
    assert facts["prefixes"] == {"research-": 1, "Q7-": 1}
    config = write_config(cfg, tmp_path, ["Work"])
    for extra in ([], ["--json"]):
        assert main(["--config", config, "survey", "Work", *extra]) == 0
        out = capsys.readouterr().out
        assert "Jane" not in out
        assert "Zed" not in out
