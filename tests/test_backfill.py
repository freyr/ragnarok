import json
import tarfile

import pytest

from ragnarok import backfill
from ragnarok.cli import main
from ragnarok.metadata import TYPES, parse_note


def rows_by_path(proposal):
    return {r["path"]: r for r in proposal["rows"]}


def test_propose_lists_only_the_missing_keys(vault):
    rows = rows_by_path(backfill.propose(vault, "Home"))
    car = rows["Home/Bike/shortlist.md"]
    assert car["set"] == {"type": "note", "description": ""}
    boiler = rows["Home/Heating/notes/boiler.md"]
    assert boiler["set"] == {
        "project": "Heating",
        "area": "heating",
        "type": "note",
        "status": "active",
        "description": "",
    }
    assert boiler["source"] == {"project": "path", "area": "path", "type": "default", "status": "path"}
    assert rows["Home/Budget/2026.md"]["private"] is True
    assert boiler["private"] is False
    assert all(r.startswith("Home/") for r in rows)


def test_propose_leaves_a_status_from_the_summary_stage_out(vault):
    (vault.root / "Home/Heating/SUMMARY.md").write_text("---\nstage: DONE\n---\n# Heating\n")
    boiler = rows_by_path(backfill.propose(vault, "Home"))["Home/Heating/notes/boiler.md"]
    assert "status" not in boiler["set"] and "status" not in boiler["source"]
    (vault.root / "Home/Heating/notes/boiler.md").write_text("---\nstatus: done\n---\n# Boiler\n")
    boiler = rows_by_path(backfill.propose(vault, "Home"))["Home/Heating/notes/boiler.md"]
    assert "status" not in boiler["set"]


def test_propose_skips_complete_notes_and_instruction_files(vault):
    (vault.root / "Studio/api-gateway/CLAUDE.md").write_text("# rules\n")
    (vault.root / "Studio/api-gateway/done.md").write_text(
        "---\nproject: api-gateway\narea: studio\ntype: note\nstatus: active\ndescription: Done.\n---\n# D\n"
    )
    rows = rows_by_path(backfill.propose(vault, "Studio"))
    assert list(rows) == ["Studio/api-gateway/docs/brainstorms/2026-03-01-x.md"]


def test_propose_lists_malformed_frontmatter_apart(vault):
    (vault.root / "Studio/api-gateway/bad.md").write_text("---\ntitle: refactor: x\n---\n# B\n")
    (vault.root / "Studio/api-gateway/indented.md").write_text("  ---\ntitle: x\n---\n# I\n")
    proposal = backfill.propose(vault, "Studio")
    assert proposal["malformed"] == ["Studio/api-gateway/bad.md", "Studio/api-gateway/indented.md"]
    assert not set(proposal["malformed"]) & set(rows_by_path(proposal))


def test_propose_matches_a_project_for_shared_docs(vault):
    (vault.root / "Clients/agents-platform").mkdir()
    (vault.root / "Clients/agents-platform/SUMMARY.md").write_text("# Agents\n")
    (vault.root / "Clients/docs/plans/2026-03-14-agents-platform-design.md").write_text("# Design\n")
    row = rows_by_path(backfill.propose(vault, "Clients"))["Clients/docs/plans/2026-03-14-agents-platform-design.md"]
    assert row["uncertain"] is True
    assert row["set"]["project"] == "agents-platform"
    assert row["source"]["project"] == "match"


def test_propose_needs_two_common_words_for_a_match(vault):
    (vault.root / "Clients/agents").mkdir()
    (vault.root / "Clients/agents/SUMMARY.md").write_text("# Agents\n")
    row = rows_by_path(backfill.propose(vault, "Clients"))["Clients/docs/plans/2026-03-13-feat-agents-plan.md"]
    assert (row["uncertain"], row["set"]["project"]) == (False, "none")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("# T\nbody\n", "---\nproject: p\n---\n# T\nbody\n"),
        ("---\na: 1\n---\n# T\n", "---\na: 1\nproject: p\n---\n# T\n"),
        ("---\n---\n# T\n", "---\nproject: p\n---\n# T\n"),
        ("---\r\na: 1\r\n---\r\n# T\r\n", "---\r\na: 1\r\nproject: p\r\n---\r\n# T\r\n"),
    ],
)
def test_insert_keys_changes_nothing_else(text, expected):
    assert backfill.insert_keys(text, {"project": "p"}) == expected


def test_apply_archives_first_and_adds_only_missing_keys(vault):
    proposal = backfill.propose(vault, "Home")
    rows = rows_by_path(proposal)
    rows["Home/Heating/notes/boiler.md"]["set"]["description"] = "How the boiler follows the weather curve."
    rows["Home/Astro/Theory/Field Stop.md"]["approve"] = False
    (vault.root / "Home/Bike/shortlist.md").write_text("# changed by the user\n")
    before = (vault.root / "Home/Heating/notes/boiler.md").read_text()

    result = backfill.apply(vault, "Home", proposal)

    with tarfile.open(result.archive) as tar:
        names = set(tar.getnames())
        assert tar.extractfile("Home/Heating/notes/boiler.md").read().decode() == before
    assert "Home/Teaching/flashcards/node_modules/pkg/README.md" not in names

    after = (vault.root / "Home/Heating/notes/boiler.md").read_text()
    fm, body = parse_note(after)
    assert fm == {
        "project": "Heating",
        "area": "heating",
        "type": "note",
        "status": "active",
        "description": "How the boiler follows the weather curve.",
    }
    assert after.endswith(before)
    assert (vault.root / "Home/Astro/Theory/Field Stop.md").read_text() == "# FS\nES-82 eyepiece.\n"
    assert "Home/Bike/shortlist.md: changed after the proposal" in result.skipped
    # an empty description is not written
    assert "description" not in parse_note((vault.root / "Home/Heating/SUMMARY.md").read_text())[0]


def test_apply_refuses_the_proposal_of_another_vault(vault):
    with pytest.raises(ValueError):
        backfill.apply(vault, "Studio", backfill.propose(vault, "Home"))


def test_coverage_counts_the_schema_keys(vault):
    share = backfill.coverage(vault)["Home"]
    assert share["notes"] == 6
    assert (share["project"], share["status"], share["description"]) == (0.17, 0.17, 0.0)


def test_export_lists_only_public_rows_that_need_a_description(vault):
    (vault.root / "Home/Heating/router.md").write_text("# Router\nadmin password: hunter2hunter\nLine two.\n")
    (vault.root / "Home/Heating/titled.md").write_text("---\ntitle: Pump list\n---\nBody text.\n")
    (vault.root / "Home/Heating/plain.md").write_text("Just text.\n")
    (vault.root / "Home/Heating/long.md").write_text("# Long\n" + "word " * 1000 + "\n")
    proposal = backfill.propose(vault, "Home")
    rows_by_path(proposal)["Home/Bike/shortlist.md"]["set"].pop("description")

    items = backfill.export_rows(vault, proposal)

    by_path = {i["path"]: i for i in items}
    assert "Home/Budget/2026.md" not in by_path
    assert "Home/Password/bank.md" not in by_path
    assert "Home/Bike/shortlist.md" not in by_path
    assert by_path["Home/Heating/notes/boiler.md"] == {
        "path": "Home/Heating/notes/boiler.md",
        "title": "Boiler",
        "type": "note",
        "excerpt": "# Boiler",
    }
    assert by_path["Home/Heating/titled.md"]["title"] == "Pump list"
    assert by_path["Home/Heating/titled.md"]["excerpt"] == "Body text."
    assert by_path["Home/Heating/plain.md"]["title"] == "plain"
    assert "hunter2hunter" not in by_path["Home/Heating/router.md"]["excerpt"]
    assert "[REDACTED]" in by_path["Home/Heating/router.md"]["excerpt"]
    assert len(by_path["Home/Heating/long.md"]["excerpt"]) <= 2000


def test_cli_export_writes_the_file_next_to_the_proposal(vault, monkeypatch, capsys):
    backfill.write_proposal(vault, "Home", backfill.propose(vault, "Home"))
    monkeypatch.setattr("ragnarok.cli.load_config", lambda *a, **k: vault)
    assert main(["backfill", "export", "Home"]) == 0
    out = vault.data_dir / "proposals" / "describe-Home.json"
    data = json.loads(out.read_text())
    assert str(out) in capsys.readouterr().out
    assert sorted(data["types"]) == sorted(TYPES)
    assert "25 words" in data["instructions"]
    assert any(i["path"] == "Home/Heating/notes/boiler.md" for i in data["rows"])


def merge_fixture(vault):
    proposal = backfill.propose(vault, "Home")
    return proposal, rows_by_path(proposal)


def test_merge_sets_descriptions_and_valid_default_types(vault):
    proposal, rows = merge_fixture(vault)
    boiler = rows["Home/Heating/notes/boiler.md"]
    hash_before, approve_before = boiler["hash"], boiler["approve"]
    long_text = " ".join(["w"] * 26)
    descriptions = {
        "Home/Heating/notes/boiler.md": {"description": "  How the boiler\n follows the curve. ", "type": "guide"},
        "Home/Heating/SUMMARY.md": {"description": "Heating summary.", "type": "log"},
        "Home/Astro/Theory/Field Stop.md": {"description": long_text, "type": "bogus"},
        "Home/Bike/shortlist.md": {"description": "Bike list.", "type": "log"},
        "Home/Budget/2026.md": {"description": "Secret budget."},
        "Home/Nowhere.md": {"description": "No row."},
    }
    result = backfill.merge_descriptions(proposal, descriptions)

    assert boiler["set"]["description"] == "How the boiler follows the curve."
    assert boiler["set"]["type"] == "guide"
    assert boiler["source"]["type"] == "claude"
    assert (boiler["hash"], boiler["approve"]) == (hash_before, approve_before)
    assert rows["Home/Heating/SUMMARY.md"]["set"]["type"] == "summary"  # the type has a rule: not the default
    assert rows["Home/Heating/SUMMARY.md"]["source"]["type"] != "claude"
    assert rows["Home/Bike/shortlist.md"]["set"]["description"] == "Bike list."
    assert rows["Home/Bike/shortlist.md"]["set"]["type"] == "log"
    assert rows["Home/Budget/2026.md"]["set"]["description"] == ""
    assert result.unknown == ["Home/Nowhere.md"]
    assert result.too_long == ["Home/Astro/Theory/Field Stop.md"]
    assert result.types == 2
    assert result.merged == 4
    assert "Home/Password/bank.md" not in result.missing
    assert "Home/Heating/SUMMARY.md" not in result.missing
    assert "Home/Budget/2026.md" not in result.missing


def test_merge_reports_missing_and_redacts(vault):
    proposal, rows = merge_fixture(vault)
    descriptions = {"Home/Heating/notes/boiler.md": {"description": "Login with password: hunter2hunter here."}}
    result = backfill.merge_descriptions(proposal, descriptions)
    text = rows["Home/Heating/notes/boiler.md"]["set"]["description"]
    assert "hunter2hunter" not in text
    assert result.redacted == 1
    assert "Home/Heating/SUMMARY.md" in result.missing
    assert "Home/Heating/notes/boiler.md" not in result.missing


def test_cli_merge_writes_the_proposal_and_fails_on_bad_json(vault, monkeypatch, tmp_path, capsys):
    backfill.write_proposal(vault, "Home", backfill.propose(vault, "Home"))
    monkeypatch.setattr("ragnarok.cli.load_config", lambda *a, **k: vault)
    good = tmp_path / "d1.json"
    good.write_text(json.dumps({"Home/Heating/notes/boiler.md": {"description": "The boiler."}}))
    more = tmp_path / "d2.json"
    more.write_text(json.dumps({"Home/Heating/SUMMARY.md": {"description": "The heating."}}))
    assert main(["backfill", "merge", "Home", str(good), str(more)]) == 0
    rows = rows_by_path(backfill.read_proposal(vault, "Home"))
    assert rows["Home/Heating/notes/boiler.md"]["set"]["description"] == "The boiler."
    assert rows["Home/Heating/SUMMARY.md"]["set"]["description"] == "The heating."
    assert "missing" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert main(["backfill", "merge", "Home", str(bad)]) == 1


def test_export_redacts_a_key_block_that_crosses_the_cut(vault):
    key = "-----BEGIN RSA PRIVATE KEY-----\n" + "MIIEowIBAAKCAQEAxyz\n" * 30 + "-----END RSA PRIVATE KEY-----\n"
    (vault.root / "Home/Heating/keys.md").write_text("# Keys\n" + "line\n" * 30 + key)
    items = backfill.export_rows(vault, backfill.propose(vault, "Home"))
    excerpt = next(i for i in items if i["path"] == "Home/Heating/keys.md")["excerpt"]
    assert "MIIEowIBAAKCAQEAxyz" not in excerpt
    assert "BEGIN RSA PRIVATE KEY" not in excerpt


def test_merge_checks_only_the_rows_that_it_writes(vault):
    proposal, rows = merge_fixture(vault)
    rows["Home/Bike/shortlist.md"]["set"].pop("description")
    text = "password: hunter2hunter " + " ".join(["w"] * 30)
    result = backfill.merge_descriptions(proposal, {"Home/Bike/shortlist.md": {"description": text}})
    assert result.too_long == []
    assert result.redacted == 0
    assert result.merged == 0


@pytest.mark.parametrize(
    "content",
    [
        {"Home/Heating/SUMMARY.md": "text"},
        {"Home/Heating/SUMMARY.md": {"description": 5}},
        {"Home/Heating/SUMMARY.md": {"description": "ok", "type": ["guide"]}},
        ["x"],
    ],
)
def test_cli_merge_rejects_a_bad_entry_and_keeps_the_proposal(vault, monkeypatch, tmp_path, capsys, content):
    backfill.write_proposal(vault, "Home", backfill.propose(vault, "Home"))
    before = backfill.proposal_path(vault, "Home").read_text()
    monkeypatch.setattr("ragnarok.cli.load_config", lambda *a, **k: vault)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(content))
    assert main(["backfill", "merge", "Home", str(bad)]) == 1
    assert "bad.json" in capsys.readouterr().err
    assert backfill.proposal_path(vault, "Home").read_text() == before
