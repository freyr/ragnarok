import tarfile

import pytest

from ragnarok import backfill
from ragnarok.metadata import parse_note


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
