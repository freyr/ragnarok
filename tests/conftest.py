import pytest

from ragnarok.config import Config, VaultRules

NOTES = {
    # Work: area = level 1, project = level 2 (marker), _archive = archived
    "Work/Devops/db-resize/SUMMARY.md": "---\ndescription: Resize the database servers.\n---\n# S\n",
    "Work/Devops/db-resize/research-cpu.md": "# CPU\nSee OPS-77 and TS-9 and FOO-1.\n",
    "Work/Devops/_archive/db-migration/SUMMARY.md": "# old\n",
    "Work/Devops/_archive/db-migration/Q7-cost.md": "# Q7\n",
    "Work/Sales/Answers/Security.md": "# Security\n",
    "Work/Personal/People/Someone.md": "# Person\n",
    "Work/SLA/Incidents/2026-01-01.md": "# Incident\n",
    "Work/SLA/2026-Q1 review.md": "# Review\n",
    # Clients: client and topic folders, Acme projects, shared docs/
    "Clients/Acme/01-Projects/fsm/notes.md": "# FSM\n",
    "Clients/Acme/99-Templates/Meeting.md": "# T\n",
    "Clients/Queue-Research/Block-04.md": "---\ntype: knowledge-base\n---\n# B4\n",
    "Clients/docs/plans/2026-03-13-feat-agents-plan.md": "---\ntype: feat\n---\n# Plan\nFR-001 is a requirement.\n",
    "Clients/Root guide.md": "# Guide\n",
    # Studio: project = level 1 folder
    "Studio/api-gateway/docs/brainstorms/2026-03-01-x.md": "---\njira: CORE-123\n---\n# X\n",
    # Home: closest marker folder, else level 2, else level 1
    "Home/Heating/SUMMARY.md": "# Heating\n",
    "Home/Heating/notes/boiler.md": "# Boiler\n",
    "Home/Astro/Theory/Field Stop.md": "# FS\nES-82 eyepiece.\n",
    "Home/Bike/shortlist.md": "---\nproject: bike-2026\narea: bikes\nstatus: archived\n---\n# Cars\n",
    "Home/Password/bank.md": "# secret\n",
    "Home/Budget/2026.md": "# budget\n",
    # excluded folders
    "Home/Teaching/flashcards/node_modules/pkg/README.md": "# pkg\n",
    "Clients/.claude/agents/ddd.md": "# agent\n",
    "Home/.obsidian/workspace.md": "# ws\n",
}


@pytest.fixture
def vault(tmp_path):
    for rel, text in NOTES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return Config(
        root=tmp_path,
        index_path=tmp_path / "index.db",
        vaults=["Work", "Clients", "Studio", "Home"],
        private_patterns=["Work/Personal/**", "Work/SLA/Incidents/**", "Home/Password/**", "Home/Budget/**"],
        jira_prefixes=["OPS", "CORE"],
        vault_rules={
            "Work": VaultRules(project_fallback=False, jira_prefixes=["TS", "FR"]),
            "Clients": VaultRules(
                project_depth=1, project_parents=["Acme/01-Projects"], ignore_folders=["docs", "todos"]
            ),
            "Studio": VaultRules(area="studio", project_depth=1),
        },
    )
