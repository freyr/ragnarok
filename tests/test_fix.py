import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import frontmatter
import pytest
import yaml
from test_index import FakeEmbedder

from ragnarok import fix as fix_module
from ragnarok.cli import main
from ragnarok.config import Config, dump_config
from ragnarok.index import connect, update

FIXTURES = Path(__file__).parent / "fixtures" / "skills"
REC = "---\nclass: record\nas_of: 2026-01-01\n"
SRC = "V/Src/links.md"
SRC_TEXT = (
    REC + 'related: "[[Old/Heat pump guide]]"\n'
    "subject: ['[[Old/Heat pump guide]]']\n"
    "---\n"
    "See [[Old/Heat pump guide]] and ![[Old/Heat pump guide.md#Top|Guide]] and [[old/heat pump GUIDE^b1]].\n"
    "| a | [[ Old/Heat pump guide \\|alias]] |\n"
    "Code `[[Old/Heat pump guide]]` stays.\n"
    "```\n"
    "[[Old/Heat pump guide]]\n"
    "```\n"
    "Stays: [[Old/README]] [[Old/Twin]] [[Old/Only in W]] [[Old/Nowhere]]\n"
)
# the expected text is written by hand: each non-code occurrence outside the reference keys changes
SRC_FIXED = (
    REC + 'related: "[[Heat pump guide]]"\n'
    "subject: ['[[Old/Heat pump guide]]']\n"
    "---\n"
    "See [[Heat pump guide]] and ![[Heat pump guide#Top|Guide]] and [[Heat pump guide^b1]].\n"
    "| a | [[ Heat pump guide \\|alias]] |\n"
    "Code `[[Old/Heat pump guide]]` stays.\n"
    "```\n"
    "[[Old/Heat pump guide]]\n"
    "```\n"
    "Stays: [[Old/README]] [[Old/Twin]] [[Old/Only in W]] [[Old/Nowhere]]\n"
)
NOTES = {
    "V/Entities/Heat pump.md": "---\ntype: entity\nkind: device\n---\n# Heat pump\n",
    "V/Guides/Heat pump guide.md": REC + "---\n# Guide\n",
    "V/Docs/README.md": REC + "---\n",
    "V/A/Twin.md": REC + "---\n",
    "V/_archive/Twin.md": REC + "---\n",
    "V/Astro/Sub/Dashboard.md": REC + "---\n",
    "W/Notes/Only in W.md": REC + "---\n",
    SRC: SRC_TEXT,
    "V/Astro/src.md": REC + "---\nSee [[Old/Dashboard|the board]].\n",
    "V/Other/src.md": REC + "---\nSee [[Old/Dashboard]].\n",
    "V/Fixtures/assets/fx.md": REC + "---\nSee [[Old/Heat pump guide]].\n",
    "V/Y/fixable.md": REC + 'title: Foo: a \\ "b"\n---\n# Y\n',
    "V/Y/quoted.md": REC + 'title: "a": b\n---\n',
}
OLD = time.time() - 3600


@pytest.fixture
def fcfg(tmp_path, monkeypatch):
    root = tmp_path / "vaults"
    for rel, text in NOTES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        os.utime(path, (OLD, OLD))
    cfg = Config(
        root=root, index_path=tmp_path / "index.db", data_dir=tmp_path / "data", vaults=["V", "W"], dimension=8
    )
    conn = connect(cfg)
    update(cfg, conn, FakeEmbedder())
    conn.close()
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", lambda c: FakeEmbedder())
    return cfg


def config_file(cfg, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(dump_config(cfg, root=cfg.root, index_path=cfg.index_path, data_dir=cfg.data_dir))
    return config


def fix(cfg, tmp_path, capsys, *args):
    code = main(["--config", str(config_file(cfg, tmp_path)), "fix", *args])
    return code, capsys.readouterr()


def text(cfg, rel):
    return (cfg.root / rel).read_text()


def backups(cfg):
    folder = cfg.data_dir / "fix-backups"
    return sorted(folder.iterdir()) if folder.exists() else []


def log_rows(run_dir):
    with (run_dir / "log.tsv").open(newline="") as f:
        return list(csv.reader(f, delimiter="\t"))


def test_fix_links_changes_only_the_auto_links_and_keeps_the_link_parts(fcfg, tmp_path, capsys):
    code, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    assert code == 0, out.err
    assert text(fcfg, SRC) == SRC_FIXED
    assert text(fcfg, "V/Astro/src.md") == REC + "---\nSee [[Dashboard|the board]].\n"  # Dashboard in the area
    assert text(fcfg, "V/Other/src.md") == NOTES["V/Other/src.md"]  # Dashboard in another area
    assert text(fcfg, "V/Fixtures/assets/fx.md") == NOTES["V/Fixtures/assets/fx.md"]  # assets
    assert f"{SRC}: [[Old/Heat pump guide]] -> [[Heat pump guide]]" in out.out.splitlines()
    assert "links: 2 in 2 notes" in out.out


def test_fix_links_writes_a_backup_and_a_log_and_validates(fcfg, tmp_path, capsys):
    _, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    [run_dir] = backups(fcfg)
    assert (run_dir / SRC).read_text() == SRC_TEXT
    assert (run_dir / "V/Astro/src.md").read_text() == NOTES["V/Astro/src.md"]
    assert log_rows(run_dir) == [
        ["path", "old", "new"],
        ["V/Astro/src.md", "Old/Dashboard", "Dashboard"],
        [SRC, "Old/Heat pump guide", "Heat pump guide"],
    ]
    assert "fixed links that resolve to no note: 0" in out.out
    assert not list(fcfg.root.rglob("*.tmp"))


def test_fix_links_dry_run_prints_the_changes_and_writes_nothing(fcfg, tmp_path, capsys):
    before = {rel: text(fcfg, rel) for rel in NOTES}
    code, out = fix(fcfg, tmp_path, capsys, "links", "--dry-run")
    assert code == 0
    assert {rel: text(fcfg, rel) for rel in NOTES} == before
    assert backups(fcfg) == []
    assert out.out.splitlines()[:2] == [
        "V/Astro/src.md: [[Old/Dashboard]] -> [[Dashboard]]",
        f"{SRC}: [[Old/Heat pump guide]] -> [[Heat pump guide]]",
    ]
    assert "links: 2 in 2 notes (dry run, nothing written)" in out.out


def test_fix_links_area_and_exclude_area(fcfg, tmp_path, capsys):
    _, out = fix(fcfg, tmp_path, capsys, "links", "--dry-run", "--area", "Astro")
    assert "links: 1 in 1 notes" in out.out and "V/Astro/src.md" in out.out
    _, out = fix(fcfg, tmp_path, capsys, "links", "--dry-run", "--exclude-area", "Astro")
    assert "links: 1 in 1 notes" in out.out and "V/Astro/src.md" not in out.out


def test_fix_links_count_equals_the_fix_links_lines_of_the_worklist(fcfg, tmp_path, capsys):
    main(["--config", str(config_file(fcfg, tmp_path)), "health", "--worklist", "--json"])
    lines = [w for w in json.loads(capsys.readouterr().out)["worklist"] if w["fix"] == "ragnarok fix links"]
    _, out = fix(fcfg, tmp_path, capsys, "links", "--dry-run")
    assert len(lines) == 2
    assert f"links: {len(lines)} in 2 notes" in out.out


def test_a_non_unique_or_cross_vault_name_is_never_changed(fcfg, tmp_path, capsys):
    (fcfg.root / "V/Src/links.md").write_text(REC + "---\n[[Old/Twin]] [[Old/Only in W]] [[Old/README]]\n")
    os.utime(fcfg.root / SRC, (OLD, OLD))
    _, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    assert text(fcfg, SRC) == REC + "---\n[[Old/Twin]] [[Old/Only in W]] [[Old/README]]\n"
    assert "links: 1 in 1 notes" in out.out  # only the Dashboard of Astro


def test_the_mtime_guard_skips_a_note_changed_in_the_last_10_minutes(fcfg, tmp_path, capsys):
    recent = time.time() - 300
    os.utime(fcfg.root / SRC, (recent, recent))
    _, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    assert text(fcfg, SRC) == SRC_TEXT
    assert f"skipped: {SRC} (changed in the last 10 minutes)" in out.out.splitlines()
    assert text(fcfg, "V/Astro/src.md") == REC + "---\nSee [[Dashboard|the board]].\n"
    [run_dir] = backups(fcfg)
    assert not (run_dir / SRC).exists()
    assert [row[0] for row in log_rows(run_dir)] == ["path", "V/Astro/src.md"]


def test_the_re_read_check_skips_a_note_whose_old_link_is_gone(fcfg, tmp_path, capsys, monkeypatch):
    copy2 = shutil.copy2
    edited = REC + "---\nThe user removed the link.\n"

    def copy_then_edit(src, dst, **kw):
        result = copy2(src, dst, **kw)
        if str(src).endswith("links.md"):
            Path(src).write_text(edited)  # an editor writes between the backup and the write
        return result

    monkeypatch.setattr(shutil, "copy2", copy_then_edit)
    _, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    assert text(fcfg, SRC) == edited
    assert f"skipped: {SRC} (the old text is not there any more)" in out.out.splitlines()
    [run_dir] = backups(fcfg)
    assert [row[0] for row in log_rows(run_dir)] == ["path", "V/Astro/src.md"]


def test_a_note_deleted_after_the_plan_is_skipped_and_the_run_goes_on(fcfg, tmp_path, capsys, monkeypatch):
    baseline = fcfg.data_dir / "health" / "baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}")
    plan = fix_module.plan_links

    def plan_then_delete(*args):
        changes = plan(*args)
        (fcfg.root / SRC).unlink()  # another session deletes or moves the note
        return changes

    monkeypatch.setattr(fix_module, "plan_links", plan_then_delete)
    code, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    assert code == 0, out.err
    assert f"skipped: {SRC} (the note is not there any more)" in out.out.splitlines()
    assert text(fcfg, "V/Astro/src.md") == REC + "---\nSee [[Dashboard|the board]].\n"
    assert "validate: fixed links that resolve to no note: 0" in out.out
    assert "baseline: 1 notes updated" in out.out
    assert list(json.loads(baseline.read_text())) == ["V/Astro/src.md"]


def test_the_changed_notes_go_into_the_baseline(fcfg, tmp_path, capsys):
    baseline = fcfg.data_dir / "health" / "baseline.json"
    baseline.parent.mkdir(parents=True)
    old = {SRC: {"sha256": hashlib.sha256(SRC_TEXT.encode()).hexdigest(), "date": "2026-01-01", "source": "proposed"}}
    baseline.write_text(json.dumps(old))
    _, out = fix(fcfg, tmp_path, capsys, "links", "--vault", "V")
    data = json.loads(baseline.read_text())
    assert data[SRC]["sha256"] == hashlib.sha256(SRC_FIXED.encode()).hexdigest()
    assert data[SRC]["source"] == "update"
    assert data["V/Astro/src.md"]["source"] == "update"
    assert "baseline: 2 notes updated" in out.out


def test_edit_links_keeps_crlf_line_endings():
    old = "a [[Old/X]]\r\n```\r\n[[Old/X]]\r\n```\r\n"
    new, counts = fix_module.edit_links(old, {"old/x": "X"})
    assert new == "a [[X]]\r\n```\r\n[[Old/X]]\r\n```\r\n"
    assert counts == {"old/x": 1}


# --- the YAML fix ------------------------------------------------------------------------------------------------


def test_fix_yaml_quotes_the_value_and_escapes_backslash_and_quote(fcfg, tmp_path, capsys):
    code, out = fix(fcfg, tmp_path, capsys, "yaml", "--vault", "V")
    assert code == 0, out.err
    assert text(fcfg, "V/Y/fixable.md") == REC + 'title: "Foo: a \\\\ \\"b\\""\n---\n# Y\n'
    assert frontmatter.loads(text(fcfg, "V/Y/fixable.md"))["title"] == 'Foo: a \\ "b"'
    assert text(fcfg, "V/Y/quoted.md") == NOTES["V/Y/quoted.md"]  # a value that starts with a quote: the user
    [run_dir] = backups(fcfg)
    assert log_rows(run_dir) == [
        ["path", "old", "new"],
        ["V/Y/fixable.md", 'title: Foo: a \\ "b"', 'title: "Foo: a \\\\ \\"b\\""'],
    ]


@pytest.mark.parametrize("skill", ["handoff", "project"])
def test_fix_yaml_on_copies_of_the_fixture_skills(fcfg, tmp_path, capsys, skill):
    source = (FIXTURES / skill / "SKILL.md").read_text()
    with pytest.raises(yaml.YAMLError):
        frontmatter.loads(source)
    rel = f"V/Skills/{skill}/SKILL.md"
    path = fcfg.root / rel
    path.parent.mkdir(parents=True)
    path.write_text(source)
    os.utime(path, (OLD, OLD))
    _, out = fix(fcfg, tmp_path, capsys, "yaml", "--vault", "V", "--area", "Skills")
    fixed = frontmatter.loads(path.read_text())
    lines = source.split("\n")
    description = next(line for line in lines if line.startswith("description: ")).removeprefix("description: ")
    assert fixed.metadata == {"name": skill, "description": description}
    assert fixed.content == frontmatter.loads(source.replace(f"description: {description}\n", "")).content
    assert path.read_text().split("\n")[3:] == lines[3:]  # only the description line changed
    assert "yaml: 1 in 1 notes" in out.out
    assert "validate: fixed notes with bad YAML: 0" in out.out
