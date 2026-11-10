"""Local tests of the synthetic generator: claude -p and the deny list of this computer. CI skips them."""

import json
import tarfile
from pathlib import Path

import pytest

from ragnarok.bench import load_set
from ragnarok.cli import main
from ragnarok.synth import PROMPTS, SynthError, archive_hits, deny_path, freeze, read_deny

pytestmark = pytest.mark.local
FROZEN = sorted(PROMPTS.parent.glob("*/manifest.json"))


def test_a_vault_of_20_questions_goes_through_all_commands_with_claude(tmp_path, capsys):
    work = tmp_path / "work"
    assert main(["synth", "graph", "--seed", "101", "--questions", "20", "--out", str(work)]) == 0
    assert main(["synth", "render", str(work), "--jobs", "8"]) == 0
    assert main(["synth", "questions", str(work), "--jobs", "8"]) == 0
    code = main(["synth", "verify", str(work)])
    printed = capsys.readouterr().out
    print(printed)
    assert code == 0
    report = json.loads((work / "verify.json").read_text())
    assert report["passed"] and report["questions"]["kept"] >= 15
    out = tmp_path / "frozen"
    assert main(["synth", "freeze", str(work), "--name", "local-20", "--out", str(out)]) == 0
    bench_set = load_set(out / "questions.json")
    assert bench_set["kind"] == "synthetic" and len(bench_set["questions"]) == report["questions"]["kept"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["models"] and all(m.startswith("claude-") for m in manifest["models"])
    assert archive_hits(out / "vault.tar.gz", read_deny(deny_path())) == []


def test_the_privacy_check_fails_on_a_fixture_with_a_deny_word(tmp_path):
    term = read_deny(deny_path())[0]
    fixture = tmp_path / "fixture.tar.gz"
    note = tmp_path / "note.md"
    note.write_text(f"# A note\n\nThe text names {term} once.\n")
    with tarfile.open(fixture, "w:gz") as tar:
        tar.add(note, arcname="seed1/Area/note.md")
    assert archive_hits(fixture, [term]) == [("seed1/Area/note.md", term)]
    assert ("seed1/Area/note.md", term) in archive_hits(fixture, read_deny(deny_path()))


def test_freeze_refuses_a_vault_with_a_deny_word(tmp_path):
    from test_synth import Fake

    from ragnarok.synth import render, write_graph, write_questions

    term = read_deny(deny_path())[0]
    work = tmp_path / "work"
    graph = write_graph(work, 5, 20)
    fake = Fake(graph)
    render(work, fake, jobs=1)
    write_questions(work, fake, jobs=1)
    note = work / "vault" / graph["notes"][0]["path"]
    note.write_text(note.read_text() + f"\nA line with {term}.\n")
    with pytest.raises(SynthError, match="privacy"):
        freeze([work], "fixture", tmp_path / "out", deny_path())


@pytest.mark.parametrize("manifest", FROZEN, ids=[m.parent.name for m in FROZEN])
def test_each_frozen_set_holds_no_deny_word(manifest: Path):
    terms = read_deny(deny_path())
    folder = manifest.parent
    assert archive_hits(folder / "vault.tar.gz", terms) == []
    for path in folder.glob("*.json"):
        from ragnarok.synth import deny_hits

        assert deny_hits(path.read_text(), terms) == [], path.name
