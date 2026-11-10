"""The synthetic generator: the fact graph, the verifier, the strata and the freeze (no LLM: a fake writer)."""

import hashlib
import json
import re
import tarfile

import pytest

from ragnarok.bench import load_set
from ragnarok.cli import main
from ragnarok.synth import (
    ATTRIBUTES,
    PROMPTS,
    SynthError,
    build_graph,
    candidates,
    deny_hits,
    forbidden_hits,
    freeze,
    graph_hash,
    property_failures,
    render,
    stratum_counts,
    verify,
    world_config,
    write_graph,
    write_questions,
)

# --- the fact graph ------------------------------------------------------------------------------------------------


def test_one_seed_gives_the_same_graph_hash_twice():
    first = build_graph(world_config(7, 40))
    second = build_graph(world_config(7, 40))
    assert first["hash"] == second["hash"]
    assert first == second
    # the hash is the SHA-256 of the canonical JSON of the graph without the hash
    body = {k: v for k, v in first.items() if k != "hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert first["hash"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert graph_hash(first) == first["hash"]
    assert build_graph(world_config(8, 40))["hash"] != first["hash"]


@pytest.mark.parametrize(
    "n, counts",
    [
        # the shares 20/20/20/15/10/10/5 of the spec
        (100, (20, 20, 20, 15, 10, 10, 5)),
        (40, (8, 8, 8, 6, 4, 4, 2)),
        # 4, 4, 4, 3, 2, 2, 1 exactly
        (20, (4, 4, 4, 3, 2, 2, 1)),
    ],
)
def test_stratum_counts_follow_the_fixed_shares(n, counts):
    names = ("direct", "lexical-gap", "deep-section", "distractor", "outdated", "continuation", "multi-hop")
    assert stratum_counts(n) == dict(zip(names, counts, strict=True))


@pytest.fixture(scope="module")
def graph():
    return build_graph(world_config(11, 40))


def test_the_graph_has_the_strata_shares_and_equal_halves(graph):
    by_stratum = {}
    for q in graph["questions"]:
        by_stratum.setdefault(q["stratum"], []).append(q["half"])
    assert {s: len(h) for s, h in by_stratum.items()} == {
        "direct": 8,
        "lexical-gap": 8,
        "deep-section": 8,
        "distractor": 6,
        "outdated": 4,
        "continuation": 4,
        "multi-hop": 2,
    }
    for halves in by_stratum.values():
        assert halves.count("dev") == halves.count("test")
        assert set(halves) == {"dev", "test"}


def test_the_property_set_of_each_question_matches_exactly_one_subject(graph):
    for q in graph["questions"]:
        assert len(q["props"]) >= 2
        cls = next(s["class"] for s in graph["subjects"] if s["id"] == q["subject"])
        fits = [
            s["id"]
            for s in graph["subjects"]
            if s["class"] == cls and all(s["props"].get(k) == v for k, v in q["props"].items())
        ]
        assert fits == [q["subject"]], q["id"]
    assert property_failures(graph) == []


def test_the_uniqueness_check_reports_a_property_set_of_two_subjects(graph):
    broken = json.loads(json.dumps(graph))
    q = broken["questions"][0]
    subject = next(s for s in broken["subjects"] if s["id"] == q["subject"])
    broken["subjects"].append({**subject, "id": "twin", "name": "Twin thing", "word": "Twin"})
    failures = property_failures(broken)
    assert [qid for qid, _ in failures] == [q["id"]]


def test_each_value_is_unique_and_no_value_is_part_of_another(graph):
    tokens = [f["token"] for f in graph["facts"] if not f["bridge"]]
    assert len(tokens) == len(set(tokens))
    for a in tokens:
        for b in tokens:
            assert a == b or a not in b


def test_each_fact_is_in_exactly_one_note_and_each_path_starts_with_the_vault(graph):
    notes = {n["id"]: n for n in graph["notes"]}
    for f in graph["facts"]:
        assert f["note"] in notes
        assert sum(f["id"] in n["facts"] for n in graph["notes"]) == 1
    paths = [n["path"] for n in graph["notes"]]
    assert len(paths) == len(set(paths))
    assert all(p.startswith("seed11/") and p.endswith(".md") for p in paths)


def _note(graph, path):
    return next(n for n in graph["notes"] if n["path"] == path)


def _fact(graph, fid):
    return next(f for f in graph["facts"] if f["id"] == fid)


def test_the_notes_of_each_stratum(graph):
    subjects = {s["id"]: s for s in graph["subjects"]}
    for q in graph["questions"]:
        gold = _note(graph, q["cluster"])
        fact = _fact(graph, q["fact"])
        assert fact["note"] == gold["id"] and fact["current"]
        assert q["cluster"] == q["expected"][0]
        if q["stratum"] == "distractor":
            others = [n for n in graph["notes"] if n["role"] == "distractor" and n["id"] in q["notes"]]
            assert 1 <= len(others) <= 3
            for n in others:
                s, g = subjects[n["subject"]], subjects[fact["subject"]]
                assert s["id"] != g["id"] and s["class"] == g["class"] and s["area"] == g["area"]
                assert list(s["props"].values())[0] == list(g["props"].values())[0]  # the same kind
                assert [_fact(graph, f)["attribute"] for f in n["facts"]] == [fact["attribute"]]
        elif q["stratum"] == "outdated":
            old = [n for n in graph["notes"] if n["role"] == "outdated" and n["id"] in q["notes"]]
            assert len(old) == 1 and old[0]["status"] == "archived"
            old_fact = _fact(graph, old[0]["facts"][0])
            assert (old_fact["subject"], old_fact["attribute"]) == (fact["subject"], fact["attribute"])
            assert not old_fact["current"] and old_fact["token"] != fact["token"]
        elif q["stratum"] == "deep-section":
            assert gold["long"] and 3000 <= gold["words"] <= 10000
            assert fact["heading"] == gold["headings"][-1]
        elif q["stratum"] == "continuation":
            assert len(q["expected"]) == 1 and q["expected"][0].endswith("/SUMMARY.md")
            assert subjects[q["subject"]]["class"] == "project"
        elif q["stratum"] == "multi-hop":
            assert len(q["expected"]) == 2
            bridge = _note(graph, q["expected"][1])
            bridge_fact = _fact(graph, q["bridge"])
            assert bridge_fact["note"] == bridge["id"] and bridge_fact["bridge"]
            assert bridge_fact["value"] == subjects[fact["subject"]]["name"]
            assert q["subject"] == bridge["subject"]
    for project in (s for s in graph["subjects"] if s["class"] == "project"):
        names = {n["path"].rsplit("/", 1)[1] for n in graph["notes"] if n["subject"] == project["id"]}
        assert {"DRIVERS.md", "SUMMARY.md"} <= names


def test_the_forbidden_words_of_a_question(graph):
    subjects = {s["id"]: s for s in graph["subjects"]}
    for q in graph["questions"]:
        fact = _fact(graph, q["fact"])
        gold = _note(graph, q["cluster"])
        subject = subjects[q["subject"]]
        for word in (subject["name"], subject["word"], gold["title"], fact["heading"], fact["value"], fact["token"]):
            assert word in q["forbidden"], (q["id"], word)
        if q["stratum"] == "multi-hop":
            # the device is the bridge: neither its name nor its kind may be in the question
            device = subjects[fact["subject"]]
            assert device["name"] in q["forbidden"] and device["props"]["kind"] in q["forbidden"]


# --- a fake writer instead of claude -p ----------------------------------------------------------------------------


class Fake:
    """Writes each note with the values of its prompt at the end, each question from its properties, and answers the
    subject check with the candidate whose properties the question names. `omit` holds the titles of notes that it
    always writes without values."""

    def __init__(self, graph, omit=()):
        self.values = [f["value"] for f in graph["facts"]]
        self.omit = set(omit)
        self.calls = 0

    def __call__(self, prompt, model):
        self.calls += 1
        if prompt.startswith("You write one note"):
            title = re.search(r"^Title of the note: (.+)$", prompt, re.M).group(1)
            words = int(re.search(r"about (\d+) words", prompt).group(1))
            headings = re.findall(r"^## (.+)$", prompt, re.M)
            body = "\n\n".join(f"## {h}\n\n" + "lorem " * (words // max(len(headings), 1)) for h in headings)
            if title not in self.omit:
                about = re.search(r"^The note is about: (.+)$", prompt, re.M).group(1)
                body += f"\n\nThis is {about}."
                body += "\n\n" + "\n".join(f"The value is {v}." for v in self.values if v in prompt)
            return body, "fake-model"
        if prompt.startswith("You write one search question"):
            props = re.findall(r"^- [^:\n]+: (.+)$", prompt.split("these properties:")[1].split("Rules:")[0], re.M)
            return "Which one is " + ", ".join(props) + "?", "fake-model"
        question = re.search(r"^Question: (.+)$", prompt, re.M).group(1)
        best, score = "NONE", -1
        for line in re.findall(r"^- (.+)$", prompt.split("Subjects:")[1], re.M):
            name, props = line.split(" | ", 1)
            values = [p.split(": ", 1)[1] for p in props.split("; ")]
            hits = sum(bool(re.search(rf"\b{re.escape(v)}\b", question)) for v in values)
            if hits > score:
                best, score = name, hits
        return best, "fake-model"


@pytest.fixture
def work(tmp_path):
    out = tmp_path / "work"
    graph = write_graph(out, 5, 20)
    return out, graph


def _written(out):
    render_state = json.loads((out / "notes.json").read_text())
    written = json.loads((out / "written.json").read_text())
    return render_state, written


def test_render_questions_and_verify_pass_with_a_correct_writer(work):
    out, graph = work
    fake = Fake(graph)
    render(out, fake, jobs=2)
    write_questions(out, fake, jobs=2)
    report = verify(out)
    assert report["failures"] == []
    assert report["passed"]
    assert report["questions"]["kept"] == 20
    assert report["notes"]["kept"] == len(graph["notes"])
    assert json.loads((out / "verify.json").read_text()) == report
    # a long note is written in parts, each one call
    long_notes = [n for n in graph["notes"] if n["long"]]
    assert fake.calls > len(graph["notes"]) + len(long_notes) + 2 * 20 - 1
    vault = out / "vault"
    text = (vault / graph["notes"][0]["path"]).read_text()
    assert text.startswith("---\n") and f"\n# {graph['notes'][0]['title']}\n" in text


def test_a_failed_note_is_written_again_twice_and_then_removed_with_its_question(work):
    out, graph = work
    q = next(q for q in graph["questions"] if q["stratum"] == "direct")
    gold = _note(graph, q["cluster"])
    fake = Fake(graph, omit=[gold["title"]])
    render(out, fake, jobs=1)
    write_questions(out, fake, jobs=1)
    render_state, written = _written(out)
    assert render_state[gold["id"]]["status"] == "removed"
    assert render_state[gold["id"]]["attempts"] == 3
    assert written[q["id"]]["status"] == "removed"
    assert not (out / "vault" / gold["path"]).exists()
    report = verify(out)
    assert report["passed"]
    assert report["removed"] == {"notes": [gold["path"]], "questions": [q["id"]]}
    assert report["questions"]["kept"] == 19


def test_verify_reports_each_failed_check_of_a_broken_fixture(work):
    out, graph = work
    fake = Fake(graph)
    render(out, fake, jobs=2)
    write_questions(out, fake, jobs=2)
    vault = out / "vault"
    by_stratum = {}
    for q in graph["questions"]:
        by_stratum.setdefault(q["stratum"], []).append(q)

    def path(q, i=0):
        return vault / q["expected"][i]

    # 1. a gold value is missing from its gold note
    direct = by_stratum["direct"][0]
    token = _fact(graph, direct["fact"])["token"]
    path(direct).write_text(path(direct).read_text().replace(token, "gone"))
    # 2. a gold value is in another current note
    other = by_stratum["direct"][1]
    p = path(other)
    p.write_text(p.read_text() + f"\nAlso {_fact(graph, by_stratum['lexical-gap'][0]['fact'])['token']}.\n")
    # 3. a distractor note holds the gold value
    dq = by_stratum["distractor"][0]
    distractor = next(n for n in graph["notes"] if n["role"] == "distractor" and n["id"] in dq["notes"])
    p = vault / distractor["path"]
    p.write_text(p.read_text() + f"\n{_fact(graph, dq['fact'])['token']}\n")
    # 4. a long note that is too short, with the deep value at the start
    deep = by_stratum["deep-section"][0]
    p = path(deep)
    p.write_text(f"# x\n{_fact(graph, deep['fact'])['token']}\nshort\n")
    # 5. a question with a forbidden word, 6. a question that the blind check resolves to another subject
    written = json.loads((out / "written.json").read_text())
    lex = by_stratum["lexical-gap"][1]
    written[lex["id"]]["question"] += " " + lex["forbidden"][0]
    cont = by_stratum["continuation"][0]
    written[cont["id"]]["answer"] = "NONE"
    (out / "written.json").write_text(json.dumps(written))
    # 7. a note file is missing
    os_note = by_stratum["outdated"][0]
    old = next(n for n in graph["notes"] if n["role"] == "outdated" and n["id"] in os_note["notes"])
    (vault / old["path"]).unlink()
    # 8. a property set that matches two subjects; the graph file no longer has its hash
    g = json.loads((out / "graph.json").read_text())
    subject = next(s for s in g["subjects"] if s["id"] == g["questions"][0]["subject"])
    g["subjects"].append({**subject, "id": "twin", "name": "Twin thing", "word": "Twin"})
    (out / "graph.json").write_text(json.dumps(g))

    report = verify(out)
    assert not report["passed"]
    found = {(f["check"], f["id"]) for f in report["failures"]}
    expected = {
        ("gold-in-note", direct["expected"][0]),
        ("gold-elsewhere", other["expected"][0]),
        ("distractor-holds-gold", distractor["path"]),
        ("long-note-length", deep["expected"][0]),
        ("deep-fact-position", deep["expected"][0]),
        ("property-in-note", deep["expected"][0]),  # the short text no longer states the properties
        ("forbidden-word", lex["id"]),
        ("subject-check", cont["id"]),
        ("note-missing", old["path"]),
        ("unique-properties", g["questions"][0]["id"]),
        ("graph-hash", "graph.json"),
    }
    assert expected <= found
    assert {c for c, _ in found} == {c for c, _ in expected}


def test_verify_reports_the_strata_and_the_halves(work):
    out, graph = work
    g = json.loads((out / "graph.json").read_text())
    next(q for q in g["questions"] if q["stratum"] == "direct")["stratum"] = "lexical-gap"
    next(q for q in g["questions"] if q["stratum"] == "outdated")["half"] = "both"
    (out / "graph.json").write_text(json.dumps(g))
    checks = {(f["check"], f["id"]) for f in verify(out)["failures"]}
    assert {("strata", "direct"), ("strata", "lexical-gap"), ("halves", "outdated")} <= checks


# --- the freeze ----------------------------------------------------------------------------------------------------


@pytest.fixture
def done(work, tmp_path):
    out, graph = work
    fake = Fake(graph)
    render(out, fake, jobs=2)
    write_questions(out, fake, jobs=2)
    deny = tmp_path / "deny.txt"
    deny.write_text("# test terms\nqqzzxx\nnot-a-word-here\n")
    return out, graph, deny


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_writes_the_archive_the_set_the_report_and_the_manifest(done, tmp_path):
    out, graph, deny = done
    frozen = tmp_path / "bench" / "synthetic" / "pilot"
    manifest = freeze([out], "pilot", frozen, deny)
    names = {p.relative_to(frozen).as_posix() for p in frozen.rglob("*") if p.is_file()}
    assert {"vault.tar.gz", "questions.json", "verify.json", "manifest.json", "world-seed5.json"} <= names
    assert {"graph-seed5.json", "prompts/note.md", "prompts/question.md", "prompts/subject-check.md"} <= names
    on_disk = json.loads((frozen / "manifest.json").read_text())
    assert on_disk == manifest
    assert set(manifest["files"]) == names - {"manifest.json"}
    for rel, sha in manifest["files"].items():
        assert _sha(frozen / rel) == sha
    assert manifest["models"] == ["fake-model"]
    assert manifest["vaults"] == [
        {"vault": "seed5", "seed": 5, "graph_hash": graph["hash"], "notes": len(graph["notes"]), "questions": 20}
    ]
    bench_set = load_set(frozen / "questions.json")
    assert (bench_set["name"], bench_set["kind"], bench_set["corpus"]) == ("pilot", "synthetic", "pilot")
    assert manifest["set_hash"] == bench_set["hash"]
    assert len(bench_set["questions"]) == 20
    assert {q["half"] for q in bench_set["questions"]} == {"dev", "test"}
    with tarfile.open(frozen / "vault.tar.gz") as tar:
        members = sorted(m.name for m in tar.getmembers() if m.isfile())
    assert members == sorted(n["path"] for n in graph["notes"])
    for q in bench_set["questions"]:
        assert set(q["expected"]) <= set(members)
    # the same input gives the same bytes
    again = freeze([out], "pilot", tmp_path / "again", deny)
    assert again["files"]["vault.tar.gz"] == manifest["files"]["vault.tar.gz"]
    assert again["files"]["questions.json"] == manifest["files"]["questions.json"]


def test_freeze_refuses_a_deny_word_a_missing_deny_list_a_failed_verify_and_an_existing_set(done, tmp_path):
    out, graph, deny = done
    word = graph["subjects"][0]["word"]
    hit = tmp_path / "hit.txt"
    hit.write_text(f"{word.lower()}\n")
    with pytest.raises(SynthError, match=word.lower()):
        freeze([out], "pilot", tmp_path / "a", hit)
    with pytest.raises(SynthError, match="deny list"):
        freeze([out], "pilot", tmp_path / "b", tmp_path / "absent.txt")
    freeze([out], "pilot", tmp_path / "c", deny)
    with pytest.raises(SynthError, match="exists"):
        freeze([out], "pilot", tmp_path / "c", deny)
    first = graph["notes"][0]
    (out / "vault" / first["path"]).unlink()
    with pytest.raises(SynthError, match="verify"):
        freeze([out], "pilot", tmp_path / "d", deny)
    assert not (tmp_path / "a").exists() and not (tmp_path / "d").exists()


@pytest.mark.parametrize(
    "text, hits",
    [
        ("A note about Kelso and more.", ["kelso"]),
        ("kelsos are not a hit", []),
        ("The path /Users/someone/Notes is here", ["/users/someone"]),
        ("Two: KELSO, Marn Vale", ["kelso", "marn vale"]),
    ],
)
def test_deny_hits_match_whole_words_without_case(text, hits):
    assert deny_hits(text, ["kelso", "/users/someone", "marn vale"]) == hits


# --- the commands --------------------------------------------------------------------------------------------------


def test_the_synth_commands_graph_and_verify(tmp_path, capsys):
    out = tmp_path / "w"
    assert main(["synth", "graph", "--seed", "3", "--questions", "20", "--out", str(out)]) == 0
    graph = json.loads((out / "graph.json").read_text())
    printed = capsys.readouterr().out
    assert graph["hash"] in printed and "20 questions" in printed
    # nothing is written yet: each note and each question is missing
    assert main(["synth", "verify", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "note-missing" in printed and "question-missing" in printed


# --- review round 1 ------------------------------------------------------------------------------------------------


def test_the_note_prompt_asks_for_the_properties_and_verify_checks_them(work):
    out, graph = work
    assert "State each property of the subject in the note" in (PROMPTS / "note.md").read_text()
    fake = Fake(graph)
    render(out, fake, jobs=2)
    write_questions(out, fake, jobs=2)
    assert verify(out)["passed"]
    vault = out / "vault"
    dq = next(q for q in graph["questions"] if q["stratum"] == "distractor")
    # the gold note no longer states a property that the question uses (not the area: the frontmatter holds it)
    key, value = next((k, v) for k, v in dq["props"].items() if k != "area")
    gold = vault / dq["cluster"]
    gold.write_text(re.sub(rf"\b{re.escape(value)}\b", "gone", gold.read_text()))
    # a sibling note no longer states its own value of the property that differs from the gold subject
    subjects = {s["id"]: s for s in graph["subjects"]}
    sib_note = next(n for n in graph["notes"] if n["role"] == "distractor" and n["id"] in dq["notes"])
    sib, gsub = subjects[sib_note["subject"]], subjects[dq["subject"]]
    differs = next(k for k in sib["props"] if sib["props"][k] != gsub["props"][k])
    if differs in dq["props"]:
        p = vault / sib_note["path"]
        p.write_text(re.sub(rf"\b{re.escape(sib['props'][differs])}\b", "gone", p.read_text()))
    found = {(f["check"], f["id"]) for f in verify(out)["failures"]}
    assert ("property-in-note", dq["cluster"]) in found
    if differs in dq["props"]:
        assert ("property-in-note", sib_note["path"]) in found


def test_the_blind_check_shows_a_near_twin_of_another_area(graph):
    q = next(q for q in graph["questions"] if "area" not in q["props"] and q["stratum"] != "continuation")
    s = next(x for x in graph["subjects"] if x["id"] == q["subject"])
    other_area = next(a for a in graph["areas"] if a != s["area"])
    key = next(k for k in q["props"] if k != list(s["props"])[0])
    twin_props = {**s["props"], "area": other_area.lower()}
    twin_props[key] = "something else"
    broken = json.loads(json.dumps(graph))
    broken["subjects"].append(
        {
            "id": "near",
            "class": s["class"],
            "name": "Near twin",
            "word": "Near",
            "area": other_area,
            "props": twin_props,
        }
    )
    names = [c["name"] for c in candidates(broken, q)]
    assert "Near twin" in names and s["name"] in names


def test_questions_refuse_to_run_before_each_note_is_rendered(work):
    out, graph = work
    fake = Fake(graph)
    render(out, fake, jobs=1)
    state = json.loads((out / "notes.json").read_text())
    unrendered = graph["notes"][0]["id"]
    del state[unrendered]
    (out / "notes.json").write_text(json.dumps(state))
    with pytest.raises(SynthError, match="not rendered"):
        write_questions(out, fake, jobs=1)
    assert not (out / "written.json").exists() or json.loads((out / "written.json").read_text()) == {}


def test_no_heading_is_a_word_of_an_attribute_name_of_its_class():
    for cls, attributes in ATTRIBUTES.items():
        words = {w for attribute, _, _ in attributes for w in attribute.split()}
        for _, heading, _ in attributes:
            assert heading.lower() not in words, (cls, heading)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_forbidden_word_is_a_property_value_of_the_question(seed):
    graph = build_graph(world_config(seed, 650))
    facts = {f["id"]: f for f in graph["facts"]}
    for q in graph["questions"]:
        assert forbidden_hits(" ".join(q["props"].values()), q["forbidden"]) == [], q["id"]
        # the value and the full token stay forbidden
        assert {facts[q["fact"]]["value"], facts[q["fact"]]["token"]} <= set(q["forbidden"])


def test_freeze_refuses_notes_or_questions_from_other_prompts(work, tmp_path):
    out, graph = work
    changed = tmp_path / "prompts"
    import shutil

    shutil.copytree(PROMPTS, changed)
    (changed / "note.md").write_text((changed / "note.md").read_text() + "\nOne more rule.\n")
    fake = Fake(graph)
    render(out, fake, jobs=2, prompts=changed)
    write_questions(out, fake, jobs=2, prompts=changed)
    state = json.loads((out / "notes.json").read_text())
    assert all(set(v["prompts"]) == {"note.md"} for v in state.values())
    deny = tmp_path / "deny.txt"
    deny.write_text("qqzzxx\n")
    with pytest.raises(SynthError, match="prompt"):
        freeze([out], "pilot", tmp_path / "a", deny)
    manifest = freeze([out], "pilot", tmp_path / "b", deny, prompts=changed)
    assert manifest["prompts"]["note.md"] == hashlib.sha256((changed / "note.md").read_bytes()).hexdigest()
