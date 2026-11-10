"""The synthetic set: a seeded fact graph, notes and questions written by `claude -p`, a verifier and the freeze.

A work folder holds one seeded vault: world.json (the world config), graph.json (the fact graph and its hash),
vault/<vault>/... (the notes), notes.json and written.json (the state of each note and question) and verify.json.
The style targets are aggregate numbers of the real vaults only; no text of a real vault goes into a prompt."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from string import Template

from .bench import set_hash, write_set
from .config import home_dir

GENERATOR_VERSION = 1
PROMPTS = Path(__file__).resolve().parents[2] / "bench" / "synthetic" / "prompts"
DENY_ENV = "RAGNAROK_SYNTH_DENY"

STRATA = {
    "direct": 0.20,
    "lexical-gap": 0.20,
    "deep-section": 0.20,
    "distractor": 0.15,
    "outdated": 0.10,
    "continuation": 0.10,
    "multi-hop": 0.05,
}
# aggregate numbers of the real vaults (research-synthetic-vault); no name, path or text
STYLE = {
    "words_median": 882,
    "words_p90": 3526,
    "headings_median": 5,
    "table_share": 0.16,
    "code_share": 0.12,
    "list_share": 0.44,
    "link_mean": 2.08,
    "no_link_share": 0.53,
    "description_share": 0.71,
    "project_notes_median": 4,
}
NOTE_WORDS = (120, 2500)
LONG_WORDS = (3000, 10000)
PART_WORDS = 1500
RETRIES = 2  # a failed note or question is written again at most twice

# --- the invented world --------------------------------------------------------------------------------------------

AREAS = [
    "Aquaponics",
    "Beekeeping",
    "Boatyard",
    "Ceramics",
    "Cider",
    "Forge",
    "Glasshouse",
    "Orchard",
    "Radio",
    "Rainwater",
    "Sawmill",
    "Weather",
]
COLORS = ["green", "grey", "red", "blue", "yellow", "black"]
PLACE_KINDS = ["shed", "greenhouse", "cellar", "loft", "workshop", "pond", "yard", "barn"]
ROLES = ["supplier contact", "electrician", "plumber", "inspector", "neighbor", "mentor", "carpenter", "accountant"]
PROPERTIES = {
    "device": {
        "kind": [
            "circulation pump",
            "kiln controller",
            "data logger",
            "solar charger",
            "air compressor",
            "dehumidifier",
            "level sensor",
            "valve actuator",
            "band saw",
            "radio receiver",
            "scale",
            "heat exchanger",
        ],
        "color": COLORS,
        "energy source": ["mains", "battery", "solar"],
        "year installed": [str(y) for y in range(2014, 2026)],
        "location": PLACE_KINDS,
    },
    "place": {
        "kind": PLACE_KINDS,
        "orientation": ["north", "south", "east", "west"],
        "size": ["small", "medium", "large"],
        "year built": [str(y) for y in range(1988, 2025)],
        "floor": ["ground level", "raised", "below ground"],
    },
    "person": {
        "role": ROLES,
        "contact": ["phone", "email", "in person"],
        "availability": ["weekdays", "weekends", "evenings"],
        "known since": [str(y) for y in range(2008, 2026)],
    },
    "project": {
        "goal": ["retrofit", "rebuild", "upgrade", "survey", "repair", "expansion", "automation", "trial"],
        "start year": [str(y) for y in range(2019, 2027)],
        "season": ["spring", "summer", "autumn", "winter"],
        "lead": ROLES,
    },
}
# (attribute, heading, value kind) of each class
ATTRIBUTES = {
    "device": [
        ("part number", "Parts", "part"),
        ("serial number", "Parts", "serial"),
        ("firmware version", "Software", "version"),
        ("measured power draw", "Measurements", "watts"),
        ("calibration offset", "Adjustment", "offset"),
        ("flow rate", "Measurements", "flow"),
    ],
    "place": [
        ("floor area", "Dimensions", "area"),
        ("measured humidity", "Measurements", "humidity"),
        ("lease reference", "Admin", "lease"),
    ],
    "person": [
        ("quoted price", "Quotes", "price"),
        ("contract date", "Contracts", "date"),
    ],
}
FILLER_HEADINGS = [
    "Overview",
    "Background",
    "Setup",
    "Maintenance",
    "Open questions",
    "History",
    "Costs",
    "Risks",
    "Ideas",
    "Observations",
    "Checklist",
    "Lessons",
    "Supplies",
    "Schedule",
    "Problems",
    "References",
    "Safety",
    "Tools",
]
NOTE_TYPES = [("note", "notes"), ("research", "tests"), ("log", "log"), ("plan", "plan")]
ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "kr", "st", "tr", "gl"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "io"]
CODAS = ["", "", "n", "r", "l", "s", "th", "nd", "rk"]
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


class SynthError(Exception):
    pass


def stratum_counts(n: int) -> dict[str, int]:
    """The number of questions of each stratum: the fixed shares of n, rounded by the largest remainder."""
    exact = {s: share * n for s, share in STRATA.items()}
    counts = {s: math.floor(v) for s, v in exact.items()}
    rest = n - sum(counts.values())
    for s in sorted(STRATA, key=lambda s: -(exact[s] - counts[s]))[:rest]:
        counts[s] += 1
    return counts


def world_config(seed: int, questions: int) -> dict:
    return {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "vault": f"seed{seed}",
        "questions": questions,
        "strata": STRATA,
        "style": STYLE,
        "note_words": list(NOTE_WORDS),
        "long_words": list(LONG_WORDS),
    }


def graph_hash(graph: dict) -> str:
    body = {k: v for k, v in graph.items() if k != "hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class _Builder:
    def __init__(self, world: dict):
        self.world = world
        self.rng = random.Random(world["seed"])
        self.vault = world["vault"]
        self.subjects: list[dict] = []
        self.facts: list[dict] = []
        self.notes: list[dict] = []
        self.words: set[str] = set()
        self.tuples: set[tuple] = set()
        self.tokens: list[str] = []
        self.dates: set[str] = set()

    # names and values

    def word(self) -> str:
        while True:
            w = "".join(
                self.rng.choice(ONSETS) + self.rng.choice(VOWELS) + self.rng.choice(CODAS)
                for _ in range(self.rng.choice((2, 2, 3)))
            ).capitalize()
            if 4 <= len(w) <= 10 and w not in self.words:
                self.words.add(w)
                return w

    def _digits(self, n: int) -> str:
        return "".join(self.rng.choice("0123456789") for _ in range(n))

    def value(self, kind: str, before: int | None = None) -> tuple[str, str]:
        """A value and its token; the token is unique and no token is a part of another one."""
        r = self.rng
        while True:
            if kind == "part":
                v = f"{r.choice(LETTERS)}{r.choice(LETTERS)}-{self._digits(4)}-{r.choice(LETTERS)}"
                t = v
            elif kind == "serial":
                v = t = f"SN-{self._digits(8)}"
            elif kind == "version":
                t = f"{r.randint(1, 9)}.{r.randint(10, 49)}.{r.randint(100, 999)}"
                v = f"v{t}"
            elif kind in ("watts", "flow", "area", "humidity", "price", "budget", "offset"):
                lo, hi, unit, places = {
                    "watts": (5, 900, "W", 2),
                    "flow": (40, 2000, "l/h", 2),
                    "area": (4, 400, "m²", 2),
                    "humidity": (20, 95, "%", 2),
                    "price": (100, 9000, "EUR", 2),
                    "budget": (500, 40000, "EUR", 2),
                    "offset": (0, 3, "mm", 3),
                }[kind]
                t = f"{r.uniform(lo, hi):.{places}f}"
                v = f"{t} {unit}"
            elif kind == "lease":
                v = t = f"LR-{self._digits(6)}"
            elif kind == "next":
                v = t = f"NX-{self._digits(4)}"
            elif kind == "date":
                year = r.randint(2018, before - 1) if before else r.randint(2018, 2026)
                v = t = f"{year}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"
            else:
                raise ValueError(kind)
            if t.rstrip("0").endswith(".") or any(t in o or o in t for o in self.tokens):
                continue
            self.tokens.append(t)
            return v, t

    # subjects, facts and notes

    def subject(self, cls: str, area: str, props: dict | None = None, fixed: dict | None = None) -> dict:
        while True:
            p = props or {}
            if not props:
                keys = list(PROPERTIES[cls])
                p = {keys[0]: self.rng.choice(PROPERTIES[cls][keys[0]]), "area": area.lower()}
                p.update({k: self.rng.choice(PROPERTIES[cls][k]) for k in keys[1:]})
                p.update(fixed or {})
            key = (cls, tuple(sorted(p.items())))
            if key not in self.tuples:
                break
            if props:
                raise ValueError("a subject with these properties exists")
        self.tuples.add(key)
        word = self.word()
        main = list(p.values())[0]
        name = f"{word} {self.word()}" if cls == "person" else f"{word} {main}"
        s = {"id": f"s{len(self.subjects) + 1:04d}", "class": cls, "name": name, "word": word, "area": area, "props": p}
        self.subjects.append(s)
        return s

    def sibling(self, s: dict) -> dict:
        """A subject of the same class, kind and area with one other property."""
        keys = [k for k in PROPERTIES[s["class"]] if k != list(PROPERTIES[s["class"]])[0]]
        for _ in range(200):
            k = self.rng.choice(keys)
            p = dict(s["props"])
            p[k] = self.rng.choice([v for v in PROPERTIES[s["class"]][k] if v != p[k]])
            if (s["class"], tuple(sorted(p.items()))) not in self.tuples:
                return self.subject(s["class"], s["area"], p)
        raise ValueError("no sibling")

    def fact(self, s: dict, attribute: str, heading: str, kind: str, current=True, before=None) -> dict:
        value, token = self.value(kind, before)
        f = {
            "id": f"f{len(self.facts) + 1:04d}",
            "subject": s["id"],
            "attribute": attribute,
            "heading": heading,
            "value": value,
            "token": token,
            "note": None,
            "current": current,
            "bridge": False,
        }
        self.facts.append(f)
        return f

    def note(self, role: str, s: dict, facts: list[dict], area: str, in_project: bool, long=False, **kw) -> dict:
        n = {
            "id": f"n{len(self.notes) + 1:04d}",
            "role": role,
            "subject": s["id"],
            "area": area,
            "in_project": in_project,
            "long": long,
            "facts": [f["id"] for f in facts],
            "status": "active",
            **kw,
        }
        for f in facts:
            f["note"] = n["id"]
        self.notes.append(n)
        return n


def _pick_props(subjects: list[dict], s: dict, rng: random.Random) -> dict:
    """The smallest property set (at least 2, the main one first, the others in a seeded order) that matches only s."""
    keys = list(s["props"])
    order = [keys[0]] + rng.sample(keys[1:], len(keys) - 1)
    for size in range(2, len(order) + 1):
        props = {k: s["props"][k] for k in order[:size]}
        if matches(subjects, s["class"], props) == [s["id"]]:
            return props
    raise ValueError(f"no unique property set for {s['id']}")


def matches(subjects: list[dict], cls: str, props: dict) -> list[str]:
    return [s["id"] for s in subjects if s["class"] == cls and all(s["props"].get(k) == v for k, v in props.items())]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def build_graph(world: dict) -> dict:
    """The fact graph of a world config. The same config gives the same graph."""
    b = _Builder(world)
    rng = b.rng
    n = world["questions"]
    counts = stratum_counts(n)
    areas = rng.sample(AREAS, max(2, min(len(AREAS), n // 40)))
    plan = [s for s, c in counts.items() for _ in range(c)]
    rng.shuffle(plan)
    pending = []  # (stratum, question subject, asked fact, gold note, notes needed, bridge fact)
    continuation_areas = []
    for stratum in plan:
        area = rng.choice(areas)
        if stratum == "continuation":
            continuation_areas.append(area)
            continue
        if stratum == "multi-hop":
            place = b.subject("place", area)
            device = b.subject("device", area, fixed={"location": place["props"]["kind"]})
            attribute, heading, kind = rng.choice(ATTRIBUTES["device"])
            answer = b.fact(device, attribute, heading, kind)
            gold = b.note("gold", device, [answer], area, rng.random() < 0.6)
            bridge_fact = {
                "id": f"f{len(b.facts) + 1:04d}",
                "subject": place["id"],
                "attribute": "installed device",
                "heading": "Equipment",
                "value": device["name"],
                "token": device["name"],
                "note": None,
                "current": True,
                "bridge": True,
            }
            b.facts.append(bridge_fact)
            bridge = b.note("bridge", place, [bridge_fact], area, rng.random() < 0.6)
            pending.append((stratum, place, answer, gold, [gold, bridge], bridge_fact))
            continue
        cls = "device" if stratum == "deep-section" and rng.random() < 0.7 else None
        cls = cls or ("place" if stratum == "deep-section" else rng.choice(["device", "device", "place", "person"]))
        s = b.subject(cls, area)
        attribute, heading, kind = rng.choice(ATTRIBUTES[cls])
        f = b.fact(s, attribute, heading, kind)
        gold = b.note("gold", s, [f], area, rng.random() < 0.6, long=stratum == "deep-section")
        needed = [gold]
        if stratum == "distractor":
            for _ in range(rng.randint(1, 3)):
                sib = b.sibling(s)
                sf = b.fact(sib, attribute, heading, kind)
                needed.append(b.note("distractor", sib, [sf], area, rng.random() < 0.6))
        elif stratum == "outdated":
            year = rng.randint(2018, 2024)
            old = b.fact(s, attribute, heading, kind, current=False, before=year + 1 if kind == "date" else None)
            needed.append(b.note("outdated", s, [old], area, gold["in_project"], status="archived", year=year))
        pending.append((stratum, s, f, gold, needed, None))

    # projects: each project note goes into a project of its area, about STYLE project_notes_median notes in each
    projects = []
    per = max(1, STYLE["project_notes_median"] - 1)
    for area in areas:
        members = [x for x in b.notes if x["area"] == area and x["in_project"]]
        need = max(math.ceil(len(members) / per), continuation_areas.count(area))
        made = []
        for _ in range(need):
            p = b.subject("project", area)
            budget = b.fact(p, "budget", "Budget", "budget")
            nxt = b.fact(p, "next action", "Next action", "next")
            decision = b.fact(p, "go decision date", "Decisions", "date")
            drivers = b.note("drivers", p, [budget], area, True)
            summary = b.note("summary", p, [nxt, decision], area, True)
            made.append((p, drivers, summary, nxt))
        projects += made
        # outdated notes stay in the folder of their gold note
        by_subject = {}
        for i, x in enumerate(m for m in members if m["role"] != "outdated"):
            x["project"] = made[i % len(made)][0]["id"]
            by_subject.setdefault(x["subject"], x["project"])
        for x in members:
            if x["role"] == "outdated":
                x["project"] = by_subject[x["subject"]]
        for p, _drivers, summary, nxt in made[: continuation_areas.count(area)]:
            pending.append(("continuation", p, nxt, summary, [summary], None))
    for p, drivers, summary, _ in projects:
        drivers["project"] = summary["project"] = p["id"]

    subjects = {s["id"]: s for s in b.subjects}
    facts = {f["id"]: f for f in b.facts}
    _place_notes(b, subjects, facts)
    _style_notes(b, subjects, facts)

    # questions, in the order of the plan, with halves in each stratum
    pending.sort(key=lambda x: b.notes.index(x[3]))
    questions = []
    for i, (stratum, s, f, gold, needed, bridge) in enumerate(pending, start=1):
        props = _pick_props(b.subjects, s, rng)
        fact_subject = subjects[f["subject"]]
        forbidden = [s["name"], s["word"], gold["title"], f["heading"], f["value"], f["token"]]
        forbidden += [part for part in re.split(r"[-.\s]", f["token"]) if len(part) >= 3 and part != f["token"]]
        if stratum == "multi-hop":
            kind = fact_subject["props"]["kind"]
            forbidden += [fact_subject["name"], fact_subject["word"], kind, *kind.split()]
        if stratum == "lexical-gap":
            allowed = _words(" ".join(f"{k} {v}" for k, v in props.items()))
            forbidden += sorted(w for w in _words(f["attribute"]) if len(w) >= 4 and w not in allowed)
        # the question must be able to name its properties: no forbidden word may be one of them (a year of a date
        # token, a word of the bridge kind); the value and the full token stay forbidden
        props_text = " ".join(props.values())
        forbidden = [w for w in forbidden if w in (f["value"], f["token"]) or not forbidden_hits(props_text, [w])]
        questions.append(
            {
                "id": f"{b.vault}-q{i:04d}",
                "stratum": stratum,
                "fact": f["id"],
                "subject": s["id"],
                "props": props,
                "expected": [x["path"] for x in needed if x["role"] in ("gold", "summary", "bridge")],
                "cluster": gold["path"],
                "area": gold["area"].lower(),
                "half": None,
                "forbidden": list(dict.fromkeys(forbidden)),
                "notes": [x["id"] for x in needed],
                "bridge": bridge["id"] if bridge else None,
            }
        )
    for stratum in STRATA:
        group = [q for q in questions if q["stratum"] == stratum]
        rng.shuffle(group)
        for j, q in enumerate(group):
            q["half"] = ("dev", "test")[j % 2]
    for x in b.notes:
        x.pop("in_project")
    graph = {
        "version": GENERATOR_VERSION,
        "seed": world["seed"],
        "vault": b.vault,
        "areas": areas,
        "subjects": b.subjects,
        "facts": b.facts,
        "notes": b.notes,
        "questions": questions,
    }
    graph["hash"] = graph_hash(graph)
    return graph


def _place_notes(b: _Builder, subjects: dict, facts: dict) -> None:
    """Title, type and path of each note."""
    rng = b.rng
    titles = {}
    for x in b.notes:
        s = subjects[x["subject"]]
        if x["role"] in ("drivers", "summary"):
            folder = f"{b.vault}/{x['area']}/{s['word'].lower()}-{s['props']['goal']}"
            name = "DRIVERS" if x["role"] == "drivers" else "SUMMARY"
            x["title"] = f"{'Drivers' if name == 'DRIVERS' else 'Summary'} of {s['name']}"
            x["type"] = x["role"]
            x["path"] = f"{folder}/{name}.md"
            continue
        if x.get("project"):
            p = subjects[x["project"]]
            folder = f"{b.vault}/{x['area']}/{p['word'].lower()}-{p['props']['goal']}"
            ntype, suffix = ("research", "study") if x["long"] else rng.choice(NOTE_TYPES)
            title = f"{s['name']} {suffix}"
        else:
            folder = f"{b.vault}/{x['area']}/Reference"
            ntype = "reference"
            title = s["name"] if not x["long"] else f"{s['name']} handbook"
        if x["role"] == "outdated":
            gold = next(g for g in b.notes if g["role"] == "gold" and g["subject"] == x["subject"])
            title = f"{titles[gold['id']]} {x['year']}"
        titles[x["id"]] = title
        x.update(title=title, type=ntype, path=f"{folder}/{title}.md")
    for x in b.notes:
        x.setdefault("project", None)


def _style_notes(b: _Builder, subjects: dict, facts: dict) -> None:
    """Length, headings, table, code, list, links and description of each note from the aggregate style targets."""
    rng = b.rng
    mu = math.log(STYLE["words_median"])
    sigma = math.log(STYLE["words_p90"] / STYLE["words_median"]) / 1.2816
    per_heading = STYLE["words_median"] / STYLE["headings_median"]
    linkable = {}
    for x in b.notes:
        if x["role"] not in ("drivers", "summary"):
            linkable.setdefault(x["area"], []).append(x["title"])
    for x in b.notes:
        if x["long"]:
            words = rng.randint(*LONG_WORDS)
        else:
            words = int(min(max(rng.lognormvariate(mu, sigma), NOTE_WORDS[0]), NOTE_WORDS[1]))
        fact_headings = list(dict.fromkeys(facts[f]["heading"] for f in x["facts"]))
        total = max(len(fact_headings) + 1, min(17, round(words / per_heading)))
        if x["long"]:
            total = max(total, 8)
        fillers = rng.sample([h for h in FILLER_HEADINGS if h not in fact_headings], total - len(fact_headings))
        if x["long"]:
            headings = fillers + fact_headings  # the fact is in the last section
        else:
            headings = fillers[:]
            for h in fact_headings:
                headings.insert(rng.randint(1, len(headings)), h)
        if rng.random() < STYLE["no_link_share"]:
            links = []
        else:
            pool = [t for t in linkable[x["area"]] if t != x["title"]]
            links = rng.sample(pool, min(len(pool), rng.randint(1, 8)))
        s = subjects[x["subject"]]
        description = None
        if rng.random() < STYLE["description_share"]:
            description = f"{x['type'].capitalize()} about {s['name']}."
        x.update(
            words=words,
            headings=headings,
            table=rng.random() < STYLE["table_share"],
            code=rng.random() < STYLE["code_share"],
            list=rng.random() < STYLE["list_share"],
            links=links,
            description=description,
        )


def property_failures(graph: dict) -> list[tuple[str, str]]:
    """The (question id, detail) of each question whose property set does not match exactly its subject."""
    out = []
    classes = {s["id"]: s["class"] for s in graph["subjects"]}
    for q in graph["questions"]:
        fits = matches(graph["subjects"], classes.get(q["subject"], "?"), q["props"])
        if fits != [q["subject"]]:
            out.append((q["id"], f"the properties {q['props']} match {len(fits)} subjects: {', '.join(fits)}"))
    return out


# --- the work folder -----------------------------------------------------------------------------------------------


def write_graph(out: Path, seed: int, questions: int) -> dict:
    out = Path(out)
    if (out / "graph.json").exists():
        raise SynthError(f"{out / 'graph.json'} exists; use a new folder")
    out.mkdir(parents=True, exist_ok=True)
    world = world_config(seed, questions)
    graph = build_graph(world)
    (out / "world.json").write_text(json.dumps(world, indent=1) + "\n")
    (out / "graph.json").write_text(json.dumps(graph, indent=1, ensure_ascii=False) + "\n")
    return graph


def _load(work: Path) -> dict:
    path = Path(work) / "graph.json"
    if not path.exists():
        raise SynthError(f"no graph at {path}; run `ragnarok synth graph` first")
    return json.loads(path.read_text())


def _state(work: Path, name: str) -> dict:
    path = Path(work) / name
    return json.loads(path.read_text()) if path.exists() else {}


class _State:
    """A JSON state file that each finished note or question updates, so a second run continues."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.lock = threading.Lock()

    def set(self, key: str, value: dict) -> None:
        with self.lock:
            self.data[key] = value
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
            tmp.replace(self.path)


def claude(prompt: str, model: str) -> tuple[str, str]:
    """The text and the model ID of one `claude -p` call with no tools, no settings and no CLAUDE.md of a folder."""
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--system-prompt",
        "You are a careful technical writer. You follow the instructions exactly.",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--no-session-persistence",
    ]
    with tempfile.TemporaryDirectory(prefix="ragnarok-synth-") as cwd:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise SynthError(f"claude -p failed ({out.returncode}): {out.stderr.strip()[:500]}")
    data = json.loads(out.stdout)
    if data.get("is_error"):
        raise SynthError(f"claude -p failed: {str(data.get('result'))[:500]}")
    return data["result"], next(iter(data.get("modelUsage") or {model: None}))


def prompt_hashes(prompts: Path, names: tuple[str, ...] | None = None) -> dict[str, str]:
    """The SHA-256 of each prompt file (all files of the folder when names is None)."""
    folder = Path(prompts)
    names = names or tuple(sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()))
    return {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names}


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(items: list, fn, jobs: int) -> None:
    if jobs <= 1:
        for item in items:
            fn(item)
        return
    with ThreadPoolExecutor(jobs) as pool:
        for result in pool.map(fn, items):
            del result


# --- notes ---------------------------------------------------------------------------------------------------------


def _describe(s: dict) -> str:
    return f"{s['name']}, a {s['class']} ({'; '.join(f'{k}: {v}' for k, v in s['props'].items())})"


def _context(note: dict, s: dict) -> str:
    if note["role"] == "outdated":
        return f"This note is archived. It records the state in {note['year']}; later notes replace its values."
    if note["role"] == "drivers":
        return "This is the drivers note of the project: its goal, constraints, budget and how success is measured."
    if note["role"] == "summary":
        return "This is the summary note of the project: its stage, what is done, open points and the next action."
    return ""


def note_prompts(graph: dict, note: dict, template: str, feedback: str = "") -> list[str]:
    """The prompts of a note: one, or one for each part of a long note (the facts go into the last part)."""
    subjects = {s["id"]: s for s in graph["subjects"]}
    facts = {f["id"]: f for f in graph["facts"]}
    s = subjects[note["subject"]]
    style = []
    if note["table"]:
        style.append("Include one Markdown table.")
    if note["code"]:
        style.append("Include one fenced code block, for example a configuration, a command or a log extract.")
    if note["list"]:
        style.append("Use a bulleted list in at least one section.")
    links = (
        "Include these wiki links where they fit: " + ", ".join(f"[[{t}]]" for t in note["links"])
        if note["links"]
        else "Do not write wiki links."
    )
    fact_lines = [
        f"- {facts[f]['attribute']} of {subjects[facts[f]['subject']]['name']}: {facts[f]['value']} "
        f'(section "{facts[f]["heading"]}")'
        for f in note["facts"]
    ]
    parts = math.ceil(note["words"] / PART_WORDS) if note["long"] else 1
    size = math.ceil(len(note["headings"]) / parts)
    groups = [note["headings"][i : i + size] for i in range(0, len(note["headings"]), size)]
    words = math.ceil(note["words"] / len(groups))
    prompts = []
    for i, headings in enumerate(groups):
        last = i == len(groups) - 1
        part = ""
        if len(groups) > 1:
            part = (
                f"This text is part {i + 1} of {len(groups)} of one long note. Write only the sections listed below; "
                "the other parts cover the other sections. Start with the first heading and do not write a conclusion."
            )
        prompts.append(
            Template(template).substitute(
                part=part,
                title=note["title"],
                type=note["type"],
                subject=_describe(s),
                context=_context(note, s),
                headings="\n".join(f"## {h}" for h in headings),
                words=words,
                style="\n".join(style),
                facts="\n".join(fact_lines) if last else "(no facts in this part)",
                links=links if i == 0 else "Do not write wiki links.",
                feedback=f"\nAn earlier version failed these checks: {feedback}. Correct them." if feedback else "",
            )
        )
    return prompts


def _clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def note_text(graph: dict, note: dict, bodies: list[str]) -> str:
    project = next((x for x in graph["subjects"] if x["id"] == note["project"]), None)
    front = [
        "---",
        f"type: {note['type']}",
        f"status: {note['status']}",
        f"project: {(project['word'].lower() + '-' + project['props']['goal']) if project else 'reference'}",
        f"area: {note['area'].lower()}",
    ]
    if note["description"]:
        front.append(f"description: {json.dumps(note['description'], ensure_ascii=False)}")
    return "\n".join([*front, "---", "", f"# {note['title']}", "", "\n\n".join(_clean(b) for b in bodies), ""])


class _Tokens:
    """Finds the value tokens of all facts (no bridge) in a text, without case."""

    def __init__(self, graph: dict):
        self.by_token = {f["token"].lower(): f for f in graph["facts"] if not f["bridge"]}
        tokens = sorted(self.by_token, key=len, reverse=True)
        self.pattern = re.compile("|".join(map(re.escape, tokens))) if tokens else None

    def find(self, text: str) -> set[str]:
        if self.pattern is None:
            return set()
        return {self.by_token[m.group(0)]["id"] for m in self.pattern.finditer(text.lower())}


def required_properties(graph: dict) -> dict[str, list[str]]:
    """The property values that each note must state, so that the notes separate the subjects as the questions do:
    the properties of each question (no area: the frontmatter holds it) in its gold note (the bridge note for
    multi-hop), and for a distractor question each sibling's own values of those properties in its note."""
    subjects = {s["id"]: s for s in graph["subjects"]}
    notes = {n["id"]: n for n in graph["notes"]}
    paths = {n["path"]: n["id"] for n in graph["notes"]}
    out: dict[str, list[str]] = {}
    for q in graph["questions"]:
        keys = [k for k in q["props"] if k != "area"]
        target = paths[q["expected"][1]] if q["stratum"] == "multi-hop" else paths[q["cluster"]]
        out.setdefault(target, []).extend(q["props"][k] for k in keys)
        if q["stratum"] == "distractor":
            for nid in q["notes"]:
                if notes[nid]["role"] == "distractor":
                    sib = subjects[notes[nid]["subject"]]
                    out.setdefault(nid, []).extend(sib["props"][k] for k in keys)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def _states(value: str, text: str) -> bool:
    """A property value is in a text as whole words, without case; a space also matches a hyphen."""
    pattern = r"(?<!\w)" + r"[\s-]+".join(map(re.escape, value.split())) + r"(?!\w)"
    return re.search(pattern, text, re.I) is not None


def note_failures(
    graph: dict, note: dict, text: str, tokens: _Tokens | None = None, required: dict | None = None
) -> list[tuple[str, str]]:
    """The (check, detail) of each failed string check of a note."""
    tokens = tokens or _Tokens(graph)
    required = required_properties(graph) if required is None else required
    facts = {f["id"]: f for f in graph["facts"]}
    asked = {q["fact"] for q in graph["questions"]}
    out = []
    for fid in note["facts"]:
        f = facts[fid]
        if f["token"] not in text:
            check = "gold-in-note" if fid in asked else "fact-in-note"
            out.append((check, f"{f['attribute']} {f['value']!r} is not in the note"))
    for fid in sorted(tokens.find(text) - set(note["facts"])):
        f = facts[fid]
        if fid in asked:
            check = "distractor-holds-gold" if note["role"] == "distractor" else "gold-elsewhere"
        else:
            check = "value-elsewhere"
        out.append((check, f"the value {f['value']!r} of {fid} belongs to another note"))
    for value in required.get(note["id"], []):
        if not _states(value, text):
            out.append(("property-in-note", f"the note does not state the property value {value!r}"))
    if note["long"]:
        words = len(text.split())
        if not LONG_WORDS[0] <= words <= LONG_WORDS[1]:
            out.append(("long-note-length", f"{words} words, not {LONG_WORDS[0]} to {LONG_WORDS[1]}"))
        for fid in note["facts"]:
            at = text.find(facts[fid]["token"])
            if fid in asked and 0 <= at < len(text) / 2:
                out.append(
                    ("deep-fact-position", f"the value is at {at} of {len(text)} characters, not in the second half")
                )
    return out


def render(work: Path, runner=claude, model: str = "sonnet", jobs: int = 4, prompts: Path = PROMPTS) -> dict:
    """Write each note that is not written yet. A note that fails its string checks is written again at most twice
    and then removed."""
    work = Path(work)
    graph = _load(work)
    template = (Path(prompts) / "note.md").read_text()
    hashes = prompt_hashes(prompts, ("note.md",))
    state = _State(work / "notes.json")
    tokens = _Tokens(graph)
    required = required_properties(graph)
    vault = work / "vault"

    def one(note: dict) -> None:
        if note["id"] in state.data:
            return
        path = vault / note["path"]
        failures: list[tuple[str, str]] = []
        models = set()
        attempt = 0
        while attempt <= RETRIES:
            attempt += 1
            feedback = "; ".join(f"{c}: {d}" for c, d in failures)
            bodies = []
            for prompt in note_prompts(graph, note, template, feedback):
                text, model_id = runner(prompt, model)
                bodies.append(text)
                models.add(model_id)
            text = note_text(graph, note, bodies)
            failures = note_failures(graph, note, text, tokens, required)
            if not failures:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                break
        status = "removed" if failures else "ok"
        if failures:
            path.unlink(missing_ok=True)
        state.set(
            note["id"],
            {
                "status": status,
                "attempts": attempt,
                "failures": [list(f) for f in failures],
                "models": sorted(models),
                "prompts": hashes,
                "date": _now(),
            },
        )

    _run(graph["notes"], one, jobs)
    done = state.data
    return {
        "notes": len(graph["notes"]),
        "ok": sum(v["status"] == "ok" for v in done.values()),
        "removed": sum(v["status"] == "removed" for v in done.values()),
    }


# --- questions -----------------------------------------------------------------------------------------------------


def forbidden_hits(text: str, forbidden: list[str]) -> list[str]:
    """The forbidden words in a text: a word with word boundaries, a value as a part of any word; no case."""
    low = text.lower()
    hits = []
    for w in forbidden:
        wl = w.lower()
        if re.fullmatch(r"[a-z ]+", wl):
            if re.search(rf"\b{re.escape(wl)}\b", low):
                hits.append(w)
        elif wl in low:
            hits.append(w)
    return hits


def _ask(graph: dict, q: dict) -> str:
    facts = {f["id"]: f for f in graph["facts"]}
    subjects = {s["id"]: s for s in graph["subjects"]}
    f = facts[q["fact"]]
    s = subjects[q["subject"]]
    if q["stratum"] == "continuation":
        return "where the work on a project stopped and what the next step of the project is (in its summary note)"
    if q["stratum"] == "multi-hop":
        return f"the {f['attribute']} of the device that is installed at a {s['class']}"
    if q["stratum"] == "outdated":
        return f"the current {f['attribute']} of a {s['class']}"
    return f"the {f['attribute']} of a {s['class']}"


def question_prompt(graph: dict, q: dict, template: str, instructions: dict, feedback: str = "") -> str:
    return Template(template).substitute(
        ask=_ask(graph, q),
        properties="\n".join(f"- {k}: {v}" for k, v in q["props"].items()),
        instruction=instructions[q["stratum"]]["instruction"],
        forbidden=", ".join(q["forbidden"]),
        feedback=f"\nAn earlier question failed: {feedback}. Write a different question." if feedback else "",
    )


def candidates(graph: dict, q: dict) -> list[dict]:
    """The subjects that the question can match: the same class and the same area, and the subjects of the same class
    in any area that differ from the property set of the question in at most one property."""
    s = next(x for x in graph["subjects"] if x["id"] == q["subject"])
    out = []
    for x in graph["subjects"]:
        if x["class"] != s["class"]:
            continue
        differ = sum(x["props"].get(k) != v for k, v in q["props"].items())
        if x["area"] == s["area"] or differ <= 1:
            out.append(x)
    return out


def check_prompt(graph: dict, q: dict, question: str, template: str) -> str:
    rows = [f"- {c['name']} | " + "; ".join(f"{k}: {v}" for k, v in c["props"].items()) for c in candidates(graph, q)]
    return Template(template).substitute(question=question, candidates="\n".join(rows))


def _answer(text: str) -> str:
    return text.strip().strip("`*\"'. ").splitlines()[0].strip() if text.strip() else ""


def write_questions(work: Path, runner=claude, model: str = "sonnet", jobs: int = 4, prompts: Path = PROMPTS) -> dict:
    """Write each question from its fact tuple, check its forbidden words and resolve it with the blind subject check.
    A failed question is written again at most twice and then removed; a question of a removed note is removed."""
    work = Path(work)
    graph = _load(work)
    prompts = Path(prompts)
    template = (prompts / "question.md").read_text()
    check = (prompts / "subject-check.md").read_text()
    hashes = prompt_hashes(prompts, ("question.md", "subject-check.md", "strata.toml"))
    instructions = tomllib.loads((prompts / "strata.toml").read_text())
    notes = _state(work, "notes.json")
    if unrendered := [n["path"] for n in graph["notes"] if n["id"] not in notes]:
        raise SynthError(
            f"{len(unrendered)} notes are not rendered; run `ragnarok synth render` first "
            f"(for example {', '.join(unrendered[:3])})"
        )
    state = _State(work / "written.json")
    subjects = {s["id"]: s for s in graph["subjects"]}

    def one(q: dict) -> None:
        if q["id"] in state.data:
            return
        missing = [n for n in q["notes"] if notes[n]["status"] == "removed"]
        if missing:
            state.set(q["id"], {"status": "removed", "attempts": 0, "failures": [["note-removed", ", ".join(missing)]]})
            return
        gold = subjects[q["subject"]]["name"]
        failures: list[list[str]] = []
        models = set()
        text = answer = ""
        attempt = 0
        while attempt <= RETRIES:
            attempt += 1
            feedback = "; ".join(f"{c}: {d}" for c, d in failures)
            raw, model_id = runner(question_prompt(graph, q, template, instructions, feedback), model)
            models.add(model_id)
            text = " ".join(raw.strip().strip('"').split())
            failures = []
            if hits := forbidden_hits(text, q["forbidden"]):
                failures.append(["forbidden-word", f"it uses {', '.join(hits)}"])
                continue
            raw, model_id = runner(check_prompt(graph, q, text, check), model)
            models.add(model_id)
            answer = _answer(raw)
            if answer != gold:
                failures.append(["subject-check", f"a blind reader named {answer!r}, not the subject"])
                continue
            break
        state.set(
            q["id"],
            {
                "status": "removed" if failures else "ok",
                "attempts": attempt,
                "failures": failures,
                "question": text,
                "answer": answer,
                "models": sorted(models),
                "prompts": hashes,
                "date": _now(),
            },
        )

    _run(graph["questions"], one, jobs)
    done = state.data
    return {
        "questions": len(graph["questions"]),
        "ok": sum(v["status"] == "ok" for v in done.values()),
        "removed": sum(v["status"] == "removed" for v in done.values()),
    }


# --- verify --------------------------------------------------------------------------------------------------------

CHECKS = (
    "graph-hash",
    "unique-properties",
    "strata",
    "halves",
    "note-missing",
    "gold-in-note",
    "fact-in-note",
    "property-in-note",
    "gold-elsewhere",
    "distractor-holds-gold",
    "value-elsewhere",
    "long-note-length",
    "deep-fact-position",
    "question-missing",
    "forbidden-word",
    "subject-check",
    "expected-missing",
)


def verify(work: Path) -> dict:
    """Each check of the graph, the kept notes and the kept questions. Writes verify.json; removed notes and
    questions are listed, not failed."""
    work = Path(work)
    graph = _load(work)
    notes_state = _state(work, "notes.json")
    written = _state(work, "written.json")
    failures = []

    def fail(check: str, item: str, detail: str) -> None:
        failures.append({"check": check, "id": item, "detail": detail})

    if graph_hash(graph) != graph.get("hash"):
        fail("graph-hash", "graph.json", "the graph does not agree with its hash")
    for qid, detail in property_failures(graph):
        fail("unique-properties", qid, detail)
    target = stratum_counts(len(graph["questions"]))
    strata = {}
    for s in STRATA:
        group = [q for q in graph["questions"] if q["stratum"] == s]
        dev = sum(q["half"] == "dev" for q in group)
        if len(group) != target[s]:
            fail("strata", s, f"{len(group)} questions, not {target[s]}")
        if abs(dev - (len(group) - dev)) > 1 or any(q["half"] not in ("dev", "test") for q in group):
            fail("halves", s, f"dev {dev}, test {len(group) - dev}")
        strata[s] = {"target": target[s], "planned": len(group)}
    removed_notes = {n for n, v in notes_state.items() if v["status"] == "removed"}
    removed_q = {
        q["id"]
        for q in graph["questions"]
        if written.get(q["id"], {}).get("status") == "removed" or set(q["notes"]) & removed_notes
    }
    tokens = _Tokens(graph)
    vault = work / "vault"
    kept_paths = set()
    for note in graph["notes"]:
        if note["id"] in removed_notes:
            continue
        path = vault / note["path"]
        if not path.exists():
            fail("note-missing", note["path"], "the note is not written")
            continue
        kept_paths.add(note["path"])
        for check, detail in note_failures(graph, note, path.read_text(), tokens):
            fail(check, note["path"], detail)
    subjects = {s["id"]: s for s in graph["subjects"]}
    for q in graph["questions"]:
        if q["id"] in removed_q:
            continue
        w = written.get(q["id"])
        if not w or w.get("status") != "ok":
            fail("question-missing", q["id"], "the question is not written")
            continue
        if hits := forbidden_hits(w["question"], q["forbidden"]):
            fail("forbidden-word", q["id"], f"it uses {', '.join(hits)}")
        if w.get("answer") != subjects.get(q["subject"], {}).get("name"):
            fail("subject-check", q["id"], f"the blind reader named {w.get('answer')!r}")
        for e in q["expected"]:
            if e not in kept_paths:
                fail("expected-missing", q["id"], f"{e} is not a kept note")
    kept = [q for q in graph["questions"] if q["id"] not in removed_q]
    for s in STRATA:
        group = [q for q in kept if q["stratum"] == s]
        dev = sum(q["half"] == "dev" for q in group)
        strata[s].update(kept=len(group), dev=dev, test=len(group) - dev)
    report = {
        "vault": graph["vault"],
        "seed": graph["seed"],
        "graph_hash": graph.get("hash"),
        "notes": {
            "planned": len(graph["notes"]),
            "kept": len(graph["notes"]) - len(removed_notes),
            "removed": len(removed_notes),
        },
        "questions": {"planned": len(graph["questions"]), "kept": len(kept), "removed": len(removed_q)},
        "strata": strata,
        "removed": {
            "notes": sorted(n["path"] for n in graph["notes"] if n["id"] in removed_notes),
            "questions": sorted(removed_q),
        },
        "checks": {c: sum(f["check"] == c for f in failures) for c in CHECKS},
        "failures": failures,
        "passed": not failures,
    }
    (work / "verify.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    return report


def set_questions(work: Path) -> list[dict]:
    """The kept questions of a verified work folder in the set format v2, with the half."""
    graph = _load(work)
    written = _state(work, "written.json")
    removed = set(json.loads((Path(work) / "verify.json").read_text())["removed"]["questions"])
    return [
        {
            "id": q["id"],
            "question": written[q["id"]]["question"],
            "expected": q["expected"],
            "stratum": q["stratum"],
            "area": q["area"],
            "cluster": q["cluster"],
            "half": q["half"],
        }
        for q in graph["questions"]
        if q["id"] not in removed
    ]


# --- privacy and freeze --------------------------------------------------------------------------------------------


def deny_path() -> Path:
    """RAGNAROK_SYNTH_DENY, else <ragnarok home>/data/synth-deny.txt. The file never goes into the repo."""
    return (
        Path(os.environ[DENY_ENV]).expanduser() if os.environ.get(DENY_ENV) else home_dir() / "data" / "synth-deny.txt"
    )


def read_deny(path: Path) -> list[str]:
    path = Path(path)
    if not path.is_file():
        raise SynthError(f"no deny list at {path}; write one term on each line (set {DENY_ENV} for another path)")
    terms = [line.split("#", 1)[0].strip().lower() for line in path.read_text().splitlines()]
    terms = [t for t in terms if t]
    if not terms:
        raise SynthError(f"the deny list {path} has no term")
    return terms


def deny_hits(text: str, terms: list[str]) -> list[str]:
    """The deny terms in a text, without case; a term of word characters only matches whole words."""
    low = text.lower()
    hits = []
    for t in terms:
        t = t.lower()
        pattern = rf"(?<!\w){re.escape(t)}(?!\w)" if re.fullmatch(r"[\w ]+", t) else re.escape(t)
        if re.search(pattern, low):
            hits.append(t)
    return hits


def archive_hits(path: Path, terms: list[str]) -> list[tuple[str, str]]:
    """The (member, term) of each deny term in the notes of a vault archive."""
    out = []
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if m.isfile():
                text = tar.extractfile(m).read().decode(errors="ignore")
                out += [(m.name, t) for t in deny_hits(m.name + "\n" + text, terms)]
    return out


def _archive(files: list[tuple[str, bytes]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, data in sorted(files):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = 0
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
    return raw.getvalue()


def freeze(works: list[Path], name: str, out: Path, deny: Path, prompts: Path = PROMPTS) -> dict:
    """Freeze one or more verified work folders into one set folder: vault.tar.gz, questions.json, the world config
    and the graph of each seed, the prompts, verify.json and manifest.json with the SHA-256 of each file."""
    out = Path(out)
    if (out / "manifest.json").exists():
        raise SynthError(f"a frozen set exists at {out}")
    terms = read_deny(deny)
    current = prompt_hashes(prompts)
    reports, files, questions, models, dates, vaults, extra = {}, [], [], set(), [], [], {}
    for work in map(Path, works):
        report = verify(work)
        if not report["passed"]:
            raise SynthError(f"{work}: verify failed {len(report['failures'])} checks; run `ragnarok synth verify`")
        graph = _load(work)
        if graph["vault"] in reports:
            raise SynthError(f"two work folders have the vault {graph['vault']}")
        reports[graph["vault"]] = report
        removed = set(report["removed"]["notes"])
        for note in graph["notes"]:
            if note["path"] not in removed:
                files.append((note["path"], (work / "vault" / note["path"]).read_bytes()))
        questions += set_questions(work)
        for v in [*_state(work, "notes.json").values(), *_state(work, "written.json").values()]:
            if any(current.get(n) != h for n, h in v.get("prompts", {}).items()):
                raise SynthError(f"{work}: notes or questions were written with other prompts than {prompts}")
            models.update(v.get("models", []))
            if v.get("date"):
                dates.append(v["date"])
        vaults.append(
            {
                "vault": graph["vault"],
                "seed": graph["seed"],
                "graph_hash": graph["hash"],
                "notes": report["notes"]["kept"],
                "questions": report["questions"]["kept"],
            }
        )
        extra[f"world-{graph['vault']}.json"] = (work / "world.json").read_bytes()
        extra[f"graph-{graph['vault']}.json"] = (work / "graph.json").read_bytes()
    hits = [(n, t) for n, data in files for t in deny_hits(n + "\n" + data.decode(errors="ignore"), terms)]
    hits += [(q["id"], t) for q in questions for t in deny_hits(q["question"], terms)]
    hits += [(n, t) for n, data in extra.items() for t in deny_hits(data.decode(), terms)]
    if hits:
        shown = ", ".join(f"{t} in {n}" for n, t in hits[:20])
        raise SynthError(f"the privacy check found {len(hits)} deny terms: {shown}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vault.tar.gz").write_bytes(_archive(files))
    write_set(out / "questions.json", name, "synthetic", name, questions)
    for rel, data in extra.items():
        (out / rel).write_bytes(data)
    shutil.copytree(prompts, out / "prompts", dirs_exist_ok=True)
    (out / "verify.json").write_text(json.dumps(reports, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    hashes = {
        p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "name": name,
        "kind": "synthetic",
        "generator_version": GENERATOR_VERSION,
        "frozen": _now()[:10],
        "models": sorted(models),
        "llm_dates": [min(dates), max(dates)] if dates else [],
        "vaults": vaults,
        "set_hash": set_hash(questions),
        "prompts": current,
        "files": hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
