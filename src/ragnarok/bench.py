"""`ragnarok bench`: one command that measures the new code against the frozen system A, the baseline B and the
last run, on the fixture of the evaluation home, and gives the verdict of the deciders in <home>/bench.toml.

The systems: A, the per-question file of the before home (frozen, `a_file`); B, the baseline over MCP on the
fixture (`base`), cached in <home>/bench/cache/base-<fixture hash>.json; C, the new code with no `when` (`kb`);
Cwhen, the new code with the `when` of the question (`kb-when`); last, the system C of the newest earlier run.

bench.toml:

    sets = "<folder of the kind sets>"            # relative to the home, or absolute
    regression_sets = "<folder with draft-long.json>"
    a_file = "<per-question file of system A>"
    [[decider]]                                   # the lower end of the 95% interval of the gain must be above lower
    set = "now"
    measure = "answer_first"                      # a key of kb_eval.MECHANICAL
    first = "C"                                   # A, B, C, Cwhen or last
    second = "A"
    lower = 0.0
    [[decider]]
    kind = "filtered"                             # the filtered runner: at most max_regressed regressed questions
    max_regressed = 0

The gain follows the sign of the measure, as `evaluate.compare` does: first minus second for sign +1, second minus
first for sign -1. The pairs are by question id, in the order of the run, so that the intervals are the same as
those of `evaluate compare` on the per-question files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import tomllib
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from . import kb_eval
from .agent_eval import eval_config, eval_home, mirror_hash
from .config import Config
from .evaluate import bootstrap
from .search import question_kind

NAMES = ("A", "B", "C", "Cwhen", "last")
SYSTEM = {"B": "base", "C": "kb", "Cwhen": "kb-when"}  # the bench name of each system of the mechanical runner
GAINS = (("C", "A"), ("C", "B"), ("C", "last"), ("Cwhen", "A"))
STOP = "the fixture is not the indexed one: run `ragnarok bench index`"
SHORT = 12  # the characters of a short hash


class BenchError(Exception):
    """A bench run that cannot start: the message is one line for the user."""


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_config(home: Path) -> dict:
    """bench.toml of the home, with its paths resolved against the home."""
    path = home / "bench.toml"
    if not path.exists():
        raise BenchError(f"no bench config at {path}")
    conf = tomllib.loads(path.read_text())
    for key in ("sets", "regression_sets", "a_file"):
        if key not in conf:
            raise BenchError(f"{path} has no key {key}")
        conf[key] = home / Path(conf[key]).expanduser()
    return conf


def fixture_hash(cfg: Config) -> str:
    return mirror_hash(eval_home(cfg) / "vaults")


def record_index(cfg: Config, home: Path) -> str:
    """Write <home>/bench/index-state.json with the fixture hash, and give the hash."""
    digest = fixture_hash(cfg)
    state = home / "bench" / "index-state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"fixture_hash": digest}))
    return digest


def build_index(cfg: Config, home: Path) -> str:
    """Rebuild the new-code index of the fixture and update the baseline index, then record the fixture hash."""
    from .embed import OllamaEmbedder
    from .index import rebuild, update

    if not (cfg.baseline_checkout and cfg.baseline_home):
        raise BenchError("bench index needs baseline_checkout and baseline_home")
    ecfg = eval_config(cfg)
    conn = rebuild(ecfg)
    try:
        update(ecfg, conn, OllamaEmbedder(ecfg))
    finally:
        conn.close()
    subprocess.run(
        ["uv", "--directory", cfg.baseline_checkout, "run", "ragnarok", "index"],
        env={**os.environ, "RAGNAROK_HOME": cfg.baseline_home},
        check=True,
    )
    return record_index(cfg, home)


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def _mechanical(ecfg, conn, embedder, sets_dir: Path, base_call, systems, collect=None) -> tuple[dict, list[str]]:
    """The per-question rows of each system and the MEASURE, ERROR and QUERY lines of one mechanical run."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "per-question.json"
        lines = kb_eval.kb_mechanical(ecfg, conn, embedder, sets_dir, base_call, systems, out, collect)
        rows = json.loads(out.read_text())["systems"]
    return rows, [x for x in lines if x.startswith(("MEASURE", "ERROR", "QUERY"))]


def _base(home, ecfg, conn, embedder, sets_dir, base_call, fixture, set_hashes) -> tuple[list[dict], list[str], bool]:
    """The B rows: from the cache of the fixture when the set hashes are the same, else a baseline run. A run with
    an ERROR line is no measurement, so it does not go into the cache."""
    cache = home / "bench" / "cache" / f"base-{fixture}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        if data["set_hashes"] == set_hashes:
            return data["rows"], data["lines"], True
    rows, lines = _mechanical(ecfg, conn, embedder, sets_dir, base_call, ["base"])
    if not any(x.startswith("ERROR") for x in lines):
        cache.parent.mkdir(parents=True, exist_ok=True)
        data = {"fixture_hash": fixture, "set_hashes": set_hashes, "rows": rows["base"], "lines": lines}
        cache.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return rows["base"], lines, False


def _doc(conn, path: str | None) -> dict:
    row = conn.execute("SELECT type, class, validity FROM doc WHERE path = ?", (path,)).fetchone() if path else None
    return dict(zip(("type", "class", "validity"), row or (None, None, None), strict=True))


def diagnostics(ecfg: Config, conn, sets_dir: Path, collect: list[dict]) -> list[dict]:
    """One row for each query form of C: the kind of question of the rule list, the rule word, and for the hits
    without and with `when` the rank-1 note (path, type, class, validity) and the rank of the first top gold."""
    questions = {(p.stem, q["id"]): q for p in sorted(sets_dir.glob("*.json")) for q in json.loads(p.read_text())}
    by_question: dict[tuple, dict[str, list[dict]]] = {}
    for c in collect:
        by_question.setdefault((c["set"], c["id"]), {}).setdefault(c["system"], []).append(c)

    def info(q: dict, paths: list[str]) -> dict:
        gold = kb_eval.top_gold(q)
        first = paths[0] if paths else None
        rank = next((i for i, p in enumerate(paths, start=1) if p in gold), None)
        return {"path": first, **_doc(conn, first), "gold_rank": rank}

    rows = []
    for key, systems in by_question.items():
        q = questions[key]
        for plain, when in zip(systems.get("kb", []), systems.get("kb-when", []), strict=True):
            kind, _, word = question_kind(ecfg, plain["form"])
            rows.append(
                {
                    "set": key[0],
                    "id": key[1],
                    "form": plain["form"],
                    "kind": kind,
                    "word": word,
                    "plain": info(q, plain["paths"]),
                    "when": info(q, when["paths"]),
                }
            )
    return rows


# --- the report ---------------------------------------------------------------------------------------------------


def _keyed(rows: list[dict] | None) -> dict[tuple, dict]:
    return {(r["set"], r["id"]): r for r in rows or []}


def gain(first: dict, second: dict, keys: list[tuple], measure: str):
    """The gain of first over second on measure (sign of the measure), paired by key in the order of keys: mean,
    interval and the paired differences; None with no pair."""
    pairs = [(first[k][measure], second[k][measure]) for k in keys if k in first and k in second]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return None
    va, vb = [a for a, _ in pairs], [b for _, b in pairs]
    sign = kb_eval.MECHANICAL[measure][1]
    d, lo, hi = bootstrap(va, vb) if sign > 0 else bootstrap(vb, va)
    return d, lo, hi, (np.array(va) - np.array(vb)) * sign


def detectable(diffs) -> str:
    """The smallest detectable difference: 2.8 * sd(paired differences) / sqrt(n); n/a when sd is 0."""
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    return "n/a" if sd == 0 else f"{2.8 * sd / math.sqrt(len(diffs)):.2f}"


def _interval(g) -> str:
    return "n/a" if g is None else f"{g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}]"


def _tcv(doc: dict) -> str:
    return "/".join(str(doc.get(k) or "-") for k in ("type", "class", "validity"))


def _rank(r) -> str:
    return "-" if r is None else str(r)


def report(
    label: str,
    stamp: str,
    commit: str,
    fixture: str,
    set_hashes: dict,
    systems: dict[str, list[dict] | None],
    deciders: list[dict],
    regressed: list[str],
    last: dict | None,
    diag: list[dict],
    notes: list[str],
) -> str:
    """The text of report.md."""
    keyed = {n: _keyed(rows) for n, rows in systems.items()}
    order = [(r["set"], r["id"]) for r in systems["C"] or []]
    sets = list(dict.fromkeys(s for s, _ in order))
    measures = {
        s: [m for m in kb_eval.MECHANICAL if m in keyed["C"][next(k for k in order if k[0] == s)]] for s in sets
    }
    out = []
    if last is not None:
        if last.get("fixture_hash") != fixture:
            out.append(
                f"**The fixture hash differs from the last run "
                f"({str(last.get('fixture_hash'))[:SHORT]} → {fixture[:SHORT]}).**"
            )
        before = last.get("set_hashes", {})
        changed = sorted(n for n in {*before, *set_hashes} if before.get(n) != set_hashes.get(n))
        if changed:
            out.append(f"**The set hashes differ from the last run ({', '.join(changed)}).**")
        out.append("")
    out += [
        f"# Bench run {label}",
        "",
        f"- Date: {stamp}",
        f"- Commit: {commit}",
        f"- Fixture: {fixture[:SHORT]}",
        "- Sets: " + ", ".join(f"{n} {h[:SHORT]}" for n, h in set_hashes.items()),
        "",
        "## Values",
        "",
        "| Set | Measure | n | " + " | ".join(NAMES) + " |",
        "|---|---|---|" + "---|" * len(NAMES),
    ]
    for s in sets:
        keys = [k for k in order if k[0] == s]
        for m in measures[s]:
            cells = []
            for n in NAMES:
                vals = [keyed[n][k][m] for k in keys if k in keyed[n] and keyed[n][k].get(m) is not None]
                cells.append(f"{np.mean(vals):.2f}" if vals else "n/a")
            out.append(f"| {s} | {kb_eval.MECHANICAL[m][0]} | {len(keys)} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Gains (95% paired bootstrap; for a measure with sign -1 the gain is second minus first)",
        "",
        "| Set | Measure | " + " | ".join(f"{a}-{b}" for a, b in GAINS) + " |",
        "|---|---|" + "---|" * len(GAINS),
    ]
    for s in sets:
        keys = [k for k in order if k[0] == s]
        for m in measures[s]:
            cells = [_interval(gain(keyed[a], keyed[b], keys, m)) for a, b in GAINS]
            out.append(f"| {s} | {kb_eval.MECHANICAL[m][0]} | " + " | ".join(cells) + " |")
    out += ["", "## Deciders", "", "| Decider | Value | Threshold | Result |", "|---|---|---|---|"]
    power, verdict = [], True
    for d in deciders:
        if d.get("kind") == "filtered":
            ok = len(regressed) <= d["max_regressed"]
            out.append(
                f"| filtered regressed | {len(regressed)} | <= {d['max_regressed']} | {'pass' if ok else 'fail'} |"
            )
        else:
            keys = [k for k in order if k[0] == d["set"]]
            g = gain(keyed[d["first"]], keyed[d["second"]], keys, d["measure"])
            ok = g is not None and g[1] > d["lower"]  # no pair is no measurement: it cannot pass
            name = f"{d['set']} {kb_eval.MECHANICAL[d['measure']][0]} {d['first']}-{d['second']}"
            out.append(f"| {name} | {_interval(g)} | > {d['lower']:+.2f} | {'pass' if ok else 'fail'} |")
            n = 0 if g is None else len(g[3])
            power.append(f"- {name}: smallest detectable difference {'n/a' if g is None else detectable(g[3])} (n={n})")
        verdict = verdict and ok
    out += ["", f"**Verdict: {'go' if verdict else 'no-go'}**", "", "## Power", "", *power]
    if notes or regressed:
        out += ["", "## Run notes", "", *(f"- {x}" for x in notes)]
        out += [f"- filtered regressed: {', '.join(regressed)}"] if regressed else []
    out += ["", "## Diagnostics", "", "### C against the last run", ""]
    if last is None:
        out.append("No earlier run.")
    else:
        changed = [
            f"- {k[1]} {kb_eval.MECHANICAL[m][0]} {keyed['last'][k][m]:.2f} → {keyed['C'][k][m]:.2f}"
            for k in order
            if k in keyed["last"]
            for m in measures[k[0]]
            if keyed["last"][k].get(m) != keyed["C"][k][m]
        ]
        out += changed or ["No change."]
    out += ["", "### Query forms of C: the kind of question of the rule list", "", "| Set | Forms |", "|---|---|"]
    for s in dict.fromkeys(r["set"] for r in diag):
        kinds = Counter(r["kind"] for r in diag if r["set"] == s)
        out.append(f"| {s} | {', '.join(f'{k} {n}' for k, n in sorted(kinds.items()))} |")

    def not_gold(info: dict) -> bool:
        return info["path"] is not None and info["gold_rank"] != 1

    plain = Counter(_tcv(r["plain"]) for r in diag if not_gold(r["plain"]))
    when = Counter(_tcv(r["when"]) for r in diag if not_gold(r["when"]))
    out += [
        "",
        "### Rank-1 notes that are not gold, by type/class/validity",
        "",
        "| Type/class/validity | Without when | With when |",
        "|---|---|---|",
        *(f"| {t} | {plain[t]} | {when[t]} |" for t in sorted({*plain, *when})),
    ]
    losers = {
        k
        for k in order
        if k in keyed["A"]
        and "answer_first" in keyed["A"][k]
        and keyed["C"][k]["answer_first"] < keyed["A"][k]["answer_first"]
    }
    out += [
        "",
        "### Non-gold rank-1 forms of the questions where C loses answer-first against A",
        "",
        "| Id | Form | Kind | Rank 1 without when | Type/class/validity | Gold rank | Rank 1 with when | Gold rank |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in diag:
        if (r["set"], r["id"]) in losers and not_gold(r["plain"]):
            p, w = r["plain"], r["when"]
            out.append(
                f"| {r['id']} | {r['form']} | {r['kind']} | {p['path']} | {_tcv(p)} | {_rank(p['gold_rank'])} | "
                f"{w['path']} | {_rank(w['gold_rank'])} |"
            )
    return "\n".join(out) + "\n"


def run(cfg: Config, home: Path, label: str, base_call, embedder) -> str:
    """One bench run: check the index state, take B from the cache or run it, run C and Cwhen with the collector,
    run the filtered runner, and write per-question.json, diagnostics.json and report.md in a new run folder.
    Give the report."""
    from .index import connect

    conf = read_config(home)
    ecfg = eval_config(cfg)
    fixture = fixture_hash(cfg)
    state = home / "bench" / "index-state.json"
    if not state.exists() or json.loads(state.read_text()).get("fixture_hash") != fixture:
        raise BenchError(STOP)
    sets_dir, regression = conf["sets"], conf["regression_sets"] / "draft-long.json"
    kind_hashes = {p.name: _sha(p) for p in sorted(sets_dir.glob("*.json"))}
    set_hashes = {**kind_hashes, regression.name: _sha(regression)}
    runs_dir = home / "bench" / "runs"
    earlier = sorted(p for p in runs_dir.glob("*") if (p / "per-question.json").exists())
    last = json.loads((earlier[-1] / "per-question.json").read_text()) if earlier else None
    stamp = _stamp()
    run_dir = runs_dir / f"{stamp}-{label}"
    if run_dir.exists():
        raise BenchError(f"the run folder {run_dir} exists")
    collect: list[dict] = []
    conn = connect(ecfg)
    try:
        base, notes, _ = _base(home, ecfg, conn, embedder, sets_dir, base_call, fixture, kind_hashes)
        new, new_notes = _mechanical(ecfg, conn, embedder, sets_dir, None, ["kb", "kb-when"], collect)
        f_lines, regressed = kb_eval.filtered(ecfg, conn, embedder, json.loads(regression.read_text()), base_call)
        diag = diagnostics(ecfg, conn, sets_dir, collect)
    finally:
        conn.close()
    notes = [*notes, *new_notes, *(x for x in f_lines if x.startswith(("ERROR", "MAP")))]
    a_rows = json.loads(conf["a_file"].read_text())["systems"]["base"]
    systems = {
        "A": a_rows,
        "B": base,
        "C": new["kb"],
        "Cwhen": new["kb-when"],
        "last": (last or {}).get("systems", {}).get("kb"),
    }
    commit = _commit()
    extra = {
        "label": label,
        "commit": commit,
        "fixture_hash": fixture,
        "set_hashes": set_hashes,
        "filtered_regressed": regressed,
    }
    values = {"base": base, "kb": new["kb"], "kb-when": new["kb-when"]}
    kb_eval.write_per_question(run_dir / "per-question.json", "mechanical", kb_eval.MECHANICAL, values, extra)
    (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=1, ensure_ascii=False))
    text = report(
        label, stamp, commit, fixture, set_hashes, systems, conf.get("decider", []), regressed, last, diag, notes
    )
    (run_dir / "report.md").write_text(text)
    return text
