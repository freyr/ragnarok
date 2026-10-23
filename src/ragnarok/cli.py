"""Command-line entry point."""

import argparse
import json
import sys
from pathlib import Path

from .config import load_config

COMMANDS = {
    "index": "Build or update the full index of all vaults",
    "status": "Print index statistics",
    "search": "Run a hybrid search from the command line",
    "inspect": "Print the resolved metadata of one note",
    "chunks": "Print splitter statistics",
    "serve": "Run the MCP server on stdio",
    "backfill": "Propose, describe and apply frontmatter for existing notes",
    "survey": "Print the facts that the path rules of a vault need",
    "eval": "Run the evaluation sets",
    "init": "Set up this computer: config, Ollama, model, index and the MCP server in Claude Code",
    "doctor": "Check the six parts of the setup and print the fix for each failure",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragnarok", description="Local RAG search over Obsidian vaults.")
    parser.add_argument("--config", type=Path, help="config file (default <home>/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    parsers = {name: sub.add_parser(name, help=help_text) for name, help_text in COMMANDS.items()}
    parsers["inspect"].add_argument("paths", nargs="*", help="note paths, relative to the vault root or absolute")
    sp = parsers["search"]
    sp.add_argument("query")
    sp.add_argument("--vault")
    sp.add_argument("--project")
    sp.add_argument("--area")
    sp.add_argument("--type")
    sp.add_argument("--no-archived", action="store_true", help="only notes with status active")
    sp.add_argument("--include-private", action="store_true", help="needs allow_private = true in the config")
    sp.add_argument("--limit", type=int, default=8)
    sp.add_argument("--mode", choices=["hybrid", "dense", "fts"], default="hybrid")
    sp.add_argument("--json", action="store_true", help="print one JSON object for each note")
    ep = parsers["eval"]
    ep.add_argument("kind", choices=["mechanical", "pool", "agent"])
    ep.add_argument("--set", action="append", dest="set_names", help="agent: a set name (default draft-long and hard)")
    ep.add_argument("--calls", type=int, default=8, help="agent: the limit of tool calls for each agent")
    ep.add_argument("--model", default="sonnet", help="agent: the Claude model of both agents")
    ep.add_argument("--limit", type=int, help="agent: only the first N questions of each set")
    ep.add_argument("--sets", type=Path, help="folder with the question sets (default <data_dir>/eval/sets)")
    ep.add_argument("--out", type=Path, help="folder for the results (default <data_dir>/eval/results)")
    ep.add_argument("--min-words", type=int, help="use a separate index with this minimum chunk size")
    parsers["index"].add_argument("--min-words", type=int, help="build a separate index with this minimum chunk size")
    parsers["index"].add_argument("--rebuild", action="store_true", help="delete the index and make it again")
    parsers["chunks"].add_argument("--stats", action="store_true", help="print totals for all vaults")
    parsers["chunks"].add_argument("--min-words", type=int, help="override min_words from the config")
    parsers["chunks"].add_argument("paths", nargs="*", help="print the chunks of these notes")
    parsers["init"].add_argument("--yes", action="store_true", help="accept each question with its default answer")
    parsers["init"].add_argument("--root", help="the vault root, with no question")
    bp = parsers["backfill"]
    bp.add_argument("action", choices=["propose", "export", "merge", "apply"])
    bp.add_argument("vault")
    bp.add_argument("files", nargs="*", help="merge: the JSON files with the descriptions from Claude")
    sv = parsers["survey"]
    sv.add_argument("vault", help="a folder name under the vault root; it need not be in the config yet")
    sv.add_argument("--top", type=int, default=30, help="the number of rows in each list (default 30)")
    sv.add_argument("--json", action="store_true", help="print one JSON object")
    return parser


def _relative(cfg, raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve().relative_to(cfg.root.resolve()).as_posix()
    return path.as_posix()


def cmd_inspect(cfg, args) -> int:
    from .metadata import resolve_file

    for raw in args.paths:
        meta = resolve_file(cfg, _relative(cfg, raw))
        print(json.dumps(meta.as_dict(), ensure_ascii=False))
    return 0


def cmd_chunks(cfg, args) -> int:
    from dataclasses import replace
    from statistics import median

    from .metadata import iter_notes
    from .splitter import chunk_file

    if args.min_words is not None:
        cfg = replace(cfg, min_words=args.min_words)
    for raw in args.paths:
        _meta, chunks, count = chunk_file(cfg, _relative(cfg, raw))
        for c in chunks:
            print(json.dumps({"ord": c.ord, "heading": c.heading, "words": len(c.text.split())}, ensure_ascii=False))
        print(f"redactions: {count}")
    if not args.stats:
        return 0
    totals = {"all": [], "normal": []}
    notes = redacted_notes = 0
    for rel in iter_notes(cfg):
        meta, chunks, count = chunk_file(cfg, rel)
        notes += 1
        redacted_notes += count > 0
        sizes = [len(c.text.split()) for c in chunks]
        totals["all"].extend(sizes)
        if meta.sensitivity != "private":
            totals["normal"].extend(sizes)
    print(f"notes: {notes}  notes with redactions: {redacted_notes}  min_words: {cfg.min_words}")
    for name, sizes in totals.items():
        small = sum(n < 30 for n in sizes)
        print(f"{name:7} chunks: {len(sizes)}  under 30 words: {small}  median words: {median(sizes) if sizes else 0}")
    return 0


def _with_min_words(cfg, min_words):
    """A configuration with a different minimum chunk size uses its own index file."""
    from dataclasses import replace

    if min_words is None or min_words == cfg.min_words:
        return cfg
    path = cfg.index_path.with_name(f"{cfg.index_path.stem}-min{min_words}{cfg.index_path.suffix}")
    return replace(cfg, min_words=min_words, index_path=path)


def cmd_index(cfg, args) -> int:
    from .embed import EmbedError, OllamaEmbedder
    from .index import IndexMismatch, connect, update

    cfg = _with_min_words(cfg, args.min_words)

    if args.rebuild:
        for suffix in ("", "-wal", "-shm"):
            cfg.index_path.with_name(cfg.index_path.name + suffix).unlink(missing_ok=True)
    try:
        conn = connect(cfg)
    except IndexMismatch as e:
        print(e, file=sys.stderr)
        return 1
    try:
        result = update(cfg, conn, OllamaEmbedder(cfg), progress=lambda m: print(m, file=sys.stderr, flush=True))
    except EmbedError as e:
        print(f"ragnarok index: {e}. The notes indexed before the error stay in the index.", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(result.__dict__, default=str))
    return 0


def cmd_status(cfg, args) -> int:
    from .index import IndexMismatch, connect, status

    if not cfg.index_path.exists():
        print(f"No index at {cfg.index_path}. Run `ragnarok index`.", file=sys.stderr)
        return 1
    try:
        conn = connect(cfg)
    except IndexMismatch as e:
        print(e, file=sys.stderr)
        return 1
    from .backfill import coverage

    print(json.dumps({**status(cfg, conn), "frontmatter": coverage(cfg)}, indent=2))
    conn.close()
    return 0


def cmd_search(cfg, args) -> int:
    import time

    from .embed import EmbedError, OllamaEmbedder
    from .index import IndexMismatch, connect
    from .search import PrivateNotAllowed, search

    if not cfg.index_path.exists():
        print(f"No index at {cfg.index_path}. Run `ragnarok index`.", file=sys.stderr)
        return 1
    try:
        conn = connect(cfg)
        start = time.monotonic()
        hits = search(
            cfg,
            conn,
            OllamaEmbedder(cfg),
            args.query,
            vault=args.vault,
            project=args.project,
            area=args.area,
            type=args.type,
            include_archived=not args.no_archived,
            include_private=args.include_private,
            limit=args.limit,
            mode=args.mode,
        )
    except (IndexMismatch, EmbedError, PrivateNotAllowed) as e:
        print(f"ragnarok search: {e}", file=sys.stderr)
        return 1
    ms = (time.monotonic() - start) * 1000
    for i, h in enumerate(hits, start=1):
        if args.json:
            print(json.dumps(h.as_dict(), ensure_ascii=False))
            continue
        print(f"{i}. {h.path}")
        print(f"   {h.project} | {h.type} | {h.status} | dense {h.dense_rank} fts {h.fts_rank}")
        print(f"   {h.heading}: {' '.join(h.text.split()[:30])}")
    print(f"{len(hits)} notes in {ms:.0f} ms", file=sys.stderr)
    return 0


def cmd_eval(cfg, args) -> int:
    from .embed import OllamaEmbedder
    from .evaluate import mechanical, pool
    from .index import connect

    cfg = _with_min_words(cfg, args.min_words)
    if not cfg.index_path.exists():
        print(f"No index at {cfg.index_path}. Run `ragnarok index` with the same --min-words.", file=sys.stderr)
        return 1
    args.sets = args.sets or cfg.data_dir / "eval" / "sets"
    args.out = args.out or cfg.data_dir / "eval" / "results"
    if args.kind == "agent":
        from .agent_eval import evaluate

        names = args.set_names or ["draft-long", "hard"]
        print(f"# agent evaluation, sets {', '.join(names)}, model {args.model}, {args.calls} tool calls")
        out = args.out / "agent" / f"{args.model}-calls{args.calls}"
        print("\n".join(evaluate(cfg, args.sets, names, out, args.calls, args.model, args.limit)))
        return 0
    conn = connect(cfg)
    if args.kind == "mechanical":
        print(f"# mechanical evaluation, min_words {cfg.min_words}, index {cfg.index_path.name}")
        print("\n".join(mechanical(cfg, conn, OllamaEmbedder(cfg), args.sets, args.out)))
    else:
        out = args.out / "pool.json"
        print(f"{pool(cfg, conn, OllamaEmbedder(cfg), args.sets, out)} questions written to {out}")
    return 0


def cmd_serve(cfg, args) -> int:
    import logging

    from .config import home_dir
    from .server import build

    logs = home_dir() / "logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(logs / "server.log")],
    )
    build(config_file=_config_file(args)).run("stdio")
    return 0


def cmd_backfill(cfg, args) -> int:
    from . import backfill

    if args.vault not in cfg.vaults:
        print(f"Unknown vault {args.vault}. The vaults are {', '.join(cfg.vaults)}.", file=sys.stderr)
        return 1
    if args.action == "propose":
        proposal = backfill.propose(cfg, args.vault)
        path = backfill.write_proposal(cfg, args.vault, proposal)
        rows = proposal["rows"]
        print(
            f"{len(rows)} rows ({sum(r['uncertain'] for r in rows)} uncertain, "
            f"{sum(r['private'] for r in rows)} private) written to {path}"
        )
        for rel in proposal["malformed"]:
            print(f"not in the proposal, correct its frontmatter by hand: {rel}", file=sys.stderr)
        return 0
    path = backfill.proposal_path(cfg, args.vault)
    if not path.exists():
        print(f"No proposal at {path}. Run `ragnarok backfill propose {args.vault}`.", file=sys.stderr)
        return 1
    if args.action == "export":
        out, count = backfill.write_export(cfg, args.vault, backfill.read_proposal(cfg, args.vault))
        print(f"{count} rows written to {out}")
        return 0
    if args.action == "merge":
        return _merge(cfg, args)
    result = backfill.apply(cfg, args.vault, backfill.read_proposal(cfg, args.vault))
    print(f"archive {result.archive}\n{result.written} notes written, {result.keys} keys added")
    for line in result.skipped:
        print(f"skipped {line}", file=sys.stderr)
    return 0


def _merge(cfg, args) -> int:
    from . import backfill

    descriptions = {}
    for raw in args.files:
        try:
            data = json.loads(Path(raw).expanduser().read_text())
            if not isinstance(data, dict):
                raise ValueError("the top level must be an object")
            for rel, item in data.items():
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("description"), str)
                    or not isinstance(item.get("type", ""), str)
                ):
                    raise ValueError(
                        f"the entry for {rel} must be an object with a text description and an optional text type"
                    )
        except (OSError, ValueError) as exc:
            print(f"Cannot read {raw}: {exc}", file=sys.stderr)
            return 1
        descriptions.update(data)
    proposal = backfill.read_proposal(cfg, args.vault)
    result = backfill.merge_descriptions(proposal, descriptions)
    backfill.write_proposal(cfg, args.vault, proposal)
    print(
        f"{result.rows} rows, {result.merged} descriptions merged, {len(result.missing)} missing, "
        f"{len(result.too_long)} too long, {result.redacted} redacted, {result.types} types changed, "
        f"{len(result.unknown)} unknown paths"
    )
    for label, paths in (("missing", result.missing), ("too long", result.too_long), ("unknown", result.unknown)):
        for rel in paths:
            print(f"{label}: {rel}", file=sys.stderr)
    return 0


def cmd_survey(cfg, args) -> int:
    from .survey import format_text, survey

    if not (cfg.root / args.vault).is_dir():
        print(f"Unknown vault folder {args.vault} under {cfg.root}.", file=sys.stderr)
        return 1
    facts = survey(cfg, args.vault, args.top)
    print(json.dumps(facts, ensure_ascii=False, indent=2) if args.json else format_text(facts))
    return 0


def _config_file(args) -> Path:
    from .config import config_path

    return args.config or config_path()


def cmd_init(cfg, args) -> int:
    from .bootstrap import init
    from .config import home_dir

    def confirm(question: str) -> bool:
        if args.yes:
            print(f"{question} yes")
            return True
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")

    def ask(question: str, default: str) -> str:
        if args.root:
            return args.root
        if args.yes:
            return default
        return input(f"{question} [{default}]: ").strip() or default

    return init(home_dir(), _config_file(args), confirm, ask)


def cmd_doctor(cfg, args) -> int:
    from .bootstrap import doctor, print_checks
    from .config import home_dir

    return print_checks(doctor(cfg, _config_file(args), home_dir()))


HANDLERS = {
    "init": cmd_init,
    "doctor": cmd_doctor,
    "backfill": cmd_backfill,
    "survey": cmd_survey,
    "eval": cmd_eval,
    "serve": cmd_serve,
    "inspect": cmd_inspect,
    "chunks": cmd_chunks,
    "index": cmd_index,
    "status": cmd_status,
    "search": cmd_search,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        print(f"ragnarok {args.command}: not implemented yet", file=sys.stderr)
        return 2
    return handler(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
