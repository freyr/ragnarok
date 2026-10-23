"""Setup and health checks of one computer: `ragnarok init` and `ragnarok doctor`.

A setup has six parts: config.toml, the Ollama binary, the Ollama service, the model, the index and the MCP
registration in Claude Code. `doctor` checks each part; `init` repairs each part in sequence and skips a part
that is already ok.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config, load_config

MCP_NAME = "ragnarok"


@dataclass
class Check:
    part: str
    ok: bool
    detail: str
    fix: str = ""


def serve_args(home: Path) -> list[str]:
    return ["uv", "--directory", str(home), "run", "ragnarok", "serve"]


def ollama_models(cfg: Config) -> list[str] | None:
    """The model names in Ollama, or None when Ollama does not answer."""
    try:
        with urllib.request.urlopen(f"{cfg.ollama_url}/api/tags", timeout=3) as resp:
            return [m["name"] for m in json.load(resp)["models"]]
    except (OSError, ValueError, KeyError):
        return None


def brew_service_running() -> bool:
    """True when `brew services` runs Ollama: it starts at login and does not stop with a terminal or a session."""
    if not shutil.which("brew"):
        return False
    out = subprocess.run(["brew", "services", "info", "--json", "ollama"], capture_output=True, text=True, timeout=60)
    try:
        info = json.loads(out.stdout)[0]
    except (ValueError, IndexError):
        return False
    return bool(info.get("running")) and info.get("status") == "started"


def wait_for(condition: Callable[[], bool], seconds: float = 15) -> bool:
    for _ in range(int(seconds * 2)):
        if condition():
            return True
        time.sleep(0.5)
    return condition()


def has_model(models: list[str], name: str) -> bool:
    return name in models or f"{name}:latest" in models


def mcp_command() -> str | None:
    """The command line of the registered ragnarok server in Claude Code, or None."""
    if not shutil.which("claude"):
        return None
    out = subprocess.run(["claude", "mcp", "get", MCP_NAME], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        return None
    command = re.search(r"^\s*Command:\s*(.+)$", out.stdout, re.M)
    args = re.search(r"^\s*Args:\s*(.*)$", out.stdout, re.M)
    if not command:
        return None
    return f"{command.group(1).strip()} {args.group(1).strip() if args else ''}".strip()


def index_notes(cfg: Config) -> tuple[int | None, str]:
    """The number of notes in the index, or None and the reason."""
    from .index import IndexMismatch, connect

    if not cfg.index_path.exists():
        return None, f"no index at {cfg.index_path}"
    try:
        conn = connect(cfg)
    except IndexMismatch as e:
        return None, str(e)
    try:
        return conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0], str(cfg.index_path)
    finally:
        conn.close()


def doctor(cfg: Config, config_file: Path, home: Path) -> list[Check]:
    checks = []
    if not config_file.exists():
        checks.append(Check("config", False, f"no {config_file}", "ragnarok init"))
    elif missing := [v for v in cfg.vaults if not (cfg.root / v).is_dir()]:
        checks.append(
            Check("config", False, f"no vault folder {', '.join(missing)} in {cfg.root}", f"edit {config_file}")
        )
    else:
        checks.append(Check("config", True, f"{config_file}, root {cfg.root}, vaults {', '.join(cfg.vaults)}"))

    binary = shutil.which("ollama")
    checks.append(Check("ollama", bool(binary), binary or "ollama is not installed", "brew install ollama"))

    models = ollama_models(cfg)
    service = brew_service_running()
    if models is not None and service:
        checks.append(Check("service", True, f"brew services runs Ollama, it answers at {cfg.ollama_url}"))
    elif models is not None:
        detail = (
            f"Ollama answers at {cfg.ollama_url}, but not from brew services: it stops with its terminal or session"
        )
        checks.append(Check("service", False, detail, "ragnarok init (it moves Ollama to brew services)"))
    else:
        checks.append(Check("service", False, f"no answer at {cfg.ollama_url}", "brew services start ollama"))
    if models is None:
        checks.append(Check("model", False, f"{cfg.model}: unknown, Ollama does not answer", "start Ollama first"))
    else:
        found = has_model(models, cfg.model)
        checks.append(
            Check("model", found, cfg.model if found else f"{cfg.model} is not pulled", f"ollama pull {cfg.model}")
        )

    notes, detail = index_notes(cfg)
    fix = "ragnarok index --rebuild" if cfg.index_path.exists() else "ragnarok index"
    checks.append(Check("index", bool(notes), f"{notes} notes in {detail}" if notes else detail, fix))

    command = mcp_command()
    expected = " ".join(serve_args(home))
    if command is None:
        detail = "claude is not installed" if not shutil.which("claude") else f"no MCP server {MCP_NAME} in Claude Code"
        checks.append(Check("mcp", False, detail, f"claude mcp add --scope user {MCP_NAME} -- {expected}"))
    else:
        ok = str(home) in command.split()
        detail = command if ok else f"{command} does not use {home}"
        checks.append(Check("mcp", ok, detail, f"ragnarok init (it registers {expected})"))
    return checks


def print_checks(checks: list[Check]) -> int:
    for c in checks:
        print(f"{'ok  ' if c.ok else 'FAIL'} {c.part:8} {c.detail}")
        if not c.ok:
            print(f"     {'':8} fix: {c.fix}")
    return 0 if all(c.ok for c in checks) else 1


def write_config(config_file: Path, example: Path, root: Path) -> list[str]:
    """Copy the template with the vault root and the vault folders that are in it."""
    vaults = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    text = example.read_text()
    text = re.sub(r"^root = .*$", f'root = "{root}"', text, count=1, flags=re.M)
    text = re.sub(r"^vaults = .*$", f"vaults = {json.dumps(vaults)}", text, count=1, flags=re.M)
    config_file.write_text(text)
    return vaults


def init(
    home: Path,
    config_file: Path,
    confirm: Callable[[str], bool],
    ask: Callable[[str, str], str],
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    def step(n: int, text: str) -> None:
        print(f"\n[{n}/6] {text}", flush=True)

    step(1, "configuration")
    if config_file.exists():
        print(f"{config_file} exists, no change")
    else:
        root = Path(ask("Vault root (the folder that holds the vault folders)", "~/Documents/Vaults")).expanduser()
        if not root.is_dir():
            print(f"{root} is not a folder", file=sys.stderr)
            return 1
        vaults = write_config(config_file, home / "config.example.toml", root.resolve())
        print(
            f"wrote {config_file} with the vaults {', '.join(vaults)}; edit it to remove a folder that is not a vault"
        )
    cfg = load_config(config_file)

    step(2, "Ollama")
    if not shutil.which("ollama"):
        print("Ollama is not installed. Install it with `brew install ollama`, then run init again.", file=sys.stderr)
        return 1
    print(shutil.which("ollama"))

    step(3, "Ollama service")
    if brew_service_running():
        print("brew services runs Ollama, no change")
    else:
        other = ollama_models(cfg) is not None
        question = (
            "Ollama runs outside brew services, so it stops with the terminal or session that started it. "
            "Stop it and start Ollama as a login service with `brew services start ollama`?"
            if other
            else "Start Ollama as a login service with `brew services start ollama`?"
        )
        if not confirm(question):
            print("ragnarok needs Ollama as a service: run `brew services start ollama`.", file=sys.stderr)
            return 1
        if other:
            run(["pkill", "-f", "ollama serve"], check=False)
            if not wait_for(lambda: ollama_models(cfg) is None, 10):
                print(
                    f"Ollama still answers at {cfg.ollama_url}; stop it by hand, then run init again.", file=sys.stderr
                )
                return 1
        run(["brew", "services", "start", "ollama"], check=True)
        answers = lambda: ollama_models(cfg) is not None  # noqa: E731
        if not wait_for(answers):
            # launchd can load the job without a start of the process; a kickstart starts it
            run(["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/sh.brew.ollama"], check=False)
        if not answers() and not wait_for(answers):
            print(f"Ollama does not answer at {cfg.ollama_url} after 15 s.", file=sys.stderr)
            return 1
    print(f"Ollama answers at {cfg.ollama_url}")

    step(4, f"model {cfg.model}")
    if has_model(ollama_models(cfg) or [], cfg.model):
        print("pulled, no change")
    else:
        run(["ollama", "pull", cfg.model], check=True)

    step(5, f"index {cfg.index_path}")
    from .embed import EmbedError, OllamaEmbedder
    from .index import IndexMismatch, connect, update

    try:
        conn = connect(cfg)
    except IndexMismatch as e:
        print(e, file=sys.stderr)
        return 1
    try:
        result = update(cfg, conn, OllamaEmbedder(cfg), progress=lambda m: print(m, flush=True))
    except EmbedError as e:
        print(f"{e}. The notes indexed before the error stay in the index; run init again.", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(json.dumps(result.__dict__, default=str))

    step(6, "MCP server in Claude Code")
    if not shutil.which("claude"):
        print("Claude Code is not installed. After you install it, run init again.", file=sys.stderr)
        return 1
    command = mcp_command()
    if command and str(home) in command.split():
        print(f"registered, no change: {command}")
    else:
        if command:
            if not confirm(f"Claude Code runs {MCP_NAME} as `{command}`. Replace it with this home folder?"):
                return 1
            run(["claude", "mcp", "remove", "--scope", "user", MCP_NAME], check=True)
        run(["claude", "mcp", "add", "--scope", "user", MCP_NAME, "--", *serve_args(home)], check=True)
        print("registered; restart Claude Code to load the server")

    print("\n# doctor")
    return print_checks(doctor(cfg, config_file, home))
