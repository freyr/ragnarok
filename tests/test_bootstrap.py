import subprocess

import pytest

from ragnarok import bootstrap
from ragnarok.config import load_config

EXAMPLE = 'root = "~/Documents/Vaults"\nvaults = ["A"]\ndata_dir = "data"\nmodel = "m"\ndimension = 8\n'


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.example.toml").write_text(EXAMPLE)
    root = tmp_path / "vaults"
    for name in ("Studio", "Home", ".obsidian"):
        (root / name).mkdir(parents=True)
    (root / "Home/note.md").write_text("# Note\nText of the note.\n")
    monkeypatch.setenv("RAGNAROK_HOME", str(home))
    return home


def test_write_config_lists_the_vault_folders(home, tmp_path):
    vaults = bootstrap.write_config(home / "config.toml", home / "config.example.toml", tmp_path / "vaults")
    assert vaults == ["Home", "Studio"]
    cfg = load_config(home / "config.toml")
    assert (cfg.root, cfg.vaults, cfg.index_path) == (tmp_path / "vaults", vaults, home / "data/index.db")


def fake_setup(monkeypatch, models, command, binaries=("ollama", "claude"), service=True):
    monkeypatch.setattr(bootstrap, "brew_service_running", lambda: service)
    monkeypatch.setattr(bootstrap, "ollama_models", lambda cfg: models)
    monkeypatch.setattr(bootstrap, "mcp_command", lambda: command)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: f"/bin/{name}" if name in binaries else None)


def test_doctor_names_each_failure_and_its_fix(home, monkeypatch):
    fake_setup(monkeypatch, None, None, binaries=("claude",), service=False)
    checks = {c.part: c for c in bootstrap.doctor(load_config(home / "config.toml"), home / "config.toml", home)}
    assert list(checks) == ["config", "ollama", "service", "model", "index", "mcp"]
    assert not any(c.ok for c in checks.values())
    assert checks["config"].fix == "ragnarok init"
    assert checks["ollama"].fix == "brew install ollama"
    assert checks["mcp"].fix.endswith(f"uv --directory {home} run ragnarok serve")


class FakeEmbedder:
    def __init__(self, cfg):
        self.cfg = cfg

    def embed_docs(self, items):
        return [[0.1] * self.cfg.dimension for _ in items]

    def embed_query(self, query):
        return [0.1] * self.cfg.dimension


def test_init_sets_up_each_part_then_skips_it(home, tmp_path, monkeypatch):
    state = {"models": None, "command": None, "service": False}
    monkeypatch.setattr(bootstrap, "brew_service_running", lambda: state["service"])
    monkeypatch.setattr(bootstrap, "ollama_models", lambda cfg: state["models"])
    monkeypatch.setattr(bootstrap, "mcp_command", lambda: state["command"])
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", FakeEmbedder)
    calls = []

    def run(cmd, check):
        calls.append(cmd)
        if cmd[:3] == ["brew", "services", "start"]:
            state["models"], state["service"] = [], True
        elif cmd[:2] == ["ollama", "pull"]:
            state["models"] = ["m:latest"]
        elif cmd[:3] == ["claude", "mcp", "add"]:
            state["command"] = " ".join(cmd[cmd.index("--") + 1 :])
        return subprocess.CompletedProcess(cmd, 0)

    def init():
        return bootstrap.init(home, home / "config.toml", lambda q: True, lambda q, d: str(tmp_path / "vaults"), run)

    assert init() == 0
    assert [c[:3] for c in calls] == [["brew", "services", "start"], ["ollama", "pull", "m"], ["claude", "mcp", "add"]]
    assert (home / "data/index.db").exists()
    calls.clear()
    assert init() == 0
    assert calls == []


def test_init_stops_without_ollama(home, tmp_path, monkeypatch, capsys):
    fake_setup(monkeypatch, None, None, binaries=())
    assert bootstrap.init(home, home / "config.toml", lambda q: True, lambda q, d: str(tmp_path / "vaults")) == 1
    assert "brew install ollama" in capsys.readouterr().err


def test_init_asks_before_it_replaces_another_registration(home, tmp_path, monkeypatch):
    fake_setup(monkeypatch, ["m"], "uv --directory /elsewhere run ragnarok serve")
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", FakeEmbedder)
    calls = []
    result = bootstrap.init(
        home,
        home / "config.toml",
        lambda q: False,
        lambda q, d: str(tmp_path / "vaults"),
        lambda c, check: calls.append(c),
    )
    assert (result, calls) == (1, [])


def test_init_moves_an_ollama_outside_brew_services_to_the_service(home, tmp_path, monkeypatch):
    state = {"models": ["m"], "service": False}
    fake_setup(monkeypatch, None, f"uv --directory {home} run ragnarok serve")
    monkeypatch.setattr(bootstrap, "brew_service_running", lambda: state["service"])
    monkeypatch.setattr(bootstrap, "ollama_models", lambda cfg: state["models"])
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", FakeEmbedder)
    calls = []

    def run(cmd, check):
        calls.append(cmd)
        if cmd[0] == "pkill":
            state["models"] = None
        elif cmd[:3] == ["brew", "services", "start"]:
            state["models"], state["service"] = ["m"], True
        return subprocess.CompletedProcess(cmd, 0)

    questions = []
    confirm = lambda q: questions.append(q) or True  # noqa: E731
    assert bootstrap.init(home, home / "config.toml", confirm, lambda q, d: str(tmp_path / "vaults"), run) == 0
    assert calls == [["pkill", "-f", "ollama serve"], ["brew", "services", "start", "ollama"]]
    assert "outside brew services" in questions[0]


def test_doctor_fails_for_an_ollama_outside_brew_services(home, monkeypatch):
    fake_setup(monkeypatch, ["m"], None, service=False)
    service = next(
        c
        for c in bootstrap.doctor(load_config(home / "config.toml"), home / "config.toml", home)
        if c.part == "service"
    )
    assert (service.ok, service.fix) == (False, "ragnarok init (it moves Ollama to brew services)")


def test_init_kickstarts_a_service_that_launchd_did_not_start(home, tmp_path, monkeypatch):
    state = {"models": None, "service": False}
    fake_setup(monkeypatch, None, f"uv --directory {home} run ragnarok serve")
    monkeypatch.setattr(bootstrap, "brew_service_running", lambda: state["service"])
    monkeypatch.setattr(bootstrap, "ollama_models", lambda cfg: state["models"])
    monkeypatch.setattr(bootstrap, "wait_for", lambda condition, seconds=15: condition())
    monkeypatch.setattr("ragnarok.embed.OllamaEmbedder", FakeEmbedder)
    calls = []

    def run(cmd, check):
        calls.append(cmd[0])
        if cmd[0] == "launchctl":
            state["models"], state["service"] = ["m"], True
        return subprocess.CompletedProcess(cmd, 0)

    assert bootstrap.init(home, home / "config.toml", lambda q: True, lambda q, d: str(tmp_path / "vaults"), run) == 0
    assert calls == ["brew", "launchctl"]
