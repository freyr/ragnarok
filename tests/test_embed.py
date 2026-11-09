import io
import json

from ragnarok.config import Config
from ragnarok.embed import OllamaEmbedder


def capture_bodies(monkeypatch, dimension):
    """Replace the HTTP call; return the list of request bodies that the embedder sends."""
    bodies = []

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        bodies.append(body)
        return io.BytesIO(json.dumps({"embeddings": [[0.0] * dimension for _ in body["input"]]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return bodies


def test_embed_calls_keep_the_model_loaded_by_default(monkeypatch):
    cfg = Config(dimension=4)
    bodies = capture_bodies(monkeypatch, 4)
    embedder = OllamaEmbedder(cfg)
    embedder.embed_docs([("h", "text")])
    embedder.embed_query("question")
    assert [b["keep_alive"] for b in bodies] == [-1, -1]


def test_embed_calls_send_the_configured_keep_alive_unchanged(monkeypatch):
    bodies = capture_bodies(monkeypatch, 4)
    for value in ("30m", 900):
        OllamaEmbedder(Config(dimension=4, keep_alive=value)).embed_query("question")
    assert [b["keep_alive"] for b in bodies] == ["30m", 900]


def test_config_refuses_a_keep_alive_that_ollama_refuses():
    import pytest

    for bad in ("-1", "900", True, 1.5, None):
        with pytest.raises(ValueError):
            Config(dimension=4, keep_alive=bad)
    for good in (-1, 0, 900, "30m", "1h"):
        assert Config(dimension=4, keep_alive=good).keep_alive == good
