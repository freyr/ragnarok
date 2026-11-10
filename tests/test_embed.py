import io
import json
import urllib.error

import pytest

from ragnarok.config import Config
from ragnarok.embed import EmbedError, OllamaEmbedder


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
        with pytest.raises(ValueError, match="keep_alive must be"):
            Config(dimension=4, keep_alive=bad)
    for good in (-1, 0, 900, "30m", "1h"):
        assert Config(dimension=4, keep_alive=good).keep_alive == good


QWEN_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

# the shortlist table of the model research: query prefix, document text of ("N > A > B", "body"), native dimension,
# token limit (Ollama 0.40.8 with truncate false, and the research table for the models that are not installed)
PROFILES = {
    "embeddinggemma-2:270m": ("task: search result | query: ", "title: N > A > B | text: body", 768, 8192),
    "embeddinggemma-2:270m-bf16-text": ("task: search result | query: ", "title: N > A > B | text: body", 768, 8192),
    "qwen3-embedding:0.6b": (QWEN_QUERY, "N > A > B\nbody", 1024, 32768),
    "qwen3-embedding:4b": (QWEN_QUERY, "N > A > B\nbody", 2560, 32768),
    "snowflake-arctic-embed2": ("query: ", "N > A > B\nbody", 1024, 2048),
    "bge-m3": ("", "N > A > B\nbody", 1024, 2048),
}


def test_each_model_profile_gives_its_prefixes_and_keeps_the_title_and_the_heading_path(monkeypatch):
    from ragnarok.embed import doc_input, profile

    for model, (query, doc, native, tokens) in PROFILES.items():
        cfg = Config(model=model, dimension=native)
        bodies = capture_bodies(monkeypatch, native)
        embedder = OllamaEmbedder(cfg)
        embedder.embed_query("q")
        embedder.embed_docs([("N > A > B", "body")])
        assert [b["input"] for b in bodies] == [[query + "q"], [doc]], model
        assert doc_input(cfg, "N > A > B", "body") == doc
        assert (profile(cfg).native_dimension, profile(cfg).token_limit) == (native, tokens), model


def test_the_default_profile_keeps_the_texts_and_the_cache_template_of_release_2_0():
    from ragnarok.embed import doc_input, input_hash, template

    cfg = Config()
    assert template(cfg) == "title: {title} | text: "
    assert doc_input(cfg, "N > A", "body") == "title: N > A | text: body"
    assert input_hash(cfg, "N > A", "body") == "a31c689a31571a68a93489ed5f0d4e67593a75ea"


def test_a_config_value_overrides_the_profile(monkeypatch):
    from ragnarok.embed import template

    cfg = Config(model="bge-m3", dimension=1024, query_prefix="q: ", doc_template="passage {title}: ")
    bodies = capture_bodies(monkeypatch, 1024)
    OllamaEmbedder(cfg).embed_query("x")
    OllamaEmbedder(cfg).embed_docs([("N > A", "body")])
    assert [b["input"] for b in bodies] == [["q: x"], ["passage N > A: body"]]
    assert template(cfg) == "passage {title}: "


def test_the_old_doc_prefix_key_still_sets_the_document_template(tmp_path, caplog):
    from ragnarok.config import load_config

    path = tmp_path / "config.toml"
    path.write_text('doc_prefix = "passage: {title}\\n"\n')
    cfg = load_config(path)
    assert cfg.doc_template == "passage: {title}\n"
    assert "doc_prefix" in caplog.text
    assert "doc_template" in caplog.text


def test_a_model_with_no_profile_keeps_the_default_prefixes_and_sends_no_dimensions(monkeypatch):
    from ragnarok.embed import profile

    bodies = capture_bodies(monkeypatch, 8)
    OllamaEmbedder(Config(model="other-model", dimension=8)).embed_docs([("N", "body")])
    assert bodies[0]["input"] == ["title: N | text: body"]
    assert "dimensions" not in bodies[0]
    assert profile(Config(model="other-model")).token_limit == 2048


def test_dimensions_is_sent_only_below_the_native_dimension(monkeypatch):
    sent = {}
    for dim in (768, 512, 256):
        bodies = capture_bodies(monkeypatch, dim)
        OllamaEmbedder(Config(dimension=dim)).embed_query("q")
        sent[dim] = bodies[0].get("dimensions")
    assert sent == {768: None, 512: 512, 256: 256}


def test_embed_calls_never_let_ollama_truncate(monkeypatch):
    bodies = capture_bodies(monkeypatch, 4)
    embedder = OllamaEmbedder(Config(dimension=4))
    embedder.embed_docs([("h", "text")])
    embedder.embed_query("question")
    assert [b["truncate"] for b in bodies] == [False, False]


def limited_ollama(monkeypatch, status, max_chars):
    """An Ollama that refuses a batch with a text above max_chars, as Ollama does above the context length.
    The vector of a text is [number of characters, 1.0]. Return the list of texts that it embedded."""
    embedded = []

    def fake_urlopen(req, timeout):
        texts = json.loads(req.data)["input"]
        if any(len(t) > max_chars for t in texts):
            body = io.BytesIO(b'{"error":"the input length exceeds the context length"}')
            raise urllib.error.HTTPError(req.full_url, status, "Bad Request", {}, body)
        embedded.extend(texts)
        return io.BytesIO(json.dumps({"embeddings": [[float(len(t)), 1.0] for t in texts]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return embedded


@pytest.mark.parametrize("status", [400, 413])
def test_a_text_above_the_context_length_is_split_in_two_and_embedded_again(monkeypatch, status):
    embedded = limited_ollama(monkeypatch, status, 40)
    cfg = Config(model="other-model", dimension=2, doc_template="")
    long = " ".join(["word"] * 12) + " end"  # 63 characters: halves of 29 and 33 characters
    vectors = OllamaEmbedder(cfg).embed_docs([("h", "short"), ("h", long)])
    assert embedded == ["short", " ".join(["word"] * 6), " ".join(["word"] * 6) + " end"]
    # one vector for each chunk: the mean of the vectors of the two halves
    assert vectors == [[5.0, 1.0], [31.0, 1.0]]


def test_another_http_error_is_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"bad model"}'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(EmbedError):
        OllamaEmbedder(Config(dimension=4)).embed_docs([("h", "a b c d"), ("h", "e f g h")])
    assert len(calls) == 1
