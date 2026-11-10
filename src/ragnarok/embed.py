"""Ollama embedder with the model profiles: query prefix, document template, native dimension and token limit."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Config


class EmbedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    query_prefix: str
    doc_template: str  # "{title}" is the note title and the heading path of the chunk
    native_dimension: int | None  # None: never send `dimensions`
    token_limit: int  # the input limit of the model in Ollama


GEMMA = Profile("task: search result | query: ", "title: {title} | text: ", 768, 8192)
QWEN3_QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

# The shortlisted models of the embedding model research. The token limits are the limits that Ollama 0.40.8 enforces
# with truncate false (bge-m3 and snowflake-arctic-embed2 refuse more than 2,048 tokens).
PROFILES = {
    "embeddinggemma-2:270m": GEMMA,
    "embeddinggemma-2:270m-bf16-text": GEMMA,
    "qwen3-embedding:0.6b": Profile(QWEN3_QUERY, "{title}\n", 1024, 32768),
    "qwen3-embedding:4b": Profile(QWEN3_QUERY, "{title}\n", 2560, 32768),
    "snowflake-arctic-embed2": Profile("query: ", "{title}\n", 1024, 2048),
    "bge-m3": Profile("", "{title}\n", 1024, 2048),
}
# a model with no profile keeps the prefixes of the default model and sends no `dimensions`
OTHER = Profile(GEMMA.query_prefix, GEMMA.doc_template, None, 2048)
CHARS_PER_TOKEN = 4


def profile(cfg: Config) -> Profile:
    """The profile of the model, with the query prefix and the document template of the config when it sets them."""
    p = PROFILES.get(cfg.model, OTHER)
    return Profile(
        p.query_prefix if cfg.query_prefix is None else cfg.query_prefix,
        p.doc_template if cfg.doc_template is None else cfg.doc_template,
        p.native_dimension,
        p.token_limit,
    )


def max_chars(cfg: Config) -> int:
    """The estimated character limit of the text that goes to the model."""
    return profile(cfg).token_limit * CHARS_PER_TOKEN


def doc_input(cfg: Config, heading: str, text: str) -> str:
    return profile(cfg).doc_template.format(title=heading) + text


def template(cfg: Config) -> str:
    """The document template part of the cache key: how a chunk becomes the text that goes to the model."""
    return profile(cfg).doc_template


def input_hash(cfg: Config, heading: str, text: str) -> str:
    """The text hash of a chunk in the cache key: the model and the full text that goes to the model."""
    return hashlib.sha1(f"{cfg.model}\0{doc_input(cfg, heading, text)}".encode()).hexdigest()


class _TooLong(Exception):
    """Ollama refused the input: it is above the context length of the model."""


def _halves(text: str) -> tuple[str, str]:
    """Cut the text in two at the last space or line break up to the middle, else at the middle."""
    mid = len(text) // 2
    cut = max(text.rfind(" ", 0, mid + 1), text.rfind("\n", 0, mid + 1))
    if cut <= 0:
        return text[:mid], text[mid:]
    return text[:cut], text[cut + 1 :]


class OllamaEmbedder:
    def __init__(self, cfg: Config, batch: int = 32, timeout: float = 600):
        self.cfg = cfg
        self.batch = batch
        self.timeout = timeout

    def _request(self, texts: list[str]) -> list[list[float]]:
        body = {"model": self.cfg.model, "input": texts, "truncate": False, "keep_alive": self.cfg.keep_alive}
        native = profile(self.cfg).native_dimension
        if native is not None and self.cfg.dimension < native:
            body["dimensions"] = self.cfg.dimension
        req = urllib.request.Request(
            f"{self.cfg.ollama_url}/api/embed",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                vectors = json.load(resp)["embeddings"]
        except urllib.error.HTTPError as e:
            message = e.read().decode(errors="replace")
            if e.code in (400, 413) and "context length" in message:
                raise _TooLong from e
            raise EmbedError(f"Ollama returned {e.code}: {message[:200]}") from e
        except urllib.error.URLError as e:
            raise EmbedError(
                f"Ollama is not available at {self.cfg.ollama_url} ({e.reason}). Start it with `ollama serve` "
                "or `brew services start ollama`."
            ) from e
        if len(vectors) != len(texts) or any(len(v) != self.cfg.dimension for v in vectors):
            raise EmbedError(f"Ollama returned vectors that do not match the dimension {self.cfg.dimension}")
        return vectors

    def _post(self, texts: list[str]) -> list[list[float]]:
        """Embed the texts. When Ollama refuses a text above the context length, split the batch until the text is
        alone, then split the text in two and embed both halves; its vector is the mean of the two vectors."""
        try:
            return self._request(texts)
        except _TooLong:
            if len(texts) > 1:
                half = len(texts) // 2
                return self._post(texts[:half]) + self._post(texts[half:])
            if len(texts[0]) < 2:
                raise EmbedError("Ollama refused a text of one character as above the context length") from None
            first, second = (self._post([half])[0] for half in _halves(texts[0]))
            return [[(a + b) / 2 for a, b in zip(first, second, strict=True)]]

    def embed_docs(self, items: list[tuple[str, str]]) -> list[list[float]]:
        """Embed (heading, text) pairs with the document template."""
        texts = [doc_input(self.cfg, h, t) for h, t in items]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._post(texts[i : i + self.batch]))
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._post([profile(self.cfg).query_prefix + query])[0]
