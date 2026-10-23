"""Ollama embedder with the document and query prefixes of the model."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request

from .config import Config


class EmbedError(RuntimeError):
    pass


def doc_input(cfg: Config, heading: str, text: str) -> str:
    return cfg.doc_prefix.format(title=heading) + text


def input_hash(cfg: Config, heading: str, text: str) -> str:
    """Cache key of a chunk: the model and the full text that goes to the model."""
    return hashlib.sha1(f"{cfg.model}\0{doc_input(cfg, heading, text)}".encode()).hexdigest()


class OllamaEmbedder:
    def __init__(self, cfg: Config, batch: int = 32, timeout: float = 600):
        self.cfg = cfg
        self.batch = batch
        self.timeout = timeout

    def _post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.cfg.model, "input": texts, "truncate": True}).encode()
        req = urllib.request.Request(
            f"{self.cfg.ollama_url}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                vectors = json.load(resp)["embeddings"]
        except urllib.error.HTTPError as e:
            raise EmbedError(f"Ollama returned {e.code}: {e.read().decode(errors='replace')[:200]}") from e
        except urllib.error.URLError as e:
            raise EmbedError(
                f"Ollama is not available at {self.cfg.ollama_url} ({e.reason}). Start it with `ollama serve` "
                "or `brew services start ollama`."
            ) from e
        if len(vectors) != len(texts) or any(len(v) != self.cfg.dimension for v in vectors):
            raise EmbedError(f"Ollama returned vectors that do not match the dimension {self.cfg.dimension}")
        return vectors

    def embed_docs(self, items: list[tuple[str, str]]) -> list[list[float]]:
        """Embed (heading, text) pairs with the document prefix."""
        texts = [doc_input(self.cfg, h, t) for h, t in items]
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            out.extend(self._post(texts[i : i + self.batch]))
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._post([self.cfg.query_prefix + query])[0]
