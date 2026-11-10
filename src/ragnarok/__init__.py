"""Local RAG search over Obsidian vaults."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ragnarok-mcp")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0.0.0"
