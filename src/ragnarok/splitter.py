"""Split a note into chunks at H1-H3 headings, join small sections, split large ones."""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Section:
    path: tuple[str, ...]  # heading path, outermost first
    lines: list[str]

    @property
    def parent(self) -> tuple[str, ...]:
        return self.path[:-1]

    def words(self) -> int:
        return sum(len(line.split()) for line in self.lines)


@dataclass
class Chunk:
    ord: int
    heading: str  # "<note title> > <heading path>"
    text: str


def sections(body: str) -> list[Section]:
    """Cut the body at H1-H3 headings. Headings inside code fences do not count."""
    result = [Section((), [])]
    path: list[str] = []
    in_fence = False
    for line in body.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            result.append(Section(tuple(path), []))
        else:
            result[-1].lines.append(line)
    return [s for s in result if s.path or s.words()]


def _join_small(secs: list[Section], min_words: int) -> list[Section]:
    """Join a section below min_words with the next section under the same parent (or below it)."""

    def in_branch(sec: Section, prev: Section) -> bool:
        return sec.path[: len(prev.parent)] == prev.parent

    out: list[Section] = []
    for sec in secs:
        if out:
            prev = out[-1]
            if prev.words() < min_words and in_branch(sec, prev) and len(sec.path) >= len(prev.path):
                prev.lines.extend([sec.path[-1]] if sec.path else [])
                prev.lines.extend(sec.lines)
                continue
        out.append(Section(sec.path, list(sec.lines)))
    # a small last section joins the one before it when it is in the same branch
    if len(out) > 1 and out[-1].words() < min_words and in_branch(out[-1], out[-2]):
        last = out.pop()
        out[-1].lines.extend([last.path[-1]] if last.path else [])
        out[-1].lines.extend(last.lines)
    return out


def _split_large(text: str, max_words: int) -> list[str]:
    """Split text at paragraph breaks into parts of at most max_words; split a long paragraph by words."""
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for para in re.split(r"\n\s*\n", text):
        n = len(para.split())
        if n == 0:
            continue
        if n > max_words:
            if current:
                parts.append("\n\n".join(current))
                current, count = [], 0
            words = para.split()
            parts.extend(" ".join(words[i : i + max_words]) for i in range(0, n, max_words))
            continue
        if count + n > max_words and current:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += n
    if current:
        parts.append("\n\n".join(current))
    return parts


def split(
    title: str,
    body: str,
    min_words: int = 80,
    max_words: int = 300,
    description: str | None = None,
    project: str | None = None,
    area: str | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for sec in _join_small(sections(body), min_words):
        heading = " > ".join((title, *sec.path))
        text = "\n".join(sec.lines).strip()
        if not text:
            if sec.path:  # heading with no text: keep the heading as the only text
                chunks.append(Chunk(len(chunks), heading, sec.path[-1]))
            continue
        for part in _split_large(text, max_words):
            chunks.append(Chunk(len(chunks), heading, part))
    if description:
        text = f"project: {project}. area: {area}. {description}"
        chunks.append(Chunk(len(chunks), f"{title} > description", text))
    return chunks


def chunk_note(cfg, meta, body: str) -> tuple[list[Chunk], int]:
    """Redact and split the body of a resolved note. Return (chunks, number of redactions)."""
    from .redact import redact

    body, count = redact(body)
    description = meta.description
    if description:
        description, n = redact(description)
        count += n
    chunks = split(meta.title, body, cfg.min_words, cfg.max_words, description, meta.project, meta.area)
    return chunks, count


def chunk_file(cfg, rel: str):
    """Read, resolve, redact and split one note. Return (meta, chunks, number of redactions)."""
    from .metadata import read_note, resolve

    fm, body = read_note(cfg.root / rel)
    meta = resolve(cfg, rel, fm, body)
    chunks, count = chunk_note(cfg, meta, body)
    return meta, chunks, count
