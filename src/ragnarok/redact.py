"""Replace credential values with [REDACTED] before text is stored, embedded or returned."""

from __future__ import annotations

import re

MARK = "[REDACTED]"

# Whole-match patterns: the full token is a secret.
TOKEN_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
]

# key: value / key=value, where only the value is secret.
KEY_VALUE = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9$-])                 # not $PWD
    (?P<key>[\w-]*?(?:password|passwd|pwd|[_-]pass|token|secret|api[_-]?key|access[_-]?key|auth[_-]?key))
    (?![\w-])                          # the key ends here: not max_tokens, passive, data-token-slot
    (?P<sep>\s*[:=]\s*)
    (?P<q>["']?)
    (?P<value>[^\s"'`,;)]{4,})
    (?P=q)
    """
)

# Values that are references, placeholders, types or function calls, not literals.
NOT_LITERAL = re.compile(
    r"[^(]*\("  # function call
    r"|^(?:string|number|null|none|true|false|bool|boolean|int|str|read|write)\b"  # type or permission
    r"|^(?:!secret|\$\{|\$\(|\$[A-Z_]|<|\{\{|%|\*+$|x+$|\.\.\.|changeme$|redacted)",  # reference or placeholder
    re.I,
)

URL_CREDENTIALS = re.compile(r"(?P<pre>\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)(?P<value>[^\s@/]{3,})(?P<post>@)", re.I)


def redact(text: str) -> tuple[str, int]:
    """Return (redacted text, number of replacements)."""
    count = 0

    def token_sub(_m: re.Match) -> str:
        nonlocal count
        count += 1
        return MARK

    for pattern in TOKEN_PATTERNS:
        text = pattern.sub(token_sub, text)

    def kv_sub(m: re.Match) -> str:
        nonlocal count
        value = m.group("value")
        if NOT_LITERAL.match(value) or value == MARK:
            return m.group(0)
        count += 1
        return f"{m.group('key')}{m.group('sep')}{m.group('q')}{MARK}{m.group('q')}"

    text = KEY_VALUE.sub(kv_sub, text)

    def url_sub(m: re.Match) -> str:
        nonlocal count
        if NOT_LITERAL.match(m.group("value")):
            return m.group(0)
        count += 1
        return f"{m.group('pre')}{MARK}{m.group('post')}"

    text = URL_CREDENTIALS.sub(url_sub, text)
    return text, count
