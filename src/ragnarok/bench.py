"""The bench: the index of a variant on a corpus."""

from __future__ import annotations

import sqlite3

from .config import Config, variant_hash
from .index import IndexMismatch, connect


def open_variant(cfg: Config, variant: str, corpus: str) -> tuple[Config, sqlite3.Connection]:
    """The configuration and the index of a variant on a corpus.

    The index keeps the hash of the variant settings: a change of the settings does not make the change scan rewrite
    the notes, so an index of other settings is refused. Raise KeyError for an unknown variant or corpus."""
    vcfg = cfg.variant(variant, corpus)
    other_settings = IndexMismatch(
        f"The index at {vcfg.index_path} was built with other settings of the variant {variant}. "
        f"Delete the folder {vcfg.index_path.parent} and run `ragnarok bench index` again."
    )
    try:
        conn = connect(vcfg)
    except IndexMismatch as e:
        # a changed model or dimension: connect() gives the advice of the default index, which does not apply here
        raise other_settings from e
    expected = variant_hash(vcfg)
    stored = conn.execute("SELECT value FROM meta WHERE key = 'variant_hash'").fetchone()
    if stored is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('variant_hash', ?)", (expected,))
        conn.commit()
    elif stored[0] != expected:
        conn.close()
        raise other_settings
    return vcfg, conn
