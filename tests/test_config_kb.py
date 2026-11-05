from dataclasses import replace
from pathlib import Path

import pytest

from ragnarok.config import Config, KbConfig, dump_config, load_config, rules_hash

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"

KINDS = {"domain", "system", "component", "resource", "api", "group", "person"} | {
    "client",
    "environment",
    "vendor",
    "feature",
    "process",
}
RELATIONS = {"partOf", "dependsOn", "providesApi", "consumesApi", "requests", "memberOf"}
DOC_TYPES = {
    "how-to": ("state", "subject+task"),
    "tutorial": ("state", "subject+task"),
    "reference": ("state", "subject+task"),
    "explanation": ("state", "subject+task"),
    "entity": ("state", "entity"),
    "drivers": ("state", "project"),
    "summary": ("state", "project"),
    "spec": ("state", "project"),
    "plan": ("state", "project"),
    "decision": ("record", "none"),
    "incident": ("record", "none"),
    "research": ("record", "none"),
    "measurement": ("record", "none"),
    "meeting": ("record", "none"),
    "review": ("record", "none"),
    "brainstorm": ("record", "none"),
    "log": ("record", "none"),
    "guide": ("none", "none"),
    "note": ("none", "none"),
    "template": ("none", "none"),
}

CUSTOM = """
[kb]
exact_limit = 100

[kb.kinds]
team = "local"
"api" = "backstage"

[kb.doc_types.runbook]
class = "state"
one_per = "subject"
required_sections = ["Steps"]
review_months = 6
standard = "local"

[kb.relation_types.runs]
src_kinds = ["team"]
dst_kinds = ["component"]
inverse = "runBy"

[kb.attributes.team.channel]
value_type = "string"
live = true
system = "chat"
skill = "chat-read"
scoped = false

[kb.authority]
canonical = 2.0

[trace]
retention_days = 7

[w0]
summary_chars = 500
private_paths = ["Notes/Private"]
band = [0.05, 0.4]
"""


def test_kb_tables_load_with_defaults(tmp_path):
    cfg = Config()
    assert set(cfg.kb.kinds) == KINDS
    assert set(cfg.kb.relation_types) == RELATIONS
    assert {name: (t.class_, t.one_per) for name, t in cfg.kb.doc_types.items()} == DOC_TYPES
    assert cfg.kb.attributes == {}
    assert cfg.kb.exact_limit == 25000
    a = cfg.kb.authority
    assert (a.canonical, a.review_overdue, a.archived, a.superseded, a.duplicate_identity) == (1.5, 0.8, 0.5, 0.5, 0.5)
    assert cfg.trace.retention_days == 90
    w0 = cfg.w0
    assert (w0.summary_chars, w0.private_paths, w0.cost_ratio, list(w0.band)) == (2000, [], 10, [0.03, 0.30])

    # the example file shows the same generic defaults
    shipped = load_config(EXAMPLE)
    assert (shipped.kb, shipped.trace, shipped.w0) == (cfg.kb, cfg.trace, cfg.w0)

    path = tmp_path / "config.toml"
    path.write_text(CUSTOM)
    loaded = load_config(path)
    assert loaded.kb.exact_limit == 100
    assert set(loaded.kb.kinds) == {"team", "api"}  # a table replaces the default list
    assert (loaded.kb.doc_types["runbook"].class_, loaded.kb.doc_types["runbook"].review_months) == ("state", 6)
    assert loaded.kb.relation_types["runs"].inverse == "runBy"
    assert loaded.kb.attributes["team"]["channel"].value_type == "string"
    assert (loaded.kb.authority.canonical, loaded.kb.authority.archived) == (2.0, 0.5)
    assert loaded.trace.retention_days == 7
    assert (loaded.w0.summary_chars, loaded.w0.private_paths, loaded.w0.cost_ratio) == (500, ["Notes/Private"], 10)


@pytest.mark.parametrize(
    "text",
    [
        "[kb]\nexact_limt = 1\n",
        "[kb.authority]\ncanonicl = 2.0\n",
        "[kb.doc_types.x]\nclass = 'state'\none_per = 'none'\nwhat = 1\n",
        "[kb.relation_types.x]\ninvers = 'y'\n",
        "[kb.attributes.k.a]\nvalue_typ = 'string'\n",
        "[trace]\nretention = 1\n",
        "[w0]\nsummary = 1\n",
    ],
)
def test_unknown_kb_key_fails(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError)):
        load_config(path)


def test_vocabulary_change_changes_rules_hash():
    base = Config(root=Path("/vaults"), vaults=["Notes"])
    assert rules_hash(base) == rules_hash(replace(base))
    assert rules_hash(base) == "d0b71fa70ee3f689f3d3bcdcea69ec1612e3c144"  # a default KB keeps the hash of an old index
    kinds = replace(base, kb=replace(base.kb, kinds={**base.kb.kinds, "team": "local"}))
    types = replace(base, kb=replace(base.kb, doc_types={}))
    relations = replace(base, kb=replace(base.kb, relation_types={}))
    attributes = replace(base, kb=replace(base.kb, attributes={"team": {}}))
    assert len({rules_hash(c) for c in (base, kinds, types, relations, attributes)}) == 5
    # a read-path value is not a vocabulary
    assert rules_hash(replace(base, kb=replace(base.kb, exact_limit=1))) == rules_hash(base)


def test_dump_config_round_trips_kb_tables(tmp_path):
    path = tmp_path / "in.toml"
    path.write_text(CUSTOM)
    cfg = load_config(path)
    out = tmp_path / "out.toml"
    out.write_text(dump_config(cfg, root="vaults", data_dir="data"))
    copy = load_config(out)
    assert (copy.kb, copy.trace, copy.w0) == (cfg.kb, cfg.trace, cfg.w0)
    assert copy.kb.attributes["team"]["channel"].live is True
    assert copy.kb.doc_types["runbook"].required_sections == ["Steps"]
    # the defaults round trip as well
    out.write_text(dump_config(Config(), root="vaults", data_dir="data"))
    assert load_config(out).kb == Config().kb
    assert Config().kb.entity_first is True
    # a false entity_first survives the round trip, so that evaluate prepare keeps it
    off = replace(cfg, kb=replace(cfg.kb, entity_first=False))
    out.write_text(dump_config(off, root="vaults", data_dir="data"))
    assert load_config(out).kb.entity_first is False


def test_default_relation_kinds_admit_the_spec_examples():
    rel = KbConfig().relation_types
    assert rel["requests"].src_kinds == ["doc", "client"]
    assert rel["requests"].dst_kinds == ["feature"]
    assert rel["memberOf"].src_kinds == ["person"]
    for name in ("partOf", "dependsOn", "providesApi", "consumesApi"):
        assert rel[name].src_kinds == [] and rel[name].dst_kinds == []
