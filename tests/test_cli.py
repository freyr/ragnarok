import pytest

from ragnarok.cli import COMMANDS, build_parser

REQUIRED = {"search": ["a query"], "eval": ["mechanical"], "backfill": ["propose", "Studio"]}


def test_parser_knows_all_commands():
    parser = build_parser()
    for name in COMMANDS:
        assert parser.parse_args([name, *REQUIRED.get(name, [])]).command == name


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
