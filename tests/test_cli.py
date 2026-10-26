import pytest

from ragnarok.cli import COMMANDS, build_parser

REQUIRED = {"search": ["a query"], "eval": ["mechanical"], "backfill": ["propose", "Studio"], "survey": ["Studio"]}


def test_parser_knows_all_commands():
    parser = build_parser()
    for name in COMMANDS:
        assert parser.parse_args([name, *REQUIRED.get(name, [])]).command == name


def test_parser_requires_a_command():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_inspect_reports_a_folder_and_goes_on(vault, capsys):
    from ragnarok.cli import cmd_inspect

    args = build_parser().parse_args(["inspect", "Home/Heating", "Home/Heating/SUMMARY.md"])
    assert cmd_inspect(vault, args) == 1
    out = capsys.readouterr()
    assert "not a note: Home/Heating" in out.err
    assert '"project": "Heating"' in out.out
