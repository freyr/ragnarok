import tomllib
from pathlib import Path

import ragnarok


def test_version_is_the_package_version():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert ragnarok.__version__ == pyproject["project"]["version"]
