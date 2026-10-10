"""Every tutorial page prints what it says it prints."""

import importlib.util
from pathlib import Path

import pytest

specification = importlib.util.spec_from_file_location(
    "verify", Path(__file__).parents[1] / "docs" / "tutorial" / "verify.py")
verify = importlib.util.module_from_spec(specification)
specification.loader.exec_module(verify)


@pytest.mark.parametrize("page", verify.pages(), ids=lambda page: page.relative_to(verify.TUTORIALS).as_posix())
def test_printed_blocks_match_the_code(page: Path):
    assert verify.refreshed(page) == page.read_text(), "run docs/tutorial/verify.py --refresh"
