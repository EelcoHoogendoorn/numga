"""Notebooks are committed without outputs: the repository stays small and diffs stay readable."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(path for path in ROOT.glob("examples/**/*.ipynb") if ".ipynb_checkpoints" not in path.parts)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: str(path.relative_to(ROOT)))
def test_notebook_has_no_outputs(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    dirty = [index for index, cell in enumerate(cells)
             if cell["cell_type"] == "code" and (cell["outputs"] or cell["execution_count"] is not None)]
    assert not dirty, f"cells with outputs: {dirty}; strip them with nbstripout or `jupyter nbconvert --clear-output`"
