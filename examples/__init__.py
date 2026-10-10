"""Numga 2.0 examples suite."""

from __future__ import annotations

import importlib.util
import re

import numpy as np
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from numga import NumpyContext

if TYPE_CHECKING:
    from numga import Algebra
    from numga.backend.context import Context

PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def auto_increment_path(path: Path | str) -> Path:
    """Return an auto-incrementing file path to never overwrite existing files.

    If `file.png` exists, generates `file_00.png`, `file_01.png`, etc.
    If `file_01.png` exists, advances to `file_02.png`.
    """
    path = Path(path)
    if not path.exists():
        return path

    parent = path.parent
    stem = path.stem
    suffix = path.suffix

    match = re.search(r"^(.*)_(\d+)$", stem)
    if match:
        base_stem, num_str = match.groups()
        counter = int(num_str) + 1
        width = len(num_str)
    else:
        base_stem = stem
        counter = 0
        width = 2

    while True:
        candidate = parent / f"{base_stem}_{counter:0{width}d}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


@lru_cache(maxsize=None)
def instantiate(module: str, ga: Algebra, context: type[Context] = NumpyContext) -> ModuleType:
    """A copy of a module written against an algebra `ga` and a context `ctx` that it does not define
    itself.

    The module declares `ga: Algebra` and `ctx: Context`, and derives its types and constants from
    them at top level. The context is of the given kind, NumPy unless another is asked for, in double
    precision. Each algebra and kind of context gets its own executed copy, so instances coexist and
    nothing is rebound after loading. The copy is not registered in sys.modules.
    """
    spec = importlib.util.find_spec(module)
    instance = importlib.util.module_from_spec(spec)
    instance.ga = ga
    instance.ctx = context(ga, dtype=np.float64)
    spec.loader.exec_module(instance)
    return instance
