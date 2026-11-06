"""Numga 2.0 examples suite."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

# A module of the examples run as a program, `python -m examples.<...>`, a scenario's main, runs as one job of the
# machine's pool for heavy runs, where the checkout has it (`tools/pool.py`), one core and 4 GB unless it declares
# more: this process becomes the pool's command, which runs the program again inside the job, and again with more
# memory should it need it, which the module then learns.
_arguments = getattr(sys, "orig_argv", [])
_module = _arguments[_arguments.index("-m") + 1] if "-m" in _arguments[:-1] else ""
_pool = Path(__file__).resolve().parents[1] / "tools" / "pool.py"
# Inside a job, the program runs here.
if _module.startswith("examples.") and _pool.exists() and "NUMGA_POOL_JOB" not in os.environ:
    _program = f"python {' '.join(_arguments[_arguments.index('-m'):])} in {_pool.parents[1].name}"
    os.execv(sys.executable, [sys.executable, "-I", str(_pool), "run", "--what", _program, "--learn", _module, "--",
                              sys.executable, *_arguments[1:]])

import numpy as np
from types import ModuleType
from typing import TYPE_CHECKING

from numga import NumpyContext

if TYPE_CHECKING:
    from numga import Algebra
    from numga.backend.context import Context

# Pictures go to the folder NUMGA_PLOTS names, where several checkouts share one; else to the repository's plots, or,
# where the examples are installed rather than checked out, as in Colab, to plots beside the notebook. The meshes the
# examples load ship with them.
ROOT = Path(__file__).resolve().parents[1]
PLOT_DIR = Path(os.environ.get("NUMGA_PLOTS", (ROOT if (ROOT / "pyproject.toml").exists() else Path.cwd()) / "plots"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
# The pictures the repository tracks, under their fixed names in the checkout's own plots.
PINNED_DIR = ROOT / "plots"
MODEL_DIR = Path(__file__).resolve().parent / "util" / "models"


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


# The executed copies by module, algebra and kind of context, each beside the algebra it was made for, which keeps that
# algebra alive and so its identity unique.
_instances: dict[tuple[str, int, type], tuple[Algebra, ModuleType]] = {}


def instantiate(module: str, ga: Algebra, context: type[Context] = NumpyContext) -> ModuleType:
    """A copy of a module written against an algebra `ga` and a context `ctx` that it does not define
    itself.

    The module declares `ga: Algebra` and `ctx: Context`, and derives its types and constants from
    them at top level. The context is of the given kind, NumPy unless another is asked for, in double
    precision. Each algebra and kind of context gets its own executed copy, made on the first request
    and kept, so instances coexist and nothing is rebound after loading. The copies are kept by the
    algebra itself, which its construction interns, so that an algebra built again, by a notebook
    beside the core that imports this, is the same object with the same copy; an algebra with
    factories of its own, such as the stock PGA3D, stays apart from the plain construction of its
    signature. The copy is not registered in sys.modules.
    """
    key = (module, id(ga), context)
    if key not in _instances:
        spec = importlib.util.find_spec(module)
        instance = importlib.util.module_from_spec(spec)
        instance.ga = ga
        instance.ctx = context(ga, dtype=np.float64)
        spec.loader.exec_module(instance)
        _instances[key] = (ga, instance)
    return _instances[key][1]
