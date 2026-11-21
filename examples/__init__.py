"""Shared execution routing, figure output directories, and algebra-generic module instantiation for
the examples suite.

When an example module is invoked directly with `python -m examples.<...>`, the package re-executes
the command inside the workspace resource pool (`tools/pool.py`) when available so concurrent runs
stay within CPU and memory budgets. Picture output separates latest named files (`PLOT_DIR`,
overridden by `NUMGA_PLOTS`), append-only numbered development history (`DUMP_DIRS` from
`NUMGA_PLOTS_DUMP`), and repository-tracked gallery figures (`PINNED_DIR`). `instantiate` loads
algebra-generic modules (`ga: Algebra`, `ctx: Context`) into isolated instances keyed by
`(module, ga, context)` so multiple algebras and backends can coexist in one process.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

# Direct `python -m examples.<...>` invocations re-exec under `tools/pool.py` when present,
# reserving one core and 4 GB by default and recording peak memory for future runs.
_arguments = getattr(sys, "orig_argv", [])
_module = _arguments[_arguments.index("-m") + 1] if "-m" in _arguments[:-1] else ""
_pool = Path(__file__).resolve().parents[1] / "tools" / "pool.py"
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

# Latest named figures go to `NUMGA_PLOTS` when explicitly set, to the workspace's shared `../plots`
# when running inside a multi-checkout workspace, to `<repo>/plots/latest` inside a standalone
# repository checkout, or to `./plots` when running from an installed copy such as Colab.
ROOT = Path(__file__).resolve().parents[1]
_SHARED_PLOTS = ROOT.parent / "plots"
PLOT_DIR = Path(
    os.environ.get(
        "NUMGA_PLOTS",
        (_SHARED_PLOTS if _SHARED_PLOTS.is_dir() else ROOT / "plots" / "latest")
        if (ROOT / "pyproject.toml").exists()
        else Path.cwd() / "plots",
    )
)
# Optional `NUMGA_PLOTS_DUMP` directories accumulate numbered copies of every saved figure during
# development without overwriting earlier iterations.
DUMP_DIRS = tuple(Path(folder) for folder in os.environ.get("NUMGA_PLOTS_DUMP", "").split(os.pathsep) if folder)
# Tracked figures for the README and documentation live directly under the checkout's `plots/`.
PINNED_DIR = ROOT / "plots"
MODEL_DIR = Path(__file__).resolve().parent / "util" / "models"


def auto_increment_path(path: Path | str) -> Path:
    """An unused `path` stays unchanged, while an existing path advances through `<stem>_00`,
    `<stem>_01`, and higher suffixes until an unused filename is found."""
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


# Cached `(ga, instance)` pairs keyed by `(module, id(ga), context, units)`; retaining `ga` keeps its
# object `id` live for the lifetime of the cache entry.
_instances: dict[tuple[str, int, type, bool], tuple[Algebra, ModuleType]] = {}


def instantiate(module: str, ga: Algebra, context: type[Context] = NumpyContext, *, units: bool = False) -> ModuleType:
    """Modules written generically over an algebra declare `ga: Algebra` and `ctx: Context` at top
    level without binding them.

    Each distinct `(module, ga, context, units)` tuple executes once in `float64` precision and caches its
    loaded module outside `sys.modules`. Because `Algebra` construction interns identical signatures
    while keeping custom-factory algebras such as `PGA3D` distinct, matching algebras share one
    loaded copy while distinct algebras and backends coexist cleanly.
    """
    key = (module, id(ga), context, units)
    if key not in _instances:
        spec = importlib.util.find_spec(module)
        instance = importlib.util.module_from_spec(spec)
        instance.ga = ga
        instance.ctx = context(ga, dtype=np.float64, units=units)
        spec.loader.exec_module(instance)
        _instances[key] = (ga, instance)
    return _instances[key][1]
