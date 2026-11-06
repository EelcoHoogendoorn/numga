"""Numga 2.0 examples suite."""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import time

import numpy as np
from pathlib import Path
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


# One heavy run at a time on the machine. Where NUMGA_HEAVY_RUN_LOCK names a lock file, as the workspace's settings do,
# a scenario, a notebook run or a test run takes it before it starts, and a second one waits until the first is done;
# a run started inside one that holds it, its kernels and subprocesses, already holds it. Elsewhere nothing waits.
HEAVY_RUN_LOCK = os.environ.get("NUMGA_HEAVY_RUN_LOCK")
_heavy_run_lock = None


def hold_heavy_run_lock(what: str) -> None:
    """Wait for, then hold until this process ends, the machine's lock for heavy runs, if there is one."""
    global _heavy_run_lock
    if not HEAVY_RUN_LOCK or os.environ.get("NUMGA_HEAVY_RUN_LOCK_HELD") == "1" or _heavy_run_lock is not None:
        return
    import fcntl
    lock = open(HEAVY_RUN_LOCK, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        print(f"waiting for the heavy run holding {HEAVY_RUN_LOCK}: {lock.read().strip()}", file=sys.stderr)
        fcntl.flock(lock, fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid {os.getpid()}, since {time.strftime('%H:%M:%S')}: {what}\n")
    lock.flush()
    os.environ["NUMGA_HEAVY_RUN_LOCK_HELD"] = "1"
    _heavy_run_lock = lock


# A module of the examples run as a program, `python -m examples.<...>`, a scenario's main or the notebook runner, is
# a heavy run.
_arguments = getattr(sys, "orig_argv", [])
_module = _arguments[_arguments.index("-m") + 1] if "-m" in _arguments[:-1] else ""
if _module.startswith("examples."):
    hold_heavy_run_lock(" ".join(_arguments[_arguments.index("-m"):]))


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
