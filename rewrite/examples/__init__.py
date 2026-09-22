"""Numga 2.0 examples suite."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Force headless matplotlib backend so examples never spawn GUI windows or steal focus:
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

PLOT_DIR = Path(__file__).resolve().parents[1] / "plots"


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
