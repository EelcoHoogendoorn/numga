"""Numga 2.0 examples suite."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Force headless Agg backend when running CLI scripts outside interactive notebooks,
# so CLI execution never spawns native GUI windows or steals window focus:
def _is_interactive_notebook() -> bool:
    try:
        import matplotlib
        if "inline" in matplotlib.get_backend().lower():
            return True
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            if hasattr(ip, "kernel") or "IPKernelApp" in getattr(ip, "config", {}):
                return True
            if "google.colab" in str(type(ip)) or ip.__class__.__name__ in ("ZMQInteractiveShell", "Shell"):
                return True
    except Exception:
        pass
    return False


if not _is_interactive_notebook():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        pass

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
