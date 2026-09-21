"""Hard import boundary between the rewrite tests and the legacy source tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force headless matplotlib backend so tests never spawn GUI windows or steal focus:
os.environ["MPLBACKEND"] = "Agg"
try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

import pytest


REWRITE_ROOT = Path(__file__).resolve().parents[1]
REWRITE_SOURCE = (REWRITE_ROOT / "src").resolve()
REWRITE_PACKAGE = (REWRITE_SOURCE / "numga").resolve()
LEGACY_ROOT = REWRITE_ROOT.parent.resolve()
LEGACY_PACKAGE = (LEGACY_ROOT / "numga").resolve()


def _resolved_sys_path(entry: str) -> Path:
    return Path(entry or Path.cwd()).resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _module_origin(module: object) -> Path | None:
    filename = getattr(module, "__file__", None)
    return Path(filename).resolve() if filename else None


def _assert_rewrite_imports_only() -> None:
    offenders: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name != "numga" and not name.startswith("numga."):
            continue
        origin = _module_origin(module)
        if origin is not None and not _is_within(origin, REWRITE_PACKAGE):
            offenders.append(f"{name} from {origin}")
    if offenders:
        details = ", ".join(sorted(offenders))
        raise pytest.UsageError(
            "rewrite tests imported numga outside rewrite/src: " + details
        )


# Pytest may be launched from the repository root, which otherwise places the
# legacy package ahead of rewrite/src. Remove that accidental route and put the
# rewrite source directory first. Rewrite tests never import the sibling legacy
# package.
sys.path[:] = [
    str(REWRITE_SOURCE),
    *(
        entry
        for entry in sys.path
        if _resolved_sys_path(entry) not in {REWRITE_SOURCE, LEGACY_ROOT}
    ),
]

_assert_rewrite_imports_only()

import numga  # noqa: E402  (the import boundary must be installed first)

if not _is_within(Path(numga.__file__).resolve(), REWRITE_PACKAGE):
    raise pytest.UsageError(
        f"expected rewrite numga under {REWRITE_PACKAGE}, got {numga.__file__}"
    )


def pytest_runtest_setup() -> None:
    """Catch a legacy module smuggled in by a test or plugin."""

    _assert_rewrite_imports_only()


def pytest_runtest_teardown() -> None:
    """Keep the import invariant true for the following test as well."""

    _assert_rewrite_imports_only()
