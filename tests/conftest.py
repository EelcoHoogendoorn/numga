"""Test configuration: headless plotting, and a guard against running the whole suite by accident, with the one
sanctioned way to run it, `pytest tests --full-sweep`, before a commit."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Numerical threads a full sweep may use, so that it never saturates the machine.
SWEEP_THREADS = 4


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI flags for test execution."""
    parser.addoption(
        "--show-plots",
        "--show-plot",
        action="store_true",
        default=False,
        help="Display matplotlib figures instead of closing them headlessly.",
    )
    parser.addoption(
        "--full-sweep",
        action="store_true",
        default=False,
        help="Run the whole suite before a commit, in one process with at most "
             f"{SWEEP_THREADS} numerical threads, leaving the machine's other cores free.",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Block full test suite runs unless --full-sweep is passed."""
    if config.getoption("--full-sweep", False):
        return

    test_root = Path(__file__).parent.resolve()
    repo_root = test_root.parent.resolve()

    # Check whether the targets resolve to the full test root or repository roots
    targets = [arg for arg in config.args if not arg.startswith("-")]
    targets_all = False
    if not targets:
        targets_all = True
    else:
        for t in targets:
            try:
                p = Path(t).resolve()
                if p in {test_root, repo_root}:
                    targets_all = True
                    break
            except Exception:
                pass

    has_filter = bool(config.option.keyword or config.option.markexpr)

    if (targets_all and not has_filter) or len(items) > 200:
        raise pytest.UsageError(
            f"Full test suite run blocked ({len(items)} tests collected, takes >2 minutes).\n"
            "Target a specific file or directory (e.g. 'pytest tests/examples/...').\n"
            "Before a commit, run the whole suite as 'pytest tests --full-sweep'."
        )


def pytest_configure(config: pytest.Config) -> None:
    """Only force headless Agg backend if --show-plot was not passed; cap numerical threads for a full sweep."""
    if config.getoption("--full-sweep", False):
        # Set before numpy is imported, so that its linear algebra starts with this many threads.
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(name, str(SWEEP_THREADS))
    if not config.getoption("--show-plots", False):
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def _handle_test_plots(pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch):
    """Ensure tests calling plt.show() never block or steal focus unless --show-plots is set."""
    if not pytestconfig.getoption("--show-plots", False):
        try:
            import matplotlib.pyplot as plt
            monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
        except ImportError:
            pass


@pytest.fixture
def show_plot(pytestconfig: pytest.Config):
    """Helper fixture to conditionally display or close a figure based on --show-plot."""
    def _show(fig=None):
        try:
            import matplotlib.pyplot as plt
            if pytestconfig.getoption("--show-plots", False):
                plt.show()
            else:
                if fig is not None:
                    plt.close(fig)
                else:
                    plt.close("all")
        except ImportError:
            pass
    return _show
