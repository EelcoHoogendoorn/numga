"""Test configuration: headless plotting, and a guard that keeps a run to the tests a change could break. The whole
suite runs once, just before a push to main, declared as `pytest tests --full-sweep=push-to-main`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Numerical threads a full sweep may use, so that it never saturates the machine.
SWEEP_THREADS = 4
# The one occasion for the whole suite, the value --full-sweep takes; a bare --full-sweep is refused.
SWEEP_OCCASION = "push-to-main"
# More tests than any change's own: a run this broad is a sweep, however its targets are spelled.
MAX_TESTS = 200

TEST_ROOT = Path(__file__).parent.resolve()
TARGETED = ("Run the tests the change could break, by file or test function "
            "(e.g. 'pytest tests/examples/electromagnetism/test_constitutive.py'); "
            "the whole suite runs only once, just before a push to main.")


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
        choices=(SWEEP_OCCASION,),
        default=None,
        help=f"'--full-sweep={SWEEP_OCCASION}' runs the whole suite, once, just before a push to main, in one process "
             f"with at most {SWEEP_THREADS} numerical threads, leaving the machine's other cores free.",
    )


def pytest_collection(session: pytest.Session) -> None:
    """Refuse a target holding the whole suite, the tests folder or one above it, before anything is collected."""
    config = session.config
    if config.getoption("--full-sweep", None):
        return
    here = config.invocation_params.dir
    for target in config.args:
        if TEST_ROOT.is_relative_to((here / target.split("::")[0]).resolve()):
            raise pytest.UsageError(f"'{target}' holds the whole suite, with or without -k or -m. {TARGETED}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuse a run of more tests than any change's own, counted after -k and -m have deselected."""
    if not config.getoption("--full-sweep", None) and len(items) > MAX_TESTS:
        raise pytest.UsageError(f"{len(items)} tests collected, more than any change's own ({MAX_TESTS}). {TARGETED}")


def pytest_configure(config: pytest.Config) -> None:
    """Only force headless Agg backend if --show-plot was not passed; cap numerical threads for a full sweep."""
    if config.getoption("--full-sweep", None):
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
