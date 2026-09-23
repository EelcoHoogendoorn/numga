"""Test configuration: headless plotting, and a guard against running the whole suite by accident."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


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
        "--i-am-a-dunce-for-ignoring-instructions",
        action="store_true",
        default=False,
        help="Force running the full test suite (which takes >2 minutes).",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Block full test suite runs unless --i-am-a-dunce-for-ignoring-instructions is passed."""
    if config.getoption("--i-am-a-dunce-for-ignoring-instructions", False):
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
            "To force a full run, pass '--i-am-a-dunce-for-ignoring-instructions'."
        )


def pytest_configure(config: pytest.Config) -> None:
    """Only force headless Agg backend if --show-plot was not passed."""
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
