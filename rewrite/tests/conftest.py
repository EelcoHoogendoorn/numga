"""Hard import boundary between the rewrite tests and the legacy source tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path



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
    rewrite_root = test_root.parent.resolve()
    repo_root = rewrite_root.parent.resolve()

    # Check whether the targets resolve to the full test root or repository roots
    targets = [arg for arg in config.args if not arg.startswith("-")]
    targets_all = False
    if not targets:
        targets_all = True
    else:
        for t in targets:
            try:
                p = Path(t).resolve()
                if p in {test_root, rewrite_root, repo_root}:
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
