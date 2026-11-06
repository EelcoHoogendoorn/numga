"""Test configuration: double precision for JAX, headless plotting, and a guard that keeps a run to the tests a change
could break. The whole suite runs once, just before a push to main, declared as
`pytest tests --full-sweep=push-to-main`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# One precision for JAX across the whole suite, double, read by JAX when it is first imported; no test switches it.
os.environ["JAX_ENABLE_X64"] = "1"

# Numerical threads a test run may use: the budget every heavy run shares, three quarters of the machine's cores.
THREADS = max(1, int(0.75 * (os.cpu_count() or 1)))
# The one occasion for the whole suite, the value --full-sweep takes; a bare --full-sweep is refused.
SWEEP_OCCASION = "push-to-main"
# More tests than any change's own: a run this broad is a sweep, however its targets are spelled.
MAX_TESTS = 200

TEST_ROOT = Path(__file__).parent.resolve()
TARGETED = ("Run the tests the change could break, by file or test function "
            "(e.g. 'pytest tests/examples/electromagnetism/test_constitutive.py'); "
            "the whole suite runs only once, just before a push to main.")

# Every run adds each test's seconds, setup, call and teardown, to a record kept by test id in the workspace that holds
# this checkout, beside the checkouts; a checkout with no workspace archive beside it keeps no record.
DURATIONS = TEST_ROOT.parents[1] / "archive" / "test_durations.json"
# This run's seconds by test id and phase, and the run's time, checkout and load, as the record gets them.
_seconds: dict[str, dict[str, object]] = {}
_run: dict[str, object] = {}


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
        help=f"'--full-sweep={SWEEP_OCCASION}' runs the whole suite, once, just before a push to main, in one process.",
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
    """Hold the machine's lock for heavy runs, within the budget every heavy run shares; headless unless --show-plots."""
    # Set before numpy is imported, so that its linear algebra starts with this many threads.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, str(THREADS))
    # Every test run takes the machine's lock for heavy runs, where there is one, and waits for a run that holds it.
    from examples import hold_heavy_run_lock
    hold_heavy_run_lock("pytest " + " ".join(config.invocation_params.args))
    if not config.getoption("--show-plots", False):
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pass


def pytest_sessionstart(session: pytest.Session) -> None:
    """Note when the run starts, from which checkout, and the machine's load before the run adds its own."""
    import time
    _run.update(at=time.strftime("%Y-%m-%dT%H:%M:%S"), checkout=TEST_ROOT.parent.name, load=round(os.getloadavg()[0], 2))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Note the seconds of each phase of a test, and its outcome: the call's, or a setup's that skipped or failed."""
    phases = _seconds.setdefault(report.nodeid, {})
    phases[report.when] = round(report.duration, 4)
    if report.when == "call" or report.outcome != "passed":
        phases.setdefault("outcome", report.outcome)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Add this run's seconds to the record, under a lock, so that runs in other checkouts may finish alongside."""
    if not _seconds or not DURATIONS.parent.is_dir():
        return
    import fcntl
    import json
    with os.fdopen(os.open(DURATIONS, os.O_RDWR | os.O_CREAT), "r+") as record:
        fcntl.flock(record, fcntl.LOCK_EX)
        text = record.read()
        kept = json.loads(text) if text else {}
        for nodeid, phases in _seconds.items():
            kept.setdefault(nodeid, []).append({**_run, **phases})
        record.seek(0)
        record.truncate()
        # One test a line, its runs in the order they happened.
        record.write("{\n" + ",\n".join(f"{json.dumps(nodeid)}: {json.dumps(runs)}" for nodeid, runs in sorted(kept.items())) + "\n}\n")


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
