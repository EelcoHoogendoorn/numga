"""Test configuration: double precision for JAX, headless plotting, a guard that keeps a run to the tests a change
could break, and, where the machine has a pool for heavy runs, a run spread over workers that are each one job of the
pool. The whole suite runs once, just before a push to main, declared as `pytest tests --full-sweep=push-to-main`."""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path

import pytest

# One precision for JAX across the whole suite, double, read by JAX when it is first imported; no test switches it.
os.environ["JAX_ENABLE_X64"] = "1"

# Under the machine's pool (`examples/pool.py`), a run's tests run in workers, each one job of the pool, of one core and
# 1 GB unless NUMGA_POOL_CORES or NUMGA_POOL_MEMORY ask otherwise. The process that spreads the tests over the workers
# runs none of them and reserves nothing.
WORKER_CORES, WORKER_MEMORY = 1, 1
# A test whose worker the pool kills for exceeding its reservation goes back into the queue with twice the memory,
# up to the pool's whole budget, and the memory it then needs is learned for later runs, which reserve it up front;
# `@pytest.mark.memory(gb=4)` declares a test's need instead. Each such retry replaces a worker, which xdist counts
# against its limit of restarts.
WORKER_RESTARTS = 100
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
# The memory in GB each test has learned it needs, more than a worker reserves, kept beside the durations.
MEMORY = TEST_ROOT.parents[1] / "archive" / "test_memory.json"
# This run's seconds by test id and phase, and the run's time, checkout and load, as the record gets them.
_seconds: dict[str, dict[str, object]] = {}
_run: dict[str, object] = {}
# The memory record as last read, with the time it was written; and the memory each test of this run reserved beyond a
# worker's own.
_learned: dict[str, object] = {"written": None, "amounts": {}}
_needed: dict[str, float] = {}
# When this run started, before which no kill of the pool's is this run's.
LOADED = time.time()


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
        help=f"'--full-sweep={SWEEP_OCCASION}' runs the whole suite, once, just before a push to main.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Refuse a target holding the whole suite, the tests folder or one above it, before anything starts. Under the
    pool, and unless `-n` says otherwise, spread the run over as many workers as it has test files and the pool has
    room for, at least one, each starting with one numerical thread. A worker, which runs this too, has its tests."""
    if hasattr(config, "workerinput"):
        return
    here = config.invocation_params.dir
    targets = [(here / target.split("::")[0]).resolve() for target in config.args]
    if not config.getoption("--full-sweep", None):
        for target, path in zip(config.args, targets):
            if TEST_ROOT.is_relative_to(path):
                raise pytest.UsageError(f"'{target}' holds the whole suite, with or without -k or -m. {TARGETED}")
    from examples import pool
    if pool.active() and config.pluginmanager.hasplugin("xdist") and config.option.numprocesses is None:
        files = sum(1 if path.is_file() else len(list(path.rglob("test_*.py"))) for path in targets)
        config.option.numprocesses = max(1, min(files, pool.room(*pool.requested(WORKER_CORES, WORKER_MEMORY))))
        os.environ.update(pool.threads(pool.requested(WORKER_CORES, WORKER_MEMORY)[0]))
        if config.option.maxworkerrestart is None:
            config.option.maxworkerrestart = WORKER_RESTARTS


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_make_scheduler(config: pytest.Config, log):
    """Under the pool, start a run's tests once every worker the pool has admitted for the run has collected them,
    and let each later worker join as the pool admits it, so that a run the pool admits only in part never waits for
    the rest."""
    from examples import pool
    if not pool.active() or config.getoption("dist") != "load":
        return None
    from xdist.scheduler import LoadScheduling

    class JoiningScheduling(LoadScheduling):
        @property
        def collection_is_completed(self) -> bool:
            if self.collection is not None:
                return True
            admitted = sum(job["admitted"] for job in pool.read() if job["run"] == run_name(os.getpid()))
            return bool(self.node2collection) and len(self.node2collection) >= admitted

    return JoiningScheduling(config, log)


def refuse_a_sweep(config: pytest.Config, count: int) -> None:
    """Refuse a run of more tests than any change's own, counted after -k and -m have deselected."""
    if not config.getoption("--full-sweep", None) and count > MAX_TESTS:
        raise pytest.UsageError(f"{count} tests collected, more than any change's own ({MAX_TESTS}). {TARGETED}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """A run in one process counts its tests as it collects them; workers leave the count to the process that spreads
    their tests."""
    if not hasattr(config, "workerinput"):
        refuse_a_sweep(config, len(items))


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids: list[str]) -> None:
    """The process that spreads the tests over workers counts them as each worker reports what it collected."""
    refuse_a_sweep(node.config, len(ids))


def run_name(spreader: int) -> str:
    """The name in the pool of a test run, by its checkout and the process that spreads its tests over workers."""
    return f"pytest in {TEST_ROOT.parent.name}, pid {spreader}"


def job_name(worker: str) -> str:
    """The name in the pool of a process that runs tests, by its worker id, or `run` for a run in one process."""
    return f"pytest {worker} in {TEST_ROOT.parent.name}"


def runs_tests(config: pytest.Config) -> bool:
    """Whether this process runs tests: a worker, or a run in one process; not the process that spreads them."""
    return hasattr(config, "workerinput") or not getattr(config.option, "tx", None)


def pytest_configure(config: pytest.Config) -> None:
    """A process that runs tests takes its reservation in the machine's pool, where there is one, and waits until the
    pool admits it; headless unless --show-plots."""
    config.addinivalue_line("markers", "memory(gb): the memory in GB a test needs, more than its worker reserves")
    if runs_tests(config):
        from examples import pool
        worker = getattr(config, "workerinput", {}).get("workerid")
        pool.hold(*pool.requested(WORKER_CORES, WORKER_MEMORY), what=job_name(worker or "run"),
                  run=run_name(os.getppid() if worker else os.getpid()), report=False, spared=False)
    if not config.getoption("--show-plots", False):
        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pass


def pytest_sessionstart(session: pytest.Session) -> None:
    """Note when the run starts, from which checkout, and the machine's load before the run adds its own."""
    _run.update(at=time.strftime("%Y-%m-%dT%H:%M:%S"), checkout=TEST_ROOT.parent.name, load=round(os.getloadavg()[0], 2))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Note the seconds of each phase of a test, and its outcome: the call's, or a setup's that skipped or failed; and
    the memory it reserved beyond a worker's own."""
    _needed.update((report.nodeid, value) for key, value in report.user_properties if key == "memory")
    phases = _seconds.setdefault(report.nodeid, {})
    phases[report.when] = round(report.duration, 4)
    if report.when == "call" or report.outcome != "passed":
        phases.setdefault("outcome", report.outcome)


def learned() -> dict[str, float]:
    """The memory in GB each test has learned it needs, read again whenever the record has changed."""
    written = MEMORY.stat().st_mtime_ns if MEMORY.exists() else None
    if written != _learned["written"]:
        _learned.update(written=written, amounts=json.loads(MEMORY.read_text()) if written else {})
    return _learned["amounts"]


def learn(nodeid: str, memory: float) -> None:
    """Record the memory in GB a test needs, under a lock, beside what other runs have recorded."""
    with os.fdopen(os.open(MEMORY, os.O_RDWR | os.O_CREAT), "r+") as record:
        fcntl.flock(record, fcntl.LOCK_EX)
        text = record.read()
        amounts = {**(json.loads(text) if text else {}), nodeid: memory}
        record.seek(0)
        record.truncate()
        record.write("{\n" + ",\n".join(f"{json.dumps(key)}: {value:g}" for key, value in sorted(amounts.items()))
                     + "\n}\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    """Name each test to the pool's watchdog, and hold the memory a test needs, the memory its marker declares or else
    the memory it has learned; a worker keeps what it has grown to, since what its tests leave behind stays."""
    from examples import pool
    pool.doing(item.nodeid)
    default = pool.requested(WORKER_CORES, WORKER_MEMORY)[1]
    declared = item.get_closest_marker("memory")
    needed = declared.kwargs["gb"] if declared else learned().get(item.nodeid, default)
    if needed > default:
        item.user_properties.append(("memory", needed))
        pool.grow_to(needed)
    yield


@pytest.hookimpl(optionalhook=True)
def pytest_handlecrashitem(crashitem: str, report: pytest.TestReport, sched) -> None:
    """A test whose worker the pool killed for exceeding its reservation has not failed: it goes back into the queue
    with twice the memory, which it learns, until it passes or its reservation reaches the pool's whole budget, where
    it fails with the memory it reached."""
    from examples import pool
    if not pool.active() or not MEMORY.parent.is_dir():
        return
    # The pool's record names the worker it killed; the test xdist names as the one the worker was running is the
    # one sent back.
    worker = job_name(report.node.gateway.id)
    killed = [kill for kill in pool.kills(run_name(os.getpid())) if kill["what"] == worker and kill["time"] >= LOADED]
    if not killed:
        return
    reached, reserved = killed[-1]["footprint"], killed[-1]["reserved"]
    if reserved >= pool.MEMORY:
        report.longrepr = (f"{crashitem} needs more memory than the pool's whole budget: the pool killed it at "
                           f"{reached:.1f} GB, with {reserved:g} GB reserved")
        return
    learn(crashitem, pool.doubled(reserved))
    sched.mark_test_pending(crashitem)
    report.outcome = "rerun"


def pytest_report_teststatus(report: pytest.TestReport, config: pytest.Config):
    """A test the pool sent back into the queue with more memory is a rerun, not a failure."""
    if report.outcome == "rerun":
        return "rerun", "R", ("RERUN", {"yellow": True})
    return None


def pytest_terminal_summary(terminalreporter) -> None:
    """Name every test of the run that needed more memory than a worker reserves, with the memory it reserved."""
    if _needed:
        terminalreporter.write_sep("-", "tests that needed more memory than a worker reserves")
        for nodeid, memory in sorted(_needed.items()):
            terminalreporter.write_line(f"{memory:>5g} GB  {nodeid}")


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Add this run's seconds to the record, under a lock, so that runs in other checkouts may finish alongside; a
    worker leaves the record to the process that spread the tests, which hears every test's report."""
    if not _seconds or not DURATIONS.parent.is_dir() or hasattr(session.config, "workerinput"):
        return
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
