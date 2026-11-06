"""Test configuration: double precision for JAX, headless plotting, a guard that keeps a run to the tests a change
could break, and the machine's pool for heavy runs (`examples/pool.py`), in which a test run is one job. The whole
suite runs once, just before a push to main, declared as `pytest tests --full-sweep=push-to-main`: every test file
then runs as a job of its own, as many at once as the pool admits."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

# One precision for JAX across the whole suite, double, read by JAX when it is first imported; no test switches it.
os.environ["JAX_ENABLE_X64"] = "1"

# A test run, and in the full sweep each test file, is one job of the pool: one core and 1 GB unless the file declares
# more, `pytestmark = pytest.mark.memory(gb=4)`. A job the pool kills for exceeding its memory runs again with twice
# as much, and a file learns the memory it finally needed, which later runs reserve up front.
JOB_CORES, JOB_MEMORY = 1, 1
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
        help=f"'--full-sweep={SWEEP_OCCASION}' runs the whole suite, once, just before a push to main.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Refuse a target holding the whole suite, the tests folder or one above it, before anything is collected. Then,
    under the pool, run the full sweep, a job for each test file; or make this run one job, as this process becomes
    the pool's command, which runs it again inside the job. Inside a job, or without the pool, the run goes on here."""
    here = config.invocation_params.dir
    targets = [(here / target.split("::")[0]).resolve() for target in config.args]
    sweeping = config.getoption("--full-sweep", None)
    for target, path in zip(config.args, targets):
        if not sweeping and TEST_ROOT.is_relative_to(path):
            raise pytest.UsageError(f"'{target}' holds the whole suite, with or without -k or -m. {TARGETED}")
    from examples import pool
    if not pool.active():
        return None
    files = sorted({file for path in targets for file in ([path] if path.is_file() else path.rglob("test_*.py"))})
    if sweeping:
        return sweep(files)
    # A run of one file learns the memory it needs under the file's name.
    key = str(files[0].relative_to(TEST_ROOT.parent)) if len(files) == 1 else ""
    memory = max((declared(file) for file in files), default=JOB_MEMORY)
    what = f"pytest {' '.join(config.invocation_params.args)} in {TEST_ROOT.parent.name}"
    os.execv(sys.executable, [sys.executable, "-I", pool.__file__, "run", "--cores", str(JOB_CORES), "--memory",
                              f"{memory:g}", "--what", what, "--learn", key, "--", sys.executable, *sys.orig_argv[1:]])


def declared(file: Path) -> float:
    """The memory in GB a test file declares it needs, `pytestmark = pytest.mark.memory(gb=4)`, or a job's own."""
    for node in ast.parse(file.read_text()).body:
        if isinstance(node, ast.Assign) and "pytestmark" in [ast.unparse(target) for target in node.targets]:
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call) and ast.unparse(call.func) == "pytest.mark.memory":
                    return max(JOB_MEMORY, *(float(ast.literal_eval(keyword.value)) for keyword in call.keywords))
    return JOB_MEMORY


def sweep(files: list[Path]) -> int:
    """Run every test file as a job of the pool, `pytest -q <file>` in a process of its own, as many at once as the
    pool admits and the longest first, by the seconds recorded for them; print each file's outcome as it finishes,
    then the output of every file that failed, the files that needed more memory than a job's own, and the counts.
    The exit status counts failures only: a file the pool ran again with more memory has not failed."""
    from examples import pool
    checkout = TEST_ROOT.parent
    recorded = json.loads(DURATIONS.read_text()) if DURATIONS.exists() else {}
    seconds = {}
    for nodeid, runs in recorded.items():
        file = nodeid.split("::")[0]
        seconds[file] = seconds.get(file, 0) + sum(runs[-1].get(phase, 0) for phase in ("setup", "call", "teardown"))
    keys = sorted((str(file.relative_to(checkout)) for file in files), key=lambda key: -seconds.get(key, 0))
    jobs = [{"command": [sys.executable, "-m", "pytest", "-q", f"--full-sweep={SWEEP_OCCASION}", key], "cwd": checkout,
             "cores": JOB_CORES, "memory": declared(checkout / key), "what": f"pytest {key} in {checkout.name}",
             "key": key} for key in keys]
    counts: dict[str, int] = {}
    failed, reran = {}, []
    for index, (job, status, output) in enumerate(pool.batch(jobs, f"pytest tests in {checkout.name}"), 1):
        last = output.strip().splitlines()[-1] if output.strip() else ""
        for count, outcome in re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)", last):
            counts[outcome] = counts.get(outcome, 0) + int(count)
        reran += [job["key"]] if "pool: killed" in output else []
        # A file with no test left after deselection has not failed.
        if status not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
            failed[job["key"]] = output
        print(f"[{index}/{len(jobs)}] {'failed' if job['key'] in failed else 'passed':<7}{job['key']}", flush=True)
    for key, output in failed.items():
        print(f"\n{'=' * 30} {key}\n{output}")
    learned = pool.learned()
    needs = {key: max(declared(checkout / key), learned.get(key, 0.0)) for key in keys}
    print(f"\nfiles that needed more memory than a job's {JOB_MEMORY} GB, ran again where marked:")
    for key, memory in sorted(needs.items()):
        if memory > JOB_MEMORY:
            print(f"{memory:>5g} GB  {key}{'  (ran again)' if key in reran else ''}")
    print(", ".join(f"{count} {outcome}" for outcome, count in counts.items()) + f"; {len(failed)} files failed, "
          f"{len(reran)} ran again with more memory")
    return pytest.ExitCode.TESTS_FAILED if failed else pytest.ExitCode.OK


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuse a run of more tests than any change's own, counted after -k and -m have deselected."""
    if not config.getoption("--full-sweep", None) and len(items) > MAX_TESTS:
        raise pytest.UsageError(f"{len(items)} tests collected, more than any change's own ({MAX_TESTS}). {TARGETED}")


def pytest_configure(config: pytest.Config) -> None:
    """The memory marker, which a test file declares at its top for the pool to read; headless unless --show-plots."""
    config.addinivalue_line("markers", "memory(gb): the memory in GB a test file needs, more than a job's own")
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
