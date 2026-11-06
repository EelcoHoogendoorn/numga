"""A machine-wide pool of cores and memory that every heavy run shares: each test worker, each notebook, each module
of the examples run as a program, and any command run through it, is one job of the pool.

The pool's budget is three quarters of the machine, of its cores and of its physical memory, so that the machine stays
usable beside it. A job reserves cores and memory before it starts. It waits while an earlier job is still waiting,
while the budget's free remainder does not fit it, or while the machine's available memory, which counts every other
program, falls short of what the job reserves, plus a margin, beyond what the running jobs have reserved and not yet
used. A watchdog holds each job to its reservation: it polls the resident memory of the job's whole process tree, and
kills the tree the moment it exceeds what the job reserved, naming the job and the amount. A job's numerical threads
equal its cores, and it runs at low priority.

The reservations live in a state file in the directory NUMGA_POOL names, written under a lock; a reservation whose
process no longer exists is dropped whenever the file is read, so a crash never leaks one. A process inside a job,
the kernel of a notebook or a subprocess of a test, belongs to that job and reserves nothing of its own. Where
NUMGA_POOL is not set, as in Colab or CI, nothing is reserved and nothing waits.

The module uses only the standard library and psutil, imported where the pool is used, and runs as a script as well
as a module of the examples:

    python -m examples.pool run [--cores N] [--memory GB] -- <command>   # any command, as one job
    python -m examples.pool status                                        # the jobs holding and waiting
    python -m examples.pool trace <file> [--seconds S | --until-idle]     # the pool, sampled twice a second
    python -m examples.pool chart <file>                                  # a trace drawn to the shared plots

A job declares more than its default with `--cores` and `--memory`, or with NUMGA_POOL_CORES and NUMGA_POOL_MEMORY in
its environment, up to MAX_MEMORY GB.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import fcntl
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

POOL = "NUMGA_POOL"
# Set inside a job, to the process that holds its reservation.
JOB = "NUMGA_POOL_JOB"
CORES_REQUESTED, MEMORY_REQUESTED = "NUMGA_POOL_CORES", "NUMGA_POOL_MEMORY"
# Runs from checkouts that predate the pool take this lock exclusively, one at a time; a job of the pool holds it
# shared, so that the two kinds of run exclude each other while jobs of the pool run side by side. A process that
# such a run started already holds it.
LEGACY_LOCK, LEGACY_LOCK_HELD = "NUMGA_HEAVY_RUN_LOCK", "NUMGA_HEAVY_RUN_LOCK_HELD"

GB = 2 ** 30
# The budget's cores; its memory, three quarters of the physical memory, is `budget_memory()`.
CORES = max(1, 3 * (os.cpu_count() or 1) // 4)
MAX_MEMORY = 8
# Memory the machine keeps available beyond every reservation.
MARGIN = 1
# How often a waiting job asks again, and how often a watchdog reads its job's memory and, less often, its tree.
WAIT_SECONDS, WATCH_SECONDS, TREE_SECONDS = 0.5, 0.1, 1.0
# How often a trace samples the pool, how long an empty pool ends a trace that waits for it, and how many samples a
# chart of it keeps before the first job and after the last.
TRACE_SECONDS, IDLE_SECONDS, CHART_MARGIN = 0.5, 10, 4
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS")
NICENESS = 10
# The characters of a command that name its job.
NAME_LENGTH = 100

# This process's reservation, once it holds one, its watchdog, and the lock of runs outside the pool it shares.
_held: dict = {}
_watchdog: list[subprocess.Popen] = []
_legacy_locks: list = []


def budget_memory() -> float:
    """The budget's memory in GB, three quarters of the machine's physical memory."""
    import psutil
    return 0.75 * psutil.virtual_memory().total / GB


def active() -> bool:
    """Whether a job started here takes a reservation of its own: the pool is set, and this is not inside a job."""
    return bool(os.environ.get(POOL)) and JOB not in os.environ


def requested(cores: int, memory: float) -> tuple[int, float]:
    """A job's cores and memory in GB: these defaults, unless NUMGA_POOL_CORES or NUMGA_POOL_MEMORY ask otherwise."""
    return int(os.environ.get(CORES_REQUESTED, cores)), float(os.environ.get(MEMORY_REQUESTED, memory))


def threads(cores: int) -> dict[str, str]:
    """The environment that gives NumPy's linear algebra and JAX `cores` numerical threads."""
    flags = [flag for flag in os.environ.get("XLA_FLAGS", "").split()
             if not flag.startswith("--xla_cpu_multi_thread_eigen=")]
    return {**{name: str(cores) for name in THREADS},
            "XLA_FLAGS": " ".join((*flags, f"--xla_cpu_multi_thread_eigen={str(cores > 1).lower()}"))}


# --- the state file

def _directory() -> Path:
    directory = Path(os.environ[POOL])
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _alive(job: dict) -> bool:
    """Whether a job's process still runs: its pid, started when the job's was, not a later process given the pid,
    nor one that has ended and waits for its parent to collect its exit status."""
    import psutil
    try:
        process = psutil.Process(job["pid"])
        return process.create_time() == job["started"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def read() -> list[dict]:
    """The pool's jobs, in the order they asked, holding or waiting, those whose processes are gone dropped."""
    path = _directory() / "jobs.json"
    return [job for job in (json.loads(path.read_text()) if path.exists() else []) if _alive(job)]


@contextlib.contextmanager
def _ledger():
    """The pool's jobs under the pool's lock, written back when the block ends; the file is replaced whole, so that a
    reader without the lock always reads one consistent state."""
    directory = _directory()
    with open(directory / "jobs.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = read()
        yield jobs
        written = directory / f"jobs.{os.getpid()}.json"
        written.write_text(json.dumps(jobs, indent=1) + "\n")
        os.replace(written, directory / "jobs.json")


def resident(pid: int) -> float:
    """The resident memory in GB of a process and every process below it."""
    import psutil
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for member in tree:
        with contextlib.suppress(psutil.NoSuchProcess):
            total += member.memory_info().rss
    return total / GB


def _available() -> float:
    """The machine's available memory in GB, every other program's use counted."""
    import psutil
    return psutil.virtual_memory().available / GB


def _shortfall(jobs: list[dict], job: dict) -> str:
    """Why a job cannot hold its reservation now, or nothing when it can: an earlier job still waiting, the budget's
    free remainder, or the machine's available memory beside what the holding jobs have reserved and not yet used."""
    index = next(index for index, other in enumerate(jobs) if other["pid"] == job["pid"])
    others = [other for other in jobs if other["pid"] != job["pid"] and other["admitted"]]
    if not job["admitted"] and any(not other["admitted"] for other in jobs[:index]):
        return "an earlier job is waiting"
    cores = sum(other["cores"] for other in others)
    if cores + job["cores"] > CORES:
        return f"{cores} of {CORES} cores reserved"
    memory = sum(other["memory"] for other in others)
    if memory + job["memory"] > budget_memory():
        return f"{memory:g} of {budget_memory():.0f} GB reserved"
    unused = sum(max(0.0, other["memory"] - resident(other["pid"])) for other in others)
    wanted = max(0.0, job["memory"] - resident(job["pid"])) + MARGIN
    available = _available() - unused
    if available < wanted:
        return f"{available:.1f} GB available beyond the reservations, {wanted:.1f} GB wanted"
    return ""


def room(cores: int, memory: float) -> int:
    """How many jobs of `cores` cores and `memory` GB the pool would admit now, beside every job holding or waiting."""
    jobs = read()
    by_cores = (CORES - sum(job["cores"] for job in jobs)) // cores
    by_budget = (budget_memory() - sum(job["memory"] for job in jobs)) // memory
    unused = sum(max(0.0, job["memory"] - resident(job["pid"])) for job in jobs)
    by_machine = (_available() - unused - MARGIN) // memory
    return max(0, int(min(by_cores, by_budget, by_machine)))


# --- holding a reservation

def hold(cores: int, memory: float, what: str, run: str, report: bool) -> None:
    """Reserve `cores` cores and `memory` GB for this process and everything it starts, waiting until the pool admits
    them, and hold them until the process ends, under a watchdog; with `report`, the watchdog prints the peak at the
    end. From then on the process's numerical threads are its cores, set here for whatever it imports later and every
    process it starts, and it runs at low priority. `run` names the run the job belongs to, such as the test run of
    a worker. Outside the pool, or inside a job, nothing happens."""
    if not active():
        return
    if not (1 <= cores <= CORES and 0 < memory <= MAX_MEMORY):
        raise ValueError(f"a job reserves 1 to {CORES} cores and up to {MAX_MEMORY} GB, not {cores} and {memory:g}")
    import psutil
    _share_legacy_lock(what)
    me = psutil.Process()
    job = {"pid": me.pid, "started": me.create_time(), "cores": cores, "memory": memory, "what": what, "run": run,
           "since": time.time(), "admitted": False}
    atexit.register(release)
    with _ledger() as jobs:
        jobs.append(job)
    _held.update(job)
    waited = _wait(lambda found: found, lambda found: found.update(admitted=True), f"{cores} cores and {memory:g} GB")
    if waited:
        print(f"pool: {what} admitted after {waited:.0f} s", file=sys.stderr, flush=True)
    # The watchdog starts before this process lowers its priority, and so keeps its own.
    _watchdog.append(subprocess.Popen([sys.executable, "-I", __file__, "watch", str(me.pid), str(memory), what,
                                       *(["--report"] if report else [])], stdin=subprocess.PIPE))
    os.environ.update(threads(cores))
    os.environ[JOB] = str(me.pid)
    os.nice(NICENESS)


def _wait(candidate, admit, wanted: str) -> float:
    """Wait until the pool admits what `candidate` makes of this process's job, then let `admit` change the job's
    record; the seconds waited, or none where it was admitted at once."""
    start = time.monotonic()
    announced = False
    while True:
        with _ledger() as jobs:
            found = next(job for job in jobs if job["pid"] == _held["pid"])
            reason = _shortfall(jobs, candidate(found))
            if not reason:
                admit(found)
                return time.monotonic() - start if announced else 0.0
        if not announced:
            print(f"pool: {_held['what']} waits for {wanted}: {reason}", file=sys.stderr, flush=True)
            announced = True
        time.sleep(WAIT_SECONDS)


def resize(memory: float) -> None:
    """Change the memory this process's job holds to `memory` GB: more waits until the pool admits it, less is
    immediate. The watchdog holds the job to the larger of the two while they change."""
    if not _held:
        return
    if not 0 < memory <= MAX_MEMORY:
        raise ValueError(f"a job reserves up to {MAX_MEMORY} GB, not {memory:g}")
    if memory > _held["memory"]:
        _wait(lambda found: {**found, "memory": memory}, lambda found: found.update(memory=memory), f"{memory:g} GB")
        _tell("memory", f"{memory:g}")
    else:
        _tell("memory", f"{memory:g}")
        with _ledger() as jobs:
            next(job for job in jobs if job["pid"] == _held["pid"]).update(memory=memory)
    _held["memory"] = memory


def doing(text: str) -> None:
    """Tell this process's watchdog what the job is doing now, to name it should the watchdog kill the job."""
    _tell("doing", text)


def _tell(key: str, value: str) -> None:
    for watchdog in _watchdog:
        watchdog.stdin.write(f"{key} {value}\n".encode())
        watchdog.stdin.flush()


def release() -> None:
    """Give back this process's reservation, and stop its watchdog."""
    if not _held:
        return
    with _ledger() as jobs:
        jobs[:] = [job for job in jobs if job["pid"] != _held["pid"]]
    for watchdog in _watchdog:
        watchdog.stdin.close()
    _held.clear()
    _watchdog.clear()


def _share_legacy_lock(what: str) -> None:
    """Hold the lock of runs that predate the pool, shared, until this process ends: wait while such a run holds it."""
    path = os.environ.get(LEGACY_LOCK)
    if not path or os.environ.get(LEGACY_LOCK_HELD) == "1" or Path(path).is_dir():
        return
    lock = open(path, "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        print(f"pool: {what} waits for a run outside the pool: {lock.read().strip()}", file=sys.stderr, flush=True)
        fcntl.flock(lock, fcntl.LOCK_SH)
    _legacy_locks.append(lock)


def run(command: list[str], cores: int, memory: float, what: str) -> int:
    """Run a command as one job of the pool, and return its exit status; outside the pool, or inside a job, the
    command just runs."""
    hold(cores, memory, what, what, report=True)
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        return process.wait()


# --- the watchdog

def watch(pid: int, memory: float, what: str, report: bool) -> None:
    """Hold the job of process `pid` to `memory` GB: kill its whole process tree the moment its resident memory
    exceeds that. The job tells its watchdog, a line at a time on its input, a new reservation (`memory 4`) or what
    it is doing (`doing <text>`); the input closing, or the job's process ending, ends the watch."""
    import psutil
    try:
        job = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    limit, activity, peak = memory, "", 0.0
    tree, refreshed = [job], 0.0
    pending = b""
    while True:
        readable, _, _ = select.select([sys.stdin.fileno()], [], [], WATCH_SECONDS)
        if readable:
            data = os.read(sys.stdin.fileno(), 1 << 16)
            if not data:
                break
            *lines, pending = (pending + data).split(b"\n")
            for line in lines:
                key, _, value = line.decode().partition(" ")
                if key == "memory":
                    limit = float(value)
                else:
                    activity = value
        now = time.monotonic()
        try:
            if now - refreshed > TREE_SECONDS:
                tree = [job, *(member for member in job.children(recursive=True) if member.pid != os.getpid())]
                refreshed = now
            if not job.is_running():
                break
        except psutil.NoSuchProcess:
            break
        total = 0
        for member in tree:
            with contextlib.suppress(psutil.NoSuchProcess):
                total += member.memory_info().rss
        peak = max(peak, total / GB)
        if total > limit * GB:
            # The tree is gathered again, so that a process started since the last look is killed too.
            with contextlib.suppress(psutil.NoSuchProcess):
                tree = [job, *(member for member in job.children(recursive=True) if member.pid != os.getpid())]
            for member in reversed(tree):
                with contextlib.suppress(psutil.NoSuchProcess):
                    member.kill()
            during = f" during {activity}" if activity else ""
            print(f"\npool: killed {what}{during} at {total / GB:.2f} GB, over its reservation of {limit:g} GB",
                  file=sys.stderr, flush=True)
            sys.exit(1)
    if report:
        print(f"pool: {what} peaked at {peak:.2f} GB of its {limit:g} GB", file=sys.stderr, flush=True)


# --- looking at the pool

def status() -> None:
    """Print the budget, and the jobs holding and waiting with their resident memory."""
    jobs = read()
    held = [job for job in jobs if job["admitted"]]
    print(f"{sum(job['cores'] for job in held)} of {CORES} cores and {sum(job['memory'] for job in held):g} of "
          f"{budget_memory():.0f} GB reserved; {_available():.1f} GB available on the machine")
    for job in jobs:
        state = "holds" if job["admitted"] else "waits"
        print(f"{state} {job['cores']} cores {job['memory']:>4g} GB, resident {resident(job['pid']):5.2f} GB, "
              f"{time.time() - job['since']:5.0f} s: {job['what']}")


def trace(path: Path, seconds: float, until_idle: bool) -> None:
    """Append the pool's state to a file every TRACE_SECONDS, a line of JSON each: the time, the machine's available
    memory, and each job with its resident memory; for `seconds`, or with `until_idle` until the pool has held jobs
    and then been empty for IDLE_SECONDS."""
    start = time.time()
    seen, idle_since = False, None
    with open(path, "a") as file:
        while time.time() - start < seconds:
            jobs = [{**job, "resident": resident(job["pid"])} for job in read()]
            file.write(json.dumps({"time": time.time(), "available": _available(), "jobs": jobs}) + "\n")
            file.flush()
            seen |= bool(jobs)
            idle_since = None if jobs else (idle_since or time.time())
            if until_idle and seen and idle_since and time.time() - idle_since > IDLE_SECONDS:
                return
            time.sleep(TRACE_SECONDS)


def chart(path: Path, name: str) -> None:
    """Draw a trace into the shared plots, from a little before its first job to a little after its last: the cores
    and the memory each run reserved, stacked, against the budget, with the memory its jobs held and the memory the
    machine had available; and each job's own memory against its reservation."""
    import matplotlib
    matplotlib.use("Agg")
    from examples.util import charts
    from examples.util.animation import save_figure

    samples = [json.loads(line) for line in path.read_text().splitlines()]
    busy = [index for index, sample in enumerate(samples) if sample["jobs"]]
    samples = samples[max(0, busy[0] - CHART_MARGIN):busy[-1] + CHART_MARGIN + 1]
    start = samples[0]["time"]
    minutes = [(sample["time"] - start) / 60 for sample in samples]
    runs = list(dict.fromkeys(job["run"] for sample in samples for job in sample["jobs"]))
    jobs = list(dict.fromkeys((job["pid"], job["run"]) for sample in samples for job in sample["jobs"]))
    colours = dict(zip(runs, matplotlib.colormaps["tab10"].colors))
    budget = budget_memory()

    def total(run: str, key: str) -> list[float]:
        return [sum(job[key] for job in sample["jobs"] if job["run"] == run and job["admitted"]) for sample in samples]

    def own(pid: int, key: str) -> list[float]:
        return [next((job[key] for job in sample["jobs"] if job["pid"] == pid and job["admitted"]), float("nan"))
                for sample in samples]

    with charts.style():
        figure, (cores_axes, memory_axes, jobs_axes) = charts.subplots(560, 600, nrows=3, sharex=True)
        cores_axes.stackplot(minutes, *(total(run, "cores") for run in runs), colors=colours.values(), step="post",
                             labels=runs)
        cores_axes.axhline(CORES, color="black", linestyle="--")
        cores_axes.set_ylabel("cores reserved")
        cores_axes.set_ylim(0, CORES * 1.6)
        cores_axes.set_yticks([0, CORES])
        cores_axes.legend(loc="upper left", frameon=False)
        memory_axes.stackplot(minutes, *(total(run, "memory") for run in runs), colors=colours.values(), alpha=0.35,
                              step="post")
        memory_axes.stackplot(minutes, *(total(run, "resident") for run in runs), colors=colours.values(),
                              step="post")
        memory_axes.plot(minutes, [sample["available"] for sample in samples], color="grey",
                         label="available on the machine")
        memory_axes.axhline(budget, color="black", linestyle="--")
        memory_axes.set_ylabel("GB reserved, held")
        memory_axes.set_ylim(0, budget * 1.3)
        memory_axes.set_yticks([0, round(budget)])
        memory_axes.legend(loc="upper left", frameon=False)
        for pid, run in jobs:
            jobs_axes.plot(minutes, own(pid, "memory"), color=colours[run], linestyle=":", drawstyle="steps-post")
            jobs_axes.plot(minutes, own(pid, "resident"), color=colours[run])
        jobs_axes.set_ylabel("GB of each job")
        jobs_axes.set_ylim(0, None)
        jobs_axes.set_xlabel("minutes")
        jobs_axes.set_xlim(0, minutes[-1])
        for axes in (cores_axes, memory_axes, jobs_axes):
            axes.spines[["top", "right"]].set_visible(False)
        save_figure(figure, name)


def main(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    cores, memory = requested(1, 4)
    running = commands.add_parser("run", help="run a command as one job of the pool")
    running.add_argument("--cores", type=int, default=cores, help=f"cores the job reserves (default {cores})")
    running.add_argument("--memory", type=float, default=memory, metavar="GB",
                         help=f"memory the job reserves, up to {MAX_MEMORY} GB (default {memory:g})")
    running.add_argument("--what", help="the job's name in the pool (default: the command)")
    running.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("status", help="print the jobs holding and waiting")
    tracing = commands.add_parser("trace", help="sample the pool to a file twice a second")
    tracing.add_argument("path", type=Path)
    tracing.add_argument("--seconds", type=float, default=float("inf"))
    tracing.add_argument("--until-idle", action="store_true", help="stop once the pool has held jobs and is empty")
    charting = commands.add_parser("chart", help="draw a trace into the shared plots")
    charting.add_argument("path", type=Path)
    charting.add_argument("--name", default="pool")
    watching = commands.add_parser("watch")
    watching.add_argument("pid", type=int)
    watching.add_argument("memory", type=float)
    watching.add_argument("what")
    watching.add_argument("--report", action="store_true")
    options = parser.parse_args(arguments)
    if options.action == "run":
        command = options.command[1:] if options.command[:1] == ["--"] else options.command
        if not command:
            parser.error("a command must follow --")
        # A command is named by its first words, on one line.
        named = " ".join(" ".join(command).split())
        sys.exit(run(command, options.cores, options.memory, options.what or named[:NAME_LENGTH]))
    elif options.action == "status":
        status()
    elif options.action == "trace":
        trace(options.path, options.seconds, options.until_idle)
    elif options.action == "chart":
        chart(options.path, options.name)
    else:
        watch(options.pid, options.memory, options.what, options.report)


if __name__ == "__main__":
    main(sys.argv[1:])
