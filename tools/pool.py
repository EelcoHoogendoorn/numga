"""A pool of cores and memory that every heavy run on the machine shares: each test file, each notebook, each module of
the examples run as a program, and any command run through it, is one job of the pool.

The pool's budget is three quarters of the machine's cores and two thirds of its memory, so that the machine stays
usable beside it. A job reserves cores and memory before it starts, and waits while an earlier job is still waiting
or while the budget, less what the other jobs have reserved, does not fit it; what the jobs actually use plays no
part. A watchdog holds each job to its reservation: it polls the memory of the job's whole process tree, and kills
the tree the moment it exceeds what the job reserved, naming the job and the amount. A process's memory is what it
costs the machine: on macOS its physical footprint, as Activity Monitor shows it, which counts the pages the system
has compressed or swapped out, where the resident set does not; elsewhere its resident set. A job's numerical threads
equal its cores, and it runs at low priority.

A job the watchdog kills has not failed: it needed more memory than it reserved. It runs again with twice the memory,
and again. A job that asks for more than the whole budget, as it starts or by doubling, fails at once, and never
waits; only a job that fits the budget but not its free remainder waits. A job run under a key
learns the memory it finally needed, and starts with it the next time.

The pool lives in a directory every checkout on the machine shares, `~/.cache/numga/pool` unless NUMGA_POOL names
another: its reservations, written under a lock, its record of kills and the memory its jobs have learned. A
reservation whose process no longer exists is dropped whenever the file is read, so a crash never leaks one. A process
inside a job, the kernel of a notebook or a subprocess of a test, belongs to that job and reserves nothing of its own.
Without psutil, nothing is reserved and nothing waits.

The module uses only the standard library and psutil, imported where the pool is used, and runs as a script as well
as a module of the examples:

    python -m tools.pool run [--cores N] [--memory GB] [--learn KEY] -- <command>   # any command, as one job
    python -m tools.pool status                                                      # the jobs holding and waiting
    python -m tools.pool trace <file> [--seconds S | --until-idle]                   # the pool, twice a second
    python -m tools.pool chart <file>                                                # a trace, drawn

A job declares more than its default with `--cores` and `--memory`, or with NUMGA_POOL_CORES and NUMGA_POOL_MEMORY in
its environment; NUMGA_POOL_BUDGET_CORES and NUMGA_POOL_BUDGET_MEMORY set the budget.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import fcntl
import importlib.util
import json
import os
import select
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

# The pool's directory; set inside a job, to the process that holds its reservation; the run a batch's jobs belong to;
# a job's own cores and memory.
POOL, JOB, GROUP = "NUMGA_POOL", "NUMGA_POOL_JOB", "NUMGA_POOL_GROUP"
CORES_REQUESTED, MEMORY_REQUESTED = "NUMGA_POOL_CORES", "NUMGA_POOL_MEMORY"
CORES_BUDGETED, MEMORY_BUDGETED = "NUMGA_POOL_BUDGET_CORES", "NUMGA_POOL_BUDGET_MEMORY"
DIRECTORY = Path.home() / ".cache" / "numga" / "pool"

GB = 2 ** 30
# The budget: three quarters of the machine's cores; its memory, two thirds of the machine's, is `budget_memory()`,
# and is also the most one job reserves.
CORES = int(os.environ.get(CORES_BUDGETED, max(1, 3 * (os.cpu_count() or 1) // 4)))
# How often a waiting job asks again, how often a watchdog reads its job's memory and, less often, its tree, and how
# often a batch looks whether its last job has been admitted.
WAIT_SECONDS, WATCH_SECONDS, TREE_SECONDS, BATCH_SECONDS = 0.5, 0.1, 1.0, 0.1
# How often a trace samples the pool, how long an empty pool ends a trace that waits for it, and how many samples a
# chart of it keeps before the first job and after the last.
TRACE_SECONDS, IDLE_SECONDS, CHART_MARGIN = 0.5, 10, 4
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS")
NICENESS = 10
# The 64-bit words of proc_pid_rusage's record that are read, and the one holding the physical footprint.
RUSAGE_WORDS, FOOTPRINT_WORD = 16, 9
# The characters of a command that name its job.
NAME_LENGTH = 100
# The exit status of a job that asks for more than the whole budget.
TOO_LARGE = 1

# This process's reservation, once it holds one, its watchdog, and the library that reads macOS's footprints.
_held: dict = {}
_watchdog: list[subprocess.Popen] = []
_libraries: list = []


def budget_memory() -> float:
    """The budget's memory in GB, two thirds of the machine's, which is also the most one job reserves."""
    import psutil
    return float(os.environ.get(MEMORY_BUDGETED, 2 * psutil.virtual_memory().total / 3 / GB))


def active() -> bool:
    """Whether a job started here takes a reservation of its own: psutil is there to watch it, and this is not inside
    a job."""
    return JOB not in os.environ and importlib.util.find_spec("psutil") is not None


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
    directory = Path(os.environ.get(POOL) or DIRECTORY)
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


def _cost(process) -> int:
    """The memory a process costs the machine, in bytes."""
    if sys.platform != "darwin":
        return process.memory_info().rss
    import ctypes
    # proc_pid_rusage's first record, rusage_info_v0: a 16-byte uuid, then 64-bit counts, the physical footprint the
    # eighth of them.
    usage = (ctypes.c_uint64 * RUSAGE_WORDS)()
    if _libproc().proc_pid_rusage(process.pid, 0, usage):
        import psutil
        raise psutil.NoSuchProcess(process.pid)
    return usage[FOOTPRINT_WORD]


def _libproc():
    import ctypes
    if not _libraries:
        _libraries.append(ctypes.CDLL("/usr/lib/libproc.dylib"))
    return _libraries[0]


def footprint(pid: int) -> float:
    """The memory in GB a process and every process below it cost the machine."""
    import psutil
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for member in tree:
        with contextlib.suppress(psutil.NoSuchProcess):
            total += _cost(member)
    return total / GB


def _available() -> float:
    """The machine's available memory in GB, every other program's use counted."""
    import psutil
    return psutil.virtual_memory().available / GB


def _shortfall(jobs: list[dict], job: dict) -> str:
    """Why a job cannot hold its reservation now, or nothing when it can: an earlier job still waiting, or the budget,
    less what the other jobs hold, too small for it."""
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
    return ""


# --- holding a reservation

def hold(cores: int, memory: float, what: str) -> None:
    """Reserve `cores` cores and `memory` GB for this process and everything it starts, waiting until the pool admits
    them, and hold them until the process ends, under a watchdog that kills the processes below this one should they
    exceed them. From then on the process's numerical threads are its cores, set here for every process it starts,
    and it runs at low priority."""
    if not (1 <= cores <= CORES and 0 < memory <= budget_memory()):
        raise ValueError(f"a job reserves 1 to {CORES} cores and up to {budget_memory():.0f} GB, "
                         f"not {cores} and {memory:g}")
    import psutil
    me = psutil.Process()
    job = {"pid": me.pid, "started": me.create_time(), "cores": cores, "memory": memory, "what": what,
           "group": os.environ.get(GROUP, what), "since": time.time(), "admitted": False}
    atexit.register(release)
    with _ledger() as jobs:
        jobs.append(job)
    _held.update(job)
    waited = _wait(lambda found: found, lambda found: found.update(admitted=True), f"{cores} cores and {memory:g} GB")
    if waited:
        print(f"pool: {what} admitted after {waited:.0f} s", file=sys.stderr, flush=True)
    # The watchdog starts before this process lowers its priority, and so keeps its own.
    _watchdog.append(subprocess.Popen([sys.executable, "-I", __file__, "watch", str(me.pid), str(memory), what],
                                      stdin=subprocess.PIPE))
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
    if memory > _held["memory"]:
        _wait(lambda found: {**found, "memory": memory}, lambda found: found.update(memory=memory), f"{memory:g} GB")
        _tell("memory", f"{memory:g}")
    else:
        _tell("memory", f"{memory:g}")
        with _ledger() as jobs:
            next(job for job in jobs if job["pid"] == _held["pid"]).update(memory=memory)
    _held["memory"] = memory


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


def run(command: list[str], cores: int, memory: float, what: str, key: str) -> int:
    """Run a command as one job of the pool, and return its exit status. Where the watchdog kills the command for
    exceeding its reservation, it runs again with twice the memory. With a `key`, the job starts with the memory it
    learned under that key, where that is more, and learns the memory it finally ran with. A job that asks for more
    than the whole budget, at the start or by doubling, fails at once, and never waits. Outside the pool, or inside a
    job, the command just runs."""
    if not active():
        return subprocess.run(command).returncode
    start = max(memory, learned().get(key, 0.0)) if key else memory
    if cores > CORES or start > budget_memory():
        print(f"pool: {what} asks for {cores} cores and {start:g} GB, more than the pool's whole budget of {CORES} "
              f"cores and {budget_memory():.0f} GB", file=sys.stderr, flush=True)
        return TOO_LARGE
    hold(cores, start, what)
    while True:
        started = time.time()
        process = subprocess.Popen(command)
        try:
            status = process.wait()
        except KeyboardInterrupt:
            status = process.wait()
        if not [kill for kill in kills() if kill["pid"] == _held["pid"] and kill["time"] >= started]:
            if key and _held["memory"] > start:
                learn(key, _held["memory"])
            return status
        if 2 * _held["memory"] > budget_memory():
            print(f"pool: {what} needs more than its {_held['memory']:g} GB, and twice that is more than the pool's "
                  f"whole budget of {budget_memory():.0f} GB", file=sys.stderr, flush=True)
            return TOO_LARGE
        resize(2 * _held["memory"])
        print(f"pool: {what} runs again with {_held['memory']:g} GB", file=sys.stderr, flush=True)


def kills() -> list[dict]:
    """The jobs the watchdogs killed, in the order they were killed: each job's pid, its name, the memory it reached
    and the memory it reserved, and when."""
    path = _directory() / "kills.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def learned() -> dict[str, float]:
    """The memory in GB each key has learned its job needs."""
    path = _directory() / "memory.json"
    return json.loads(path.read_text()) if path.exists() else {}


def learn(key: str, memory: float) -> None:
    """Record the memory in GB a key's job needs, beside what other jobs have recorded."""
    with _ledger():
        amounts = {**learned(), key: memory}
        written = _directory() / f"memory.{os.getpid()}.json"
        lines = [f"{json.dumps(name)}: {value:g}" for name, value in sorted(amounts.items())]
        written.write_text("{\n" + ",\n".join(lines) + "\n}\n")
        os.replace(written, _directory() / "memory.json")


def batch(jobs: list[dict], group: str) -> Iterator[tuple[dict, int, str]]:
    """Run commands as jobs of the pool, as many at once as it admits, and yield each job with its exit status and
    output as it finishes. A job is a dict of its `command`, the folder `cwd` it runs in, its `cores` and `memory`,
    its name `what` and its `key`; `group` names the run they make together. A new job starts once the last one
    started has been admitted, so that the batch waits in the pool's queue with one job at a time; inside a job, or
    without psutil, the jobs run one by one."""
    pending, running, outputs = list(jobs), {}, {}
    while pending or running:
        admitted = {job["pid"] for job in read() if job["admitted"]} if active() else set()
        last = list(running)[-1:]
        if pending and all(process.pid in admitted for process in last) and (active() or not running):
            job = pending.pop(0)
            command = [sys.executable, "-I", __file__, "run", "--cores", str(job["cores"]), "--memory",
                       f"{job['memory']:g}", "--what", job["what"], "--learn", job["key"], "--", *job["command"]]
            process = subprocess.Popen(command, cwd=job["cwd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, env=os.environ | {GROUP: group})
            running[process] = job
            # Each output is read as it comes, so that no job stalls on a full pipe.
            threading.Thread(target=lambda process=process: outputs.update({process: process.communicate()[0]}),
                             daemon=True).start()
        for process in [process for process in running if process in outputs]:
            yield running.pop(process), process.returncode, outputs.pop(process)
        time.sleep(BATCH_SECONDS)


# --- the watchdog

def watch(pid: int, memory: float, what: str) -> None:
    """Hold the job of process `pid` to `memory` GB: the moment the memory of its whole process tree exceeds that,
    write the kill to the pool's record of kills, and kill every process below the job's own, which may then run its
    command again. The job tells its watchdog a new reservation, a line at a time on its input (`memory 4`); the
    input closing, or the job's process ending, ends the watch."""
    import psutil
    try:
        job = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    limit, peak = memory, 0.0
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
                limit = float(line.decode().partition(" ")[2])
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
                total += _cost(member)
        peak = max(peak, total / GB)
        if total > limit * GB:
            record = {"pid": pid, "what": what, "footprint": total / GB, "reserved": limit, "time": time.time()}
            with open(_directory() / "kills.jsonl", "a") as kills_file:
                kills_file.write(json.dumps(record) + "\n")
            # The tree is gathered again, so that a process started since the last look is killed too.
            with contextlib.suppress(psutil.NoSuchProcess):
                tree = [job, *(member for member in job.children(recursive=True) if member.pid != os.getpid())]
            for member in reversed(tree[1:]):
                with contextlib.suppress(psutil.NoSuchProcess):
                    member.kill()
            print(f"\npool: killed {what} at {total / GB:.2f} GB, over its reservation of {limit:g} GB",
                  file=sys.stderr, flush=True)
            tree = [job]
    print(f"pool: {what} peaked at {peak:.2f} GB of its {limit:g} GB", file=sys.stderr, flush=True)


# --- looking at the pool

def status() -> None:
    """Print the budget, and the jobs holding and waiting with the memory they cost."""
    jobs = read()
    held = [job for job in jobs if job["admitted"]]
    print(f"{sum(job['cores'] for job in held)} of {CORES} cores and {sum(job['memory'] for job in held):g} of "
          f"{budget_memory():.0f} GB reserved; {_available():.1f} GB available on the machine")
    for job in jobs:
        state = "holds" if job["admitted"] else "waits"
        print(f"{state} {job['cores']} cores {job['memory']:>4g} GB, using {footprint(job['pid']):5.2f} GB, "
              f"{time.time() - job['since']:5.0f} s: {job['what']}")


def trace(path: Path, seconds: float, until_idle: bool) -> None:
    """Append the pool's state to a file every TRACE_SECONDS, a line of JSON each: the time, the machine's available
    memory, and each job with the memory it costs; for `seconds`, or with `until_idle` until the pool has held jobs
    and then been empty for IDLE_SECONDS."""
    start = time.time()
    seen, idle_since = False, None
    with open(path, "a") as file:
        while time.time() - start < seconds:
            jobs = [{**job, "footprint": footprint(job["pid"])} for job in read()]
            file.write(json.dumps({"time": time.time(), "available": _available(), "jobs": jobs}) + "\n")
            file.flush()
            seen |= bool(jobs)
            idle_since = None if jobs else (idle_since or time.time())
            if until_idle and seen and idle_since and time.time() - idle_since > IDLE_SECONDS:
                return
            time.sleep(TRACE_SECONDS)


def chart(path: Path, name: str) -> None:
    """Draw a trace into the examples' plots, from a little before its first job to a little after its last: the
    cores and the memory each run reserved, stacked, against the budget, with the memory its jobs held and the memory
    the machine had available; and each job's own memory against its reservation."""
    import matplotlib
    matplotlib.use("Agg")
    from examples.util import charts
    from examples.util.animation import save_figure

    samples = [json.loads(line) for line in path.read_text().splitlines()]
    busy = [index for index, sample in enumerate(samples) if sample["jobs"]]
    samples = samples[max(0, busy[0] - CHART_MARGIN):busy[-1] + CHART_MARGIN + 1]
    start = samples[0]["time"]
    minutes = [(sample["time"] - start) / 60 for sample in samples]
    runs = list(dict.fromkeys(job["group"] for sample in samples for job in sample["jobs"]))
    jobs = list(dict.fromkeys((job["pid"], job["group"]) for sample in samples for job in sample["jobs"]))
    colours = dict(zip(runs, matplotlib.colormaps["tab10"].colors))

    def total(run: str, key: str) -> list[float]:
        return [sum(job[key] for job in sample["jobs"] if job["group"] == run and job["admitted"])
                for sample in samples]

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
        memory_axes.stackplot(minutes, *(total(run, "footprint") for run in runs), colors=colours.values(),
                              step="post")
        memory_axes.plot(minutes, [sample["available"] for sample in samples], color="grey",
                         label="available on the machine")
        memory_axes.axhline(budget_memory(), color="black", linestyle="--")
        memory_axes.set_ylabel("GB reserved, held")
        memory_axes.set_ylim(0, budget_memory() * 1.3)
        memory_axes.set_yticks([0, round(budget_memory())])
        memory_axes.legend(loc="upper left", frameon=False)
        for pid, run in jobs:
            jobs_axes.plot(minutes, own(pid, "memory"), color=colours[run], linestyle=":", drawstyle="steps-post")
            jobs_axes.plot(minutes, own(pid, "footprint"), color=colours[run])
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
                         help=f"memory the job reserves, up to the pool's budget (default {memory:g})")
    running.add_argument("--what", help="the job's name in the pool (default: the command)")
    running.add_argument("--learn", default="", metavar="KEY",
                         help="start with the memory learned under this key, and learn the memory finally needed")
    running.add_argument("command", nargs=argparse.REMAINDER)
    commands.add_parser("status", help="print the jobs holding and waiting")
    tracing = commands.add_parser("trace", help="sample the pool to a file twice a second")
    tracing.add_argument("path", type=Path)
    tracing.add_argument("--seconds", type=float, default=float("inf"))
    tracing.add_argument("--until-idle", action="store_true", help="stop once the pool has held jobs and is empty")
    charting = commands.add_parser("chart", help="draw a trace into the examples' plots")
    charting.add_argument("path", type=Path)
    charting.add_argument("--name", default="pool")
    watching = commands.add_parser("watch")
    watching.add_argument("pid", type=int)
    watching.add_argument("memory", type=float)
    watching.add_argument("what")
    options = parser.parse_args(arguments)
    if options.action == "run":
        command = options.command[1:] if options.command[:1] == ["--"] else options.command
        if not command:
            parser.error("a command must follow --")
        # A command is named by its first words, on one line.
        named = " ".join(" ".join(command).split())
        sys.exit(run(command, options.cores, options.memory, options.what or named[:NAME_LENGTH], options.learn))
    elif options.action == "status":
        status()
    elif options.action == "trace":
        trace(options.path, options.seconds, options.until_idle)
    elif options.action == "chart":
        chart(options.path, options.name)
    else:
        watch(options.pid, options.memory, options.what)


if __name__ == "__main__":
    main(sys.argv[1:])
