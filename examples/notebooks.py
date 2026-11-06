"""Run the example notebooks, all of them or a selection, each from its own folder and one at a time, and record how
each went. A notebook takes as long as it takes, often a minute or more, so a run of them all is never a gate before a
commit or a push: it is done every once in a while, and a selection after work on what those notebooks use.

    python -m examples.notebooks                                         # every notebook
    python -m examples.notebooks wing fluids/vortices                    # those whose paths hold any of the strings
    python -m examples.notebooks --touching examples.util.topology numga.sparse
    python -m examples.notebooks --changed main                          # their folder or a module they use changed
    python -m examples.notebooks --list --touching jax                   # the selection, without running it

A notebook uses the modules its cells import, and those that the examples' own modules among them import in turn, read
from the source without running any. The imports of the shared modules in `examples/util` are not followed, but for
those of their packages' `__init__`, which runs whenever anything inside is imported, and neither are numga's, since
importing numga loads all of it: following them would make nearly every notebook use nearly every module. A string
naming a module of the checkout counts as importing it, as `instantiate` takes one. A module touched selects the
notebooks that use it or a module inside it.

Each notebook runs as `jupyter execute`, which writes nothing into the notebook, as one job of the machine's pool
(`examples/pool.py`), of one core and 4 GB unless `--cores` or `--memory` ask for more, where the machine has one. Its
outcome, ran, failed or timed out, its seconds and, for a failure, the cell it failed in and the last lines of the
error, goes into `archive/notebook_runs.json` of the workspace that holds the checkout, as it comes, under the
checkout, its commit and the time the run started, and a summary is printed at the end, or at Ctrl-C.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime
import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
# The shared modules of the examples, whose imports a notebook's uses do not follow.
SHARED = ROOT / "examples" / "util"
# The machine's pool, run as a script, which needs nothing of the examples; the archive of the workspace a checkout
# sits in.
POOL = Path(__file__).resolve().parent / "pool.py"
LOG = ROOT.parent / "archive" / "notebook_runs.json"
# The lines of a failure's output its record keeps.
ERROR_LINES = 12
# Where the error nbclient reports quotes the source of the cell it failed in.
FAILED_CELL = re.compile(r"An error occurred while executing the following cell:\n-{18}\n(.*?)\n-{18}\n", re.S)
COLOURS = re.compile(r"\x1b\[[0-9;]*m")


def notebooks() -> list[Path]:
    """Every notebook of the examples, by its path in the checkout."""
    return sorted(path.relative_to(ROOT) for path in (ROOT / "examples").rglob("*.ipynb")
                  if ".ipynb_checkpoints" not in path.parts)


def source(module: str) -> Path:
    """The file a module of the checkout is read from, a package's `__init__.py`; for any other module it does not
    exist."""
    path = ROOT.joinpath(*module.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


def imported(code: str, package: str) -> set[str]:
    """The modules a piece of code imports, by their full names, its relative imports read inside `package`: those an
    `import` or a `from` names, the names a `from` takes that are modules, and the strings that name a module of the
    checkout; and the packages of each, whose `__init__` runs first."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names if source(f"{module}.{alias.name}").exists())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and all(part.isidentifier() for part in node.value.split(".")) and source(node.value).exists():
            names.add(node.value)
    return {".".join(name.split(".")[:end]) for name in names for end in range(1, name.count(".") + 2)}


@functools.cache
def module_imports(module: str) -> frozenset[str]:
    """The modules a module of the checkout imports."""
    path = source(module)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    return frozenset(imported(path.read_text(), package))


def cell_code(cell: dict) -> str:
    """A code cell as Python, its IPython magics and shell lines read as `pass`."""
    lines = "".join(cell["source"]).splitlines()
    return "\n".join(re.sub(r"^(\s*)[%!].*", r"\1pass", line) for line in lines)


def uses(notebook: Path) -> set[str]:
    """The modules a notebook uses: those its cells import, and those that the examples' own modules among them, and
    the packages of the shared ones, import in turn."""
    cells = json.loads((ROOT / notebook).read_text())["cells"]
    found = set().union(*(imported(cell_code(cell), "") for cell in cells if cell["cell_type"] == "code"))
    pending = list(found)
    while pending:
        module = pending.pop()
        path = source(module)
        followed = module.partition(".")[0] == "examples" and (SHARED not in path.parents or path.name == "__init__.py")
        if followed and path.exists():
            new = module_imports(module) - found
            found |= new
            pending.extend(new)
    return found


def touches(modules: set[str], touched: list[str]) -> bool:
    """Whether any of the modules is one of the touched modules or lies inside one."""
    return any(module == name or module.startswith(f"{name}.") for module in modules for name in touched)


def changed(ref: str) -> list[Path]:
    """The tracked files of the checkout that differ from a git ref, committed or not."""
    differences = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", ref],
                                 capture_output=True, text=True, check=True)
    return [Path(line) for line in differences.stdout.splitlines()]


def module_of(path: Path) -> str:
    """The module a Python file of the checkout holds."""
    parts = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def stop(process: subprocess.Popen) -> None:
    """Kill a run and everything it started, the kernel too, which jupyter starts in a session of its own."""
    with contextlib.suppress(psutil.NoSuchProcess):
        root = psutil.Process(process.pid)
        # The tree is gathered before anything is killed: a killed parent's children are no longer found below it.
        for member in [*reversed(root.children(recursive=True)), root]:
            with contextlib.suppress(psutil.NoSuchProcess):
                member.kill()
    process.communicate()


def failure(notebook: Path, output: str) -> dict:
    """What a failed run's output says: the cell it failed in, counted from one as the notebook's cells stand, where
    the error names one, and the error's last lines."""
    text = COLOURS.sub("", output)
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("pool: ")]
    record = {"error": "\n".join(lines[-ERROR_LINES:])}
    quoted = FAILED_CELL.search(text)
    sources = ["".join(cell["source"]) for cell in json.loads((ROOT / notebook).read_text())["cells"]]
    if quoted and quoted[1] in sources:
        record["cell"] = sources.index(quoted[1]) + 1
    return record


def run(notebook: Path, timeout: float, cores: int, memory: float) -> dict:
    """Execute a notebook from its own folder, as one job of the pool, and say how it went: ran, failed or timed out,
    in how many seconds."""
    jupyter = [str(Path(sys.executable).parent / "jupyter"), "execute", notebook.name]
    job = [sys.executable, "-I", str(POOL), "run", "--cores", str(cores), "--memory", str(memory),
           "--what", f"notebook {notebook} in {ROOT.name}", "--"]
    # No window for any figure.
    environment = os.environ | {"MPLBACKEND": "Agg"}
    start = time.perf_counter()
    # A session of its own, so that Ctrl-C reaches only the runner, which then stops the notebook's whole tree.
    process = subprocess.Popen([*job, *jupyter], cwd=ROOT / notebook.parent, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
        status = "ran" if process.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        stop(process)
        output, status = "", "timed out"
    except KeyboardInterrupt:
        stop(process)
        raise
    record = {"path": str(notebook), "status": status, "seconds": round(time.perf_counter() - start, 1)}
    return (record | failure(notebook, output)) if status == "failed" else record


def write_log(key: str, records: list[dict]) -> None:
    """Put a run's records into the workspace's log of notebook runs, under the run's key."""
    runs = json.loads(LOG.read_text()) if LOG.exists() else {}
    runs[key] = records
    LOG.write_text(json.dumps(runs, indent=1) + "\n")


def summary(records: list[dict]) -> None:
    """A table of the notebooks run, then each failure's error, then the counts."""
    print()
    for record in records:
        print(f"{record['status']:<10}{record['seconds']:>7.0f} s  {record['path']}")
    for record in records:
        if record["status"] == "failed":
            where = f", cell {record['cell']}" if "cell" in record else ""
            print(f"\n{record['path']}{where}\n    " + record["error"].replace("\n", "\n    "))
    counts = [f"{sum(record['status'] == status for record in records)} {status}"
              for status in ("ran", "failed", "timed out")]
    print(f"\n{', '.join(counts)} in {sum(record['seconds'] for record in records):.0f} s")


def main(arguments: list[str]) -> None:
    """Select the notebooks the arguments ask for, and list them or run them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="*", help="select the notebooks whose paths hold any of these strings")
    parser.add_argument("--touching", nargs="+", default=[], metavar="MODULE",
                        help="select the notebooks that use any of these modules, such as examples.util.topology")
    parser.add_argument("--changed", metavar="REF",
                        help="select the notebooks whose folder or a module they use changed since this git ref")
    parser.add_argument("--list", action="store_true", help="print the selection without running it")
    parser.add_argument("--timeout", type=float, default=20, metavar="MINUTES",
                        help="the time each notebook may take (default 20)")
    parser.add_argument("--cores", type=int, default=1, help="the cores each notebook reserves in the pool (default 1)")
    parser.add_argument("--memory", type=float, default=4, metavar="GB",
                        help="the memory each notebook reserves in the pool, up to 8 GB (default 4)")
    options = parser.parse_args(arguments)

    everything = notebooks()
    changes = changed(options.changed) if options.changed else []
    touched = [*options.touching, *(module_of(path) for path in changes if path.suffix == ".py")]
    chosen = [notebook for notebook in everything
              if any(pattern in str(notebook) for pattern in options.patterns)
              or any(notebook.parent in path.parents for path in changes)
              or touched and touches(uses(notebook), touched)]
    selection = chosen if options.patterns or options.touching or options.changed else everything
    if options.list:
        print("\n".join(map(str, selection)))
        print(f"{len(selection)} of {len(everything)} notebooks")
        return

    commit = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty", "--exclude=*"],
                            capture_output=True, text=True, check=True).stdout.strip()
    key = f"{ROOT.name} {commit} {datetime.datetime.now().isoformat(timespec='seconds')}"
    records = []
    try:
        for index, notebook in enumerate(selection, 1):
            print(f"[{index}/{len(selection)}] {notebook}", end="", flush=True)
            records.append(run(notebook, 60 * options.timeout, options.cores, options.memory))
            print(f"  {records[-1]['status']} in {records[-1]['seconds']:.0f} s", flush=True)
            if LOG.parent.is_dir():
                write_log(key, records)
    except KeyboardInterrupt:
        print("  interrupted, and not recorded", flush=True)
    summary(records)
    # Failing when any notebook did not run, so that a run of them tests that they all do.
    sys.exit(any(record["status"] != "ran" for record in records))


if __name__ == "__main__":
    main(sys.argv[1:])
