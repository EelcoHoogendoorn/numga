"""This example's humanoid against Brax's own, the humanoid of Brax's `humanoid.xml` stepped by Brax's
reduced-coordinate backend, `brax.generalized`, both flying free in space: how many humanoid steps a second each engine
takes for batches of humanoids from one to a few thousand, how long each takes to compile, and how many lines of code
each spends on each part of the method.

Both are reduced-coordinate: kinematics, inverse dynamics, the mass matrix and its solve are of one class, and the
benchmark runs those alone. Brax's humanoid has its floor removed and its joints' limits off, so that its constraint
solver drops out of its step; this example's step receives no ground and no stops. Both start from Brax's initial pose,
the root at rest, under gravity, each hinge turning at the same small speed, matched by name, at Brax's own 3 ms step.
Brax `lax.scan`s the steps of a call and `vmap`s the batch; this example runs in numga's dense execution and in its
sparse one. Each engine's step is compiled ahead of time for each batch, and only calls to the compiled executable are
timed: each point the best of five windows of five calls, every curve alike, on an otherwise idle machine. This example's core runs in float64, and Brax runs in float64 beside it; Brax at its default float32 runs in a
second process that never imports this example. The steps still differ: Brax's hinge springs and dampers act
explicitly and its mass matrix's inverse is refined by Newton's iterations from the step before, while this example's
step carries its hinges implicitly and solves its matrix.

It needs Brax, which the `numga` environment does not hold: run it from the checkout's root with the Python of a
virtual environment that has Brax installed from `references/brax`, on one core,
`OMP_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false python -m examples.mechanics.articulated.brax_comparison`.
Nothing imports it, and no test runs it.
"""

from __future__ import annotations

import ast
import io
import json
import re
import subprocess
import sys
import time
import tokenize
from functools import partial
from pathlib import Path

import numpy as np

# The batches timed, from one humanoid to a few thousand; the step, Brax's humanoid's own, with the steps in a timed
# call, about 40 ms simulated; the calls in a timed window, and the windows, the best of which each point takes.
robot_counts = 4 ** np.arange(7)
dt, n_steps = 0.003, 13
n_timed = 5
n_windows = 5
workspace = Path(__file__).parents[4]
humanoid_file = workspace / "references" / "brax" / "brax" / "envs" / "assets" / "humanoid.xml"


# --- plumbing -------------------------------------------------------------------------
def compiled_run(advance, state, n_timed: int):
    """The seconds a jitted advance takes to trace and lower, and to compile, for its state, the seconds a call of the
    compiled executable takes, after one call, at best over several windows of calls, and the state it reaches: the
    state committed to the device first, as the executable's results are, so that the executable takes them as they
    come and refuses a state of another type rather than recompiling."""
    import jax

    state = jax.device_put(state, jax.devices()[0])
    start = time.perf_counter()
    lowered = jax.jit(advance).lower(state)
    lowering = time.perf_counter() - start
    compiled = lowered.compile()
    compiling = time.perf_counter() - start - lowering
    state = jax.block_until_ready(compiled(state))
    windows = []
    for _ in range(n_windows):
        start = time.perf_counter()
        for _ in range(n_timed):
            state = compiled(state)
        jax.block_until_ready(state)
        windows.append((time.perf_counter() - start) / n_timed)
    return lowering, compiling, min(windows), state


def brax_free_system(dt: float):
    """Brax's humanoid in free space, its floor removed and its joints' limits off, read from its own file, at a step;
    all else as the file sets it, its hinges' springs and dampers, its inverse's iterations and its integrator."""
    from brax.io import mjcf

    xml = re.sub(r'<geom[^>]*name="floor"[^>]*/>', "", humanoid_file.read_text())
    xml = xml.replace('limited="true"', 'limited="false"')
    return mjcf.loads(xml).tree_replace({"opt.timestep": dt})


def brax_flight(system, n_robots: int):
    """Brax's humanoids `[robots]` flying free from the initial pose: the root at rest, and each hinge turning at the
    speed this example's free flight gives the hinge of its name."""
    import jax
    import mujoco
    from jax import numpy as jp
    from brax.generalized import pipeline
    from examples.mechanics.articulated import scenarios

    columns = [system.mj_model.jnt_dofadr[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
               for name in scenarios.joint_names]
    speeds = jp.zeros(system.qd_size()).at[np.array(columns)].set(jp.asarray(scenarios.flight_speeds()))
    state = pipeline.init(system, system.init_q, speeds)
    # Nothing to collide and no limits: the constraints' Jacobian is empty, so the constraint solver drops out.
    assert state.con_jac.shape[0] == 0
    return jax.tree.map(lambda part: jp.broadcast_to(part, (n_robots,) + jp.shape(part)), state)


def brax_sweep(counts: np.ndarray) -> dict[str, list[float]]:
    """Brax's generalized pipeline for a batch of each count of humanoids flying free with no actuation: humanoid steps
    a second, and seconds to lower and to compile, each batch still finite at the end."""
    import jax
    from jax import numpy as jp
    from brax.generalized import pipeline

    system = brax_free_system(dt)
    sweep = {"rates": [], "lowering": [], "compiling": []}
    for n_robots in counts:
        actions = jp.zeros((n_robots, system.act_size()))

        def step(state):
            return jax.vmap(lambda one, act: pipeline.step(system, one, act))(state, actions)

        def advance(state):
            return jax.lax.scan(lambda state, _: (step(state), None), state, None, length=n_steps)[0]

        # The state at entry of the types a step gives back, so that the steps carry one type and the compiled step
        # takes its own results.
        state = brax_flight(system, n_robots)
        state = jax.tree.map(lambda part, stepped: part.astype(stepped.dtype), state, jax.eval_shape(step, state))
        lowering, compiling, call, state = compiled_run(advance, state, n_timed)
        assert np.isfinite(np.asarray(state.qd)).all()
        sweep["rates"].append(n_robots * n_steps / call)
        sweep["lowering"].append(lowering)
        sweep["compiling"].append(compiling)
    return sweep


def numga_sweep(execution: str, counts: np.ndarray) -> dict[str, list[float]]:
    """This example's step, in numga's dense or sparse execution, for a batch of each count of humanoids flying free,
    with their weights and their hinges alone: humanoid steps a second, and seconds to lower and to compile, each batch still
    finite at the end."""
    from numga import Extensor
    from numga.algebras import PGA3D
    from numga.backend.jax import JaxContext
    from examples.mechanics.articulated import core, scenarios

    context = JaxContext(PGA3D, dtype=np.float64, execution=execution)

    def moved(value):
        """A record of extensors with each moved into the execution's context."""
        return type(value)(*(part.materialize(context) if isinstance(part, Extensor) else part for part in value))

    tree, robot = scenarios.robot()
    robot = moved(robot)
    loads = partial(core.own_loads, tree, robot, moved(scenarios.physics()))
    sweep = {"rates": [], "lowering": [], "compiling": []}
    for n_robots in counts:
        lowering, compiling, call, state = compiled_run(
            lambda state: core.advance(tree, robot, state, dt, n_steps, loads),
            moved(scenarios.flight(n_robots)), n_timed)
        assert np.isfinite(np.asarray(state.speeds.kernel)).all()
        sweep["rates"].append(n_robots * n_steps / call)
        sweep["lowering"].append(lowering)
        sweep["compiling"].append(compiling)
    return sweep


def logical_lines(path: Path, names: list[str]) -> int:
    """The lines of code of a module's named top-level definitions, a class, a function, every definition of a name
    defined more than once, or `Class.method`, or of its other statements, `:module`, or of all of it, `*`: no blank
    lines, comments, docstrings or imports."""
    source = path.read_text()
    module = ast.parse(source)
    skipped = set()
    for node in ast.walk(module):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body and isinstance(body[0], ast.Expr) \
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            skipped.update(range(body[0].lineno, body[0].end_lineno + 1))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            skipped.update(range(node.lineno, node.end_lineno + 1))
    definitions = {}
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    chosen = set()
    for name in names:
        if name == "*":
            chosen.update(range(1, len(source.splitlines()) + 1))
        elif name == ":module":
            for node in module.body:
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    chosen.update(range(node.lineno, node.end_lineno + 1))
        else:
            owner, _, member = name.partition(".")
            for node in definitions[owner]:
                if member:
                    node = next(child for child in node.body if getattr(child, "name", None) == member)
                chosen.update(range(node.lineno - len(getattr(node, "decorator_list", [])), node.end_lineno + 1))
    quiet = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    coded = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in quiet:
            coded.update(range(token.start[0], token.end[0] + 1))
    return len((coded & chosen) - skipped)


def used_functions(library: Path, users: list[Path]) -> list[str]:
    """The functions of a library module, such as Brax's `math`, that the users call by the module's name, as
    `math.name`, and those they call in turn."""
    text = "\n".join(path.read_text() for path in users)
    defined = {node.name: node for node in ast.parse(library.read_text()).body if isinstance(node, ast.FunctionDef)}
    source = library.read_text().splitlines()
    wanted, used = set(re.findall(rf"\b{library.stem}\.(\w+)", text)) & set(defined), set()
    while wanted:
        name = wanted.pop()
        used.add(name)
        body = "\n".join(source[defined[name].lineno - 1:defined[name].end_lineno])
        wanted |= (set(re.findall(r"\b(\w+)\(", body)) & set(defined)) - used
    return sorted(used)


def components() -> dict[str, tuple[int, int]]:
    """The lines of code each engine spends on each part of the method the benchmark runs, this example's and Brax's,
    and on contact and joint limits, which it does not run: this example's `core.py` and the lowest point it borrows
    from the spinning top; Brax's `base.py` in full, the functions of its `math.py`, `kinematics.py` and `scan.py` the
    pipeline calls, `generalized/*`, `actuator.py` and `contact.py`, with the plane–sphere collision MuJoCo's MJX gives
    it, the only shapes Brax's humanoid lets collide. Model loading, the general libraries under both, numga, JAX and
    jaxopt, and this example's energy, which only its checks use, are left out."""
    from mujoco.mjx._src import collision_primitive

    ours = Path(__file__).parent / "core.py"
    top = workspace / "wt-articulated" / "examples" / "mechanics" / "spinning_top" / "core.py"
    brax = workspace / "references" / "brax" / "brax"
    general = brax / "generalized"
    users = [brax / name for name in ("base.py", "scan.py", "kinematics.py", "actuator.py", "contact.py")] + [
        general / name for name in ("base.py", "constraint.py", "dynamics.py", "integrator.py", "mass.py",
                                    "pipeline.py")]
    parts = {
        "spatial math": ([(ours, [":module"])],
                         [(brax / "base.py", ["Transform", "Motion", "Force", "Inertia", "_transform_do",
                                              "_transform_inv_do", "_motion_cross", "_"]),
                          (brax / "math.py", used_functions(brax / "math.py", users))]),
        "model description": ([(ours, ["Tree", "Robot", "Physics", "State"])],
                              [(brax / "base.py", [":module", "Base", "_tree_replace", "Link", "DoF", "Actuator",
                                                   "State", "System"]), (general / "base.py", ["*"])]),
        "kinematics": ([(ours, ["tree", "forward_kinematics"])],
                       [(brax / "kinematics.py", used_functions(brax / "kinematics.py", users)),
                        (brax / "scan.py", used_functions(brax / "scan.py", users)),
                        (general / "dynamics.py", ["transform_com"])]),
        "inverse dynamics": ([(ours, ["inverse_dynamics"])], [(general / "dynamics.py", ["inverse"])]),
        "mass matrix": ([(ours, ["pullback", "mass_matrix"])], [(general / "mass.py", ["matrix"])]),
        "forward dynamics, solve": ([(ours, ["solved", "forward_dynamics"])],
                                    [(general / "dynamics.py", ["forward", "_passive"]),
                                     (general / "mass.py", ["matrix_inv"])]),
        "step, integrator": ([(ours, ["step", "advance", "own_loads"])],
                             [(general / "integrator.py", ["*"]), (general / "pipeline.py", ["*"]),
                              (brax / "actuator.py", ["*"])]),
        "contact and limits, not benchmarked": ([(ours, ["Ground", "Stops", "bottoms", "ground_contact", "stop_loads",
                                                          "drop_loads"]), (top, ["lowest_point"])],
                                                [(general / "constraint.py", ["*"]), (brax / "contact.py", ["*"]),
                                                 (brax / "base.py", ["Contact"]),
                                                 (Path(collision_primitive.__file__),
                                                  ["_plane_sphere", "plane_sphere"])]),
    }
    return {part: tuple(sum(logical_lines(path, names) for path, names in side) for side in sides)
            for part, sides in parts.items()}


def labelled(axes, x: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict]) -> None:
    """Curves drawn log against log, each labelled at its end, the labels spread up or down by their ends' rank so
    that close ends stay legible."""
    ends = np.array([values[-1] for values in curves.values()])
    nudges = 9 * (np.argsort(np.argsort(ends)) - (len(ends) - 1) / 2)
    for (label, values), nudge in zip(curves.items(), nudges):
        axes.loglog(x, values, marker="o", **styles[label])
        axes.annotate(label, (x[-1], values[-1]), xytext=(6, nudge), textcoords="offset points",
                      color=styles[label]["color"], va="center")
    axes.spines[["top", "right"]].set_visible(False)


def throughput_chart(counts: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict], dt: float):
    """Humanoid steps a second against the humanoids in a batch, at one step size, log against log, with the line
    where every humanoid runs in real time."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(520, 320)
        axes.loglog(counts, counts / dt, color="0.55", linestyle="--")
        axes.text(counts[-3], counts[-3] / dt, "real time ", color="0.55", ha="right", va="bottom")
        labelled(axes, counts, curves, styles)
        axes.set_xlabel("humanoids")
        axes.set_ylabel(f"humanoid steps per second, {round(dt * 1000)} ms steps")
        figure.tight_layout()
    return figure


def compile_chart(counts: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict]):
    """Seconds to trace, lower and compile a call's steps against the humanoids in a batch, log against log."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(520, 320)
        labelled(axes, counts, curves, styles)
        axes.set_xlabel("humanoids")
        axes.set_ylabel("seconds to compile a call")
        figure.tight_layout()
    return figure


def lines_chart(parts: dict[str, tuple[int, int]]):
    """The lines of code each engine spends on each part of its method, side by side."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(560, 340)
        rows = np.arange(len(parts))[::-1]
        ours, brax = np.array(list(parts.values())).T
        # The totals of the parts the benchmark runs, all but the last.
        axes.barh(rows + 0.2, ours, height=0.38, color="tab:blue", label=f"numga, {ours[:-1].sum()} benchmarked")
        axes.barh(rows - 0.2, brax, height=0.38, color="tab:orange", label=f"Brax, {brax[:-1].sum()} benchmarked")
        axes.set_yticks(rows, list(parts))
        axes.set_xlabel("lines of code")
        axes.legend(frameon=False, loc="center right")
        axes.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
    return figure


def main() -> None:
    from examples.util.animation import save_figure

    if "--float32" in sys.argv:
        print(json.dumps(brax_sweep(robot_counts)))
        return
    if "--lines" in sys.argv:
        parts = components()
        print(parts)
        save_figure(lines_chart(parts), "articulated_brax_lines")
        return

    start, used = time.perf_counter(), time.process_time()
    sweeps = {"numga dense, float64": numga_sweep("dense", robot_counts),
              "numga sparse, float64": numga_sweep("sparse", robot_counts),
              "Brax, float64": brax_sweep(robot_counts)}
    print(f"CPU over wall time in this process: {(time.process_time() - used) / (time.perf_counter() - start):.2f}")
    float32 = subprocess.run([sys.executable, "-m", "examples.mechanics.articulated.brax_comparison", "--float32"],
                             capture_output=True, text=True, check=True)
    sweeps["Brax, float32"] = json.loads(float32.stdout.strip().splitlines()[-1])
    styles = {"numga dense, float64": dict(color="tab:blue"), "numga sparse, float64": dict(color="tab:cyan"),
              "Brax, float64": dict(color="tab:red"), "Brax, float32": dict(color="tab:orange", linestyle=":")}
    for label, sweep in sweeps.items():
        print(f"{label}: humanoid steps per second {np.round(sweep['rates']).astype(int).tolist()}, "
              f"lowering {np.round(sweep['lowering'], 2).tolist()}, compiling {np.round(sweep['compiling'], 2).tolist()}")
    rates = {label: np.array(sweep["rates"]) for label, sweep in sweeps.items()}
    save_figure(throughput_chart(robot_counts, rates, styles, dt), "articulated_brax_throughput")
    compiles = {label: np.add(sweep["lowering"], sweep["compiling"]) for label, sweep in sweeps.items()}
    save_figure(compile_chart(robot_counts, compiles, styles), "articulated_brax_compile")
    parts = components()
    print(parts)
    save_figure(lines_chart(parts), "articulated_brax_lines")


if __name__ == "__main__":
    main()
