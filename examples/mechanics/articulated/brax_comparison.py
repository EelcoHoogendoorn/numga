"""Benchmark and line-count comparison pits `examples.mechanics.articulated` against `brax.generalized` on Brax's seventeen-hinge humanoid in free space.

Both engines run reduced-coordinate kinematics, recursive Newton-Euler inverse dynamics, joint-space mass matrix assembly, and linear solves without floor contact or joint-limit constraints, starting from Brax's initial pose at a 3 ms step. Ahead-of-time JAX compilation separates trace-and-lowering time, XLA compile time, and steady-state step throughput across batch sizes from one to several thousand humanoids, comparing `numga` dense and sparse float64 execution against `brax.generalized` in float64 and float32.

Running this standalone benchmark requires a Python environment with `brax` installed from `references/brax`:
`OMP_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false python -m examples.mechanics.articulated.brax_comparison`.
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

robot_counts = 4 ** np.arange(7)
dt, n_steps = 0.003, 13
n_timed = 5
n_windows = 5
workspace = Path(__file__).parents[4]
humanoid_file = workspace / "references" / "brax" / "brax" / "envs" / "assets" / "humanoid.xml"


# --- plumbing -------------------------------------------------------------------------
def compiled_run(advance_fn, state, n_timed_calls: int):
    """Ahead-of-time compilation measures trace-and-lowering seconds, XLA compile seconds, best steady-state seconds per call across `n_windows`, and the final device state."""
    import jax

    state = jax.device_put(state, jax.devices()[0])
    start_time = time.perf_counter()
    lowered = jax.jit(advance_fn).lower(state)
    lowering_seconds = time.perf_counter() - start_time
    compiled = lowered.compile()
    compiling_seconds = time.perf_counter() - start_time - lowering_seconds
    state = jax.block_until_ready(compiled(state))
    window_durations = []
    for _ in range(n_windows):
        window_start = time.perf_counter()
        for _ in range(n_timed_calls):
            state = compiled(state)
        jax.block_until_ready(state)
        window_durations.append((time.perf_counter() - window_start) / n_timed_calls)
    return lowering_seconds, compiling_seconds, min(window_durations), state


def brax_free_system(step_dt: float):
    """Floorless, limit-free Brax `System` loads from `humanoid.xml` at step size `step_dt`."""
    from brax.io import mjcf

    xml = re.sub(r'<geom[^>]*name="floor"[^>]*/>', "", humanoid_file.read_text())
    xml = xml.replace('limited="true"', 'limited="false"')
    return mjcf.loads(xml).tree_replace({"opt.timestep": step_dt})


def brax_flight_state(system, n_robots: int):
    """Batched Brax `State` initializes `n_robots` humanoids at rest in the home pose with matched `flight_speeds()`."""
    import jax
    import mujoco
    from brax.generalized import pipeline
    from jax import numpy as jp
    from examples.mechanics.articulated import scenarios

    columns = [
        system.mj_model.jnt_dofadr[mujoco.mj_name2id(system.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in scenarios.joint_names
    ]
    speeds = jp.zeros(system.qd_size()).at[np.array(columns)].set(jp.asarray(scenarios.flight_speeds()))
    state = pipeline.init(system, system.init_q, speeds)
    assert state.con_jac.shape[0] == 0
    return jax.tree.map(lambda part: jp.broadcast_to(part, (n_robots,) + jp.shape(part)), state)


def brax_sweep(counts: np.ndarray) -> dict[str, list[float]]:
    """Step throughput, lowering time, and compilation time sweep across batch sizes `counts` for `brax.generalized`."""
    import jax
    from brax.generalized import pipeline
    from jax import numpy as jp

    system = brax_free_system(dt)
    sweep: dict[str, list[float]] = {"rates": [], "lowering": [], "compiling": []}
    for n_robots in counts:
        actions = jp.zeros((int(n_robots), system.act_size()))

        def step_batch(current):
            return jax.vmap(lambda single, act: pipeline.step(system, single, act))(current, actions)

        def advance_batch(current):
            return jax.lax.scan(lambda carry, _: (step_batch(carry), None), current, None, length=n_steps)[0]

        state = brax_flight_state(system, int(n_robots))
        state = jax.tree.map(lambda part, stepped: part.astype(stepped.dtype), state, jax.eval_shape(step_batch, state))
        lowering, compiling, call_seconds, state = compiled_run(advance_batch, state, n_timed)
        assert np.isfinite(np.asarray(state.qd)).all()
        sweep["rates"].append(float(n_robots * n_steps / call_seconds))
        sweep["lowering"].append(lowering)
        sweep["compiling"].append(compiling)
    return sweep


def numga_sweep(execution: str, counts: np.ndarray) -> dict[str, list[float]]:
    """Step throughput, lowering time, and compilation time sweep across batch sizes `counts` for `numga` in `execution` mode (`'dense'` or `'sparse'`)."""
    from numga import Extensor
    from numga.algebras import PGA3D
    from numga.backend.jax import JaxContext
    from examples.mechanics.articulated import core, scenarios

    context = JaxContext(PGA3D, dtype=np.float64, execution=execution)

    def materialized_parts(record):
        """Tuple or value with every `Extensor` moved into `context`."""
        if isinstance(record, Extensor):
            return record.materialize(context)
        return tuple(part.materialize(context) if isinstance(part, Extensor) else part for part in record)

    tree, home_axes, inertias, mass_points = scenarios.humanoid_model()
    home_axes, inertias, mass_points = materialized_parts((home_axes, inertias, mass_points))
    gravity = materialized_parts(-scenarios.mv.z.dual() * scenarios.gravity_strength)
    stiffness, damping = scenarios.hinge_parameters()
    hinge_stiffness = materialized_parts(stiffness * core.Scalar.dual())
    hinge_damping = materialized_parts(damping * core.Scalar.dual())
    step_fn = partial(
        core.free_step, tree, home_axes, inertias, mass_points, gravity, hinge_stiffness, hinge_damping
    )
    sweep: dict[str, list[float]] = {"rates": [], "lowering": [], "compiling": []}
    for n_robots in counts:
        initial = materialized_parts(scenarios.flight_state(int(n_robots)))
        lowering, compiling, call_seconds, state = compiled_run(
            lambda current: core.advance(step_fn, current, dt, n_steps), initial, n_timed
        )
        assert state[3].isfinite().all()
        sweep["rates"].append(float(n_robots * n_steps / call_seconds))
        sweep["lowering"].append(lowering)
        sweep["compiling"].append(compiling)
    return sweep


def logical_lines(path: Path, names: list[str]) -> int:
    """Non-blank, non-comment, non-docstring, non-import logical line count totals the named definitions in `path`."""
    source = path.read_text()
    module = ast.parse(source)
    skipped = set()
    for node in ast.walk(module):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            skipped.update(range(body[0].lineno, body[0].end_lineno + 1))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            skipped.update(range(node.lineno, node.end_lineno + 1))
    definitions: dict[str, list[ast.AST]] = {}
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
    """Transitive call closure lists every top-level function in `library` invoked by `users` as `<stem>.<name>`."""
    text = "\n".join(path.read_text() for path in users)
    defined = {node.name: node for node in ast.parse(library.read_text()).body if isinstance(node, ast.FunctionDef)}
    source = library.read_text().splitlines()
    wanted, used = set(re.findall(rf"\b{library.stem}\.(\w+)", text)) & set(defined), set()
    while wanted:
        name = wanted.pop()
        used.add(name)
        body = "\n".join(source[defined[name].lineno - 1 : defined[name].end_lineno])
        wanted |= (set(re.findall(r"\b(\w+)\(", body)) & set(defined)) - used
    return sorted(used)


def components() -> dict[str, tuple[int, int]]:
    """Subsystem logical line counts pair `articulated/core.py` (plus `spinning_top.core.lowest_point`) against `brax.generalized`."""
    from mujoco.mjx._src import collision_primitive

    local_core = Path(__file__).parent / "core.py"
    spinning_top_core = Path(__file__).parents[1] / "spinning_top" / "core.py"
    brax = workspace / "references" / "brax" / "brax"
    general = brax / "generalized"
    users = [brax / name for name in ("base.py", "scan.py", "kinematics.py", "actuator.py", "contact.py")] + [
        general / name
        for name in ("base.py", "constraint.py", "dynamics.py", "integrator.py", "mass.py", "pipeline.py")
    ]
    parts = {
        "spatial math": (
            [(local_core, [":module"])],
            [
                (
                    brax / "base.py",
                    ["Transform", "Motion", "Force", "Inertia", "_transform_do", "_transform_inv_do", "_motion_cross", "_"],
                ),
                (brax / "math.py", used_functions(brax / "math.py", users)),
            ],
        ),
        "model description": (
            [],
            [
                (brax / "base.py", [":module", "Base", "_tree_replace", "Link", "DoF", "Actuator", "State", "System"]),
                (general / "base.py", ["*"]),
            ],
        ),
        "kinematics": (
            [(local_core, ["kinematic_tree", "forward_kinematics"])],
            [
                (brax / "kinematics.py", used_functions(brax / "kinematics.py", users)),
                (brax / "scan.py", used_functions(brax / "scan.py", users)),
                (general / "dynamics.py", ["transform_com"]),
            ],
        ),
        "inverse dynamics": ([(local_core, ["inverse_dynamics"])], [(general / "dynamics.py", ["inverse"])]),
        "mass matrix": ([(local_core, ["rate_pullback", "mass_matrix"])], [(general / "mass.py", ["matrix"])]),
        "forward dynamics, solve": (
            [(local_core, ["coupled_rates", "forward_dynamics"])],
            [(general / "dynamics.py", ["forward", "_passive"]), (general / "mass.py", ["matrix_inv"])],
        ),
        "step, integrator": (
            [(local_core, ["free_step", "advance", "trajectory"])],
            [(general / "integrator.py", ["*"]), (general / "pipeline.py", ["*"]), (brax / "actuator.py", ["*"])],
        ),
        "contact and limits, not benchmarked": (
            [(local_core, ["ground_contact", "drop_step"]), (spinning_top_core, ["lowest_point"])],
            [
                (general / "constraint.py", ["*"]),
                (brax / "contact.py", ["*"]),
                (brax / "base.py", ["Contact"]),
                (Path(collision_primitive.__file__), ["_plane_sphere", "plane_sphere"]),
            ],
        ),
    }
    return {
        part: tuple(sum(logical_lines(path, names) for path, names in side) for side in sides)
        for part, sides in parts.items()
    }


def draw_labelled_curves(axes, x_values: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict]) -> None:
    """Log-log curves render with rank-staggered endpoint annotations so closely spaced right-hand endpoints do not overlap."""
    endpoints = np.array([values[-1] for values in curves.values()])
    nudges = 9 * (np.argsort(np.argsort(endpoints)) - (len(endpoints) - 1) / 2)
    for (label, values), nudge in zip(curves.items(), nudges):
        axes.loglog(x_values, values, marker="o", **styles[label])
        axes.annotate(
            label,
            (x_values[-1], values[-1]),
            xytext=(6, nudge),
            textcoords="offset points",
            color=styles[label]["color"],
            va="center",
        )
    axes.spines[["top", "right"]].set_visible(False)


def throughput_chart(counts: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict], step_dt: float):
    """Log-log throughput figure compares humanoid steps per second against batch size `counts` with a dashed real-time reference."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(520, 320)
        axes.loglog(counts, counts / step_dt, color="0.55", linestyle="--")
        axes.text(counts[-3], counts[-3] / step_dt, "real time ", color="0.55", ha="right", va="bottom")
        draw_labelled_curves(axes, counts, curves, styles)
        axes.set_xlabel("humanoids")
        axes.set_ylabel(f"humanoid steps per second, {round(step_dt * 1000)} ms steps")
        figure.tight_layout()
    return figure


def compile_chart(counts: np.ndarray, curves: dict[str, np.ndarray], styles: dict[str, dict]):
    """Log-log figure plots trace, lowering, and compilation seconds against batch size `counts`."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(520, 320)
        draw_labelled_curves(axes, counts, curves, styles)
        axes.set_xlabel("humanoids")
        axes.set_ylabel("seconds to compile a call")
        figure.tight_layout()
    return figure


def lines_chart(parts: dict[str, tuple[int, int]]):
    """Horizontal bar chart compares subsystem logical line counts between `numga` and `brax.generalized`."""
    from examples.util import charts

    with charts.style():
        figure, axes = charts.subplots(560, 340)
        rows = np.arange(len(parts))[::-1]
        numga_lines, brax_lines = np.array(list(parts.values())).T
        axes.barh(rows + 0.2, numga_lines, height=0.38, color="tab:blue", label=f"numga, {numga_lines[:-1].sum()} benchmarked")
        axes.barh(rows - 0.2, brax_lines, height=0.38, color="tab:orange", label=f"Brax, {brax_lines[:-1].sum()} benchmarked")
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

    wall_start, cpu_start = time.perf_counter(), time.process_time()
    sweeps = {
        "numga dense, float64": numga_sweep("dense", robot_counts),
        "numga sparse, float64": numga_sweep("sparse", robot_counts),
        "Brax, float64": brax_sweep(robot_counts),
    }
    print(f"CPU over wall time in this process: {(time.process_time() - cpu_start) / (time.perf_counter() - wall_start):.2f}")
    float32 = subprocess.run(
        [sys.executable, "-m", "examples.mechanics.articulated.brax_comparison", "--float32"],
        capture_output=True,
        text=True,
        check=True,
    )
    sweeps["Brax, float32"] = json.loads(float32.stdout.strip().splitlines()[-1])
    styles = {
        "numga dense, float64": dict(color="tab:blue"),
        "numga sparse, float64": dict(color="tab:cyan"),
        "Brax, float64": dict(color="tab:red"),
        "Brax, float32": dict(color="tab:orange", linestyle=":"),
    }
    for label, sweep in sweeps.items():
        print(
            f"{label}: humanoid steps per second {np.round(sweep['rates']).astype(int).tolist()}, "
            f"lowering {np.round(sweep['lowering'], 2).tolist()}, compiling {np.round(sweep['compiling'], 2).tolist()}"
        )
    rates = {label: np.array(sweep["rates"]) for label, sweep in sweeps.items()}
    save_figure(throughput_chart(robot_counts, rates, styles, dt), "articulated_brax_throughput")
    compiles = {label: np.add(sweep["lowering"], sweep["compiling"]) for label, sweep in sweeps.items()}
    save_figure(compile_chart(robot_counts, compiles, styles), "articulated_brax_compile")
    parts = components()
    print(parts)
    save_figure(lines_chart(parts), "articulated_brax_lines")


if __name__ == "__main__":
    main()
