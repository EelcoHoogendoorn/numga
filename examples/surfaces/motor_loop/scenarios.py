"""Twisted and locally oscillating ribbon scenes compare linear point subdivision against Loop subdivision of motors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core

# Control-grid vertex counts along z and across x, physical dimensions, and pinned corner indices.
n_along, n_across = 5, 3
length, width = 4.0, 1.2
corner_indices = [0, n_across - 1, n_along * n_across - n_across, n_along * n_across - 1]
# Peak twist angle at the far end of the ribbon, frame count, and smooth frame duration in milliseconds.
twist_angle = 3 * np.pi
n_frames = 72
duration_ms = 40
# Oscillation amplitude and random seed for turning each vertex frame in place around its own axis.
turn_amplitude = np.pi / 3
turn_seed = 2
# Loop refinement depth and rendered axis-arrow length.
n_loop_steps = 3
axis_length = 0.3


# --- plumbing -------------------------------------------------------------------------
def ribbon_grid(n_rows: int, n_columns: int, span_z: float, span_x: float) -> tuple[np.ndarray, np.ndarray]:
    """Flattened coordinates along `z` and across `x` place the `[V]` vertices of an `n_rows`-by-`n_columns` ribbon grid."""
    z_coords, x_coords = np.meshgrid(
        np.linspace(-span_z / 2, span_z / 2, n_rows),
        np.linspace(-span_x / 2, span_x / 2, n_columns),
        indexing="ij",
    )
    return z_coords.reshape(-1), x_coords.reshape(-1)


def ribbon_setup() -> tuple[core.Topology, core.SubdividedTopology, np.ndarray, np.ndarray]:
    """Coarse `Topology.grid`, corner-pinned Loop refinement, and `[V]` grid coordinates define the ribbon surface."""
    control_topology = core.Topology.grid(n_along, n_across)
    V, E, F = control_topology.n_elements
    pinned = np.isin(np.arange(V), corner_indices)                          # [V]
    refined_topology = control_topology.loop_subdivided(pinned, n_loop_steps)
    z_coords, x_coords = ribbon_grid(n_along, n_across, length, width)      # [V], [V]
    return control_topology, refined_topology, z_coords, x_coords


def turning_axes(topology: core.Topology) -> tuple[core.Line, np.ndarray]:
    """Unit rotation lines through the origin (`[V] Line+`) and oscillation phases (`[V]`) assign each vertex its local turn."""
    V, E, F = topology.n_elements
    rng = np.random.default_rng(turn_seed)
    axes = core.mv(core.Line.derive.nondegenerate, rng.normal(size=(V, 3))).normalized()  # [V] Line+
    phases = rng.uniform(0, 2 * np.pi, V)                                   # [V]
    return axes, phases


def winding_bounds(rest_motors: core.Motor, markers: core.Point, z_coords: np.ndarray) -> core.Point:
    """Rest and peak-twist frame markers (`[2, 4] Point-[V]`) bound the ribbon across its wind-up."""
    twist_profile = ((z_coords + length / 2) / length) ** 2                 # [V]
    peak_turn = (core.mv.xy * (-twist_angle * twist_profile / 2)).exp()     # [V] Motor+
    extremes = stack([rest_motors, peak_turn * rest_motors]).field()        # [2] Motor+[V]
    return extremes[:, None] >> markers                                     # [2, 4] Point-[V]


def turning_bounds(rest_motors: core.Motor, markers: core.Point, axes: core.Line) -> core.Point:
    """Extreme positive and negative local turns (`[2, 4] Point-[V]`) bound the oscillating vertex frames."""
    extremes = (rest_motors * stack([(axes * (-turn_amplitude / 2)).exp(),
                                     (axes * (turn_amplitude / 2)).exp()])).field()  # [2] Motor+[V]
    return extremes[:, None] >> markers                                     # [2, 4] Point-[V]


def save_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments,
                name: str, duration_ms: int) -> list:
    """Streams `states` once through `animate(*arguments, ...)` into `save_animation` and collects the yielded states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    from examples.surfaces.motor_loop import render

    control_topology, refined_topology, z_coords, x_coords = ribbon_setup()
    markers = core.frame_markers(axis_length)                               # [4] Point-
    rest_motors = core.ribbon_rest_motors(z_coords, x_coords)               # [V] Motor+
    axes, phases = turning_axes(control_topology)

    winding_view = render.ribbon_view(winding_bounds(rest_motors, markers, z_coords))
    recorded_winding = save_states(
        core.winding_states(refined_topology, markers, rest_motors, z_coords, length, twist_angle, n_frames),
        render.surface_animation, control_topology, refined_topology, winding_view,
        name="motor_loop_ribbon", duration_ms=duration_ms,
    )

    turning_view = render.ribbon_view(turning_bounds(rest_motors, markers, axes))
    recorded_turning = save_states(
        core.turning_states(refined_topology, markers, rest_motors, axes, phases, turn_amplitude, n_frames),
        render.surface_animation, control_topology, refined_topology, turning_view,
        name="motor_loop_turning", duration_ms=duration_ms,
    )

    # checks
    # Untwisted, both schemes agree; wound up, point subdivision wrings the ribbon inward while motor subdivision keeps its width.
    centre_line = core.origin & core.mv.point([0.0, 0.0, 1.0])              # [] Antibivector-
    winding_endpoints = stack([recorded_winding[0][1], recorded_winding[n_frames // 2][1]]).batch()  # [2, 2, V_refined] Point-
    radial_distance = (centre_line & winding_endpoints).norm().mean(axis=-1)  # [2, 2] Scalar+
    (points_rest, motors_rest), (points_peak, motors_peak) = radial_distance
    assert (points_rest - motors_rest).abs() < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest

    # Turning frames in place leaves the point-subdivided surface fixed while bowing the motor-subdivided surface.
    turning_pair = stack([state[1] for state in recorded_turning]).batch()  # [n_frames, 2, V_refined] Point-
    point_surfaces, motor_surfaces = turning_pair[:, 0], turning_pair[:, 1]  # [n_frames, V_refined] Point-, [n_frames, V_refined] Point-
    assert ((point_surfaces & point_surfaces[:1]).norm() < 1e-12).all()
    assert ((motor_surfaces & point_surfaces).norm() > 0.05).any()

    # Right-multiplying every vertex motor by a common local rotation leaves the subdivided motor surface unchanged.
    shared_turn = (core.mv.yz * (-0.7 / 2)).exp()                           # [] Motor+
    shared_pair = (core.subdivided_motors(refined_topology, stack([rest_motors, rest_motors * shared_turn]).field()) >> core.origin).batch()  # [2, V_refined] Point-
    assert ((shared_pair[0] & shared_pair[1]).norm() < 1e-12).all()


if __name__ == "__main__":
    main()
