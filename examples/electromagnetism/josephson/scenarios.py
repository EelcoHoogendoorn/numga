"""Four scenarios evolve fluxons in a planar Josephson junction across a pendulum chain, boosted strips, a collapsing ring, and an open boundary."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.electromagnetism.josephson import core
from numga.topology import Topology

mv = core.mv

# Fluxon-antifluxon collision parameters and pendulum sampling stride along the narrow strip.
collision_offset, collision_rapidity = 8.0, 0.6
n_collision_steps, collision_stride, pendulum_step, pendulum_reach = 1500, 15, 6, 12.0
# Narrow strip dimensions in penetration depths, vertex counts, boost rapidities, and time-stepping parameters.
strip_length, strip_height, n_strip_columns, n_strip_rows = 40.0, 0.1, 801, 3
rapidities = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
n_rapidities = len(rapidities)
strip_origin, strip_dt, n_strip_steps = -12.0, 0.02, 800
# Square junction geometry and stepping parameters for the collapsing fluxon ring.
square_side, n_square_side, ring_radius = 24.0, 97, 8.0
ring_dt, n_ring_steps, ring_stride = 0.1, 240, 2
# Rectangular junction geometry, tilt angle, and stepping parameters for edge reflection.
junction_width, junction_height, n_wide_columns, n_wide_rows = 32.0, 16.0, 129, 65
line_tilt, line_offset, line_rapidity = 0.35, 4.0, 0.9
line_dt, n_line_steps, line_stride = 0.1, 360, 2
frame_duration_ms = 40


def junction_grid(n_columns: int, n_rows: int, width: float, height: float) -> tuple[Topology, np.ndarray]:
    """Uniformly spaced planar coordinates (`[V, 2]`) pair with a triangulated grid of `n_columns` by `n_rows` vertices centred at the origin."""
    x_grid, y_grid = np.meshgrid(np.linspace(-width / 2, width / 2, n_columns),
                                 np.linspace(-height / 2, height / 2, n_rows))
    return Topology.grid(n_rows, n_columns), np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1)


def pendulum_collision(n_columns: int) -> tuple[core.Point, Iterator[core.Phase]]:
    """Opposite phase-plane rotors launch an approaching fluxon and antifluxon along a narrow strip with sampled pivots (`[n_pendula] Point-`)."""
    topology, coordinates = junction_grid(n_columns, n_strip_rows, strip_length, strip_height)
    vertices = (core.x_step * coordinates[:, 0] + core.y_step * coordinates[:, 1]).field()  # Point-[V]
    areas, gradients = core.simplex_gradients(topology, vertices)
    tension = core.tension_operator(topology, areas, gradients)
    first_events = vertices + core.time_step * 0.0                            # Point-[V]
    next_events = vertices + core.time_step * strip_dt                        # Point-[V]
    first_turns = (core.fluxon_turn(first_events + core.x_step * collision_offset, collision_rapidity, 1.0) *
                   core.fluxon_turn(first_events - core.x_step * collision_offset, -collision_rapidity, -1.0))
    next_turns = (core.fluxon_turn(next_events + core.x_step * collision_offset, collision_rapidity, 1.0) *
                  core.fluxon_turn(next_events - core.x_step * collision_offset, -collision_rapidity, -1.0))
    phases = first_turns >> mv.a                                              # Phase+[V]
    rates = (phases ^ (next_turns >> mv.a)) / strip_dt                        # Rate+[V]
    middle_mask = np.isclose(coordinates[:, 1], 0.0, atol=1e-9) & (np.abs(coordinates[:, 0]) <= pendulum_reach)
    pendulum_indices = np.flatnonzero(middle_mask)[::pendulum_step]
    n_pendula = len(pendulum_indices)
    pivots = vertices.batch()[pendulum_indices]                               # [n_pendula] Point-
    frames = core.phase_frames(tension, phases, rates, strip_dt, n_collision_steps, collision_stride)
    return pivots, (frame_phases.batch()[pendulum_indices] for frame_phases in frames)


def lorentz_contraction(n_columns: int, n_steps: int) -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    """Straight fluxons evolved across `rapidities` on a narrow strip give speeds (`[n_rapidities]`) with initial and final squared field strengths (`[n_rapidities] Scalar+[F]`)."""
    topology, coordinates = junction_grid(n_columns, n_strip_rows, strip_length, strip_height)
    vertices = (core.x_step * coordinates[:, 0] + core.y_step * coordinates[:, 1]).field()  # Point-[V]
    areas, gradients = core.simplex_gradients(topology, vertices)
    tension = core.tension_operator(topology, areas, gradients)
    shifted_vertices = vertices - core.x_step * strip_origin                  # Point-[V]
    first_turns = core.fluxon_turn(shifted_vertices + core.time_step * 0.0, rapidities, 1.0)
    next_turns = core.fluxon_turn(shifted_vertices + core.time_step * strip_dt, rapidities, 1.0)
    phases = first_turns >> mv.a                                              # [n_rapidities] Phase+[V]
    rates = (phases ^ (next_turns >> mv.a)) / strip_dt                        # [n_rapidities] Rate+[V]
    snapshots = list(core.phase_frames(tension, phases, rates, strip_dt, n_steps, n_steps))
    initial_strengths = core.field_strengths(topology, gradients, snapshots[0])  # [n_rapidities] Scalar+[F]
    final_strengths = core.field_strengths(topology, gradients, snapshots[-1])   # [n_rapidities] Scalar+[F]
    return np.tanh(rapidities), initial_strengths, final_strengths


def collapsing_ring(n_side: int) -> tuple[Topology, core.Point, Iterator[core.Rate]]:
    """Line tension shrinks a circular fluxon loop initialized at rest on a square junction and streams its tunnelling current (`Rate+[V]`)."""
    topology, coordinates = junction_grid(n_side, n_side, square_side, square_side)
    vertices = (core.x_step * coordinates[:, 0] + core.y_step * coordinates[:, 1]).field()  # Point-[V]
    areas, gradients = core.simplex_gradients(topology, vertices)
    tension = core.tension_operator(topology, areas, gradients)
    distances = vertices.antinorm()                                           # Scalar+[V]
    angles = 4 * (distances - ring_radius).exp().arctan()                     # Scalar+[V]
    ring_turns = (mv.ab * (-angles / 2)).exp()                                # Turn+[V]
    phases = ring_turns >> mv.a                                               # Phase+[V]
    frames = core.phase_frames(tension, phases, 0 * (mv.a ^ phases), ring_dt, n_ring_steps, ring_stride)
    return topology, vertices, (mv.a ^ frame_phases for frame_phases in frames)


def edge_reflection(n_columns: int, n_rows: int) -> tuple[Topology, core.Point, Iterator[core.Rate]]:
    """Spatial rotation and a boost launch an oblique fluxon line toward an open junction boundary and stream its tunnelling current (`Rate+[V]`)."""
    topology, coordinates = junction_grid(n_columns, n_rows, junction_width, junction_height)
    vertices = (core.x_step * coordinates[:, 0] + core.y_step * coordinates[:, 1]).field()  # Point-[V]
    areas, gradients = core.simplex_gradients(topology, vertices)
    tension = core.tension_operator(topology, areas, gradients)
    tilt_rotor = (mv.xy * (line_tilt / 2)).exp()                              # [] Rotor+
    shifted_vertices = vertices - core.x_step * line_offset                   # Point-[V]
    first_events = tilt_rotor >> (shifted_vertices + core.time_step * 0.0)    # Point-[V]
    next_events = tilt_rotor >> (shifted_vertices + core.time_step * line_dt)  # Point-[V]
    first_turns = core.fluxon_turn(first_events, line_rapidity, 1.0)          # Turn+[V]
    next_turns = core.fluxon_turn(next_events, line_rapidity, 1.0)            # Turn+[V]
    phases = first_turns >> mv.a                                              # Phase+[V]
    rates = (phases ^ (next_turns >> mv.a)) / line_dt                         # Rate+[V]
    frames = core.phase_frames(tension, phases, rates, line_dt, n_line_steps, line_stride)
    return topology, vertices, (mv.a ^ frame_phases for frame_phases in frames)


def main() -> None:
    from examples.electromagnetism.josephson import render
    from examples.util.animation import save_animation, save_figure

    pivots, pendulum_stream = pendulum_collision(n_strip_columns)
    pendulum_states = list(pendulum_stream)
    save_animation(render.pendulum_frames(pivots, pendulum_states), "josephson_pendula", frame_duration_ms)

    speeds, initial_strengths, final_strengths = lorentz_contraction(n_strip_columns, n_strip_steps)
    save_figure(render.contraction_figure(speeds, initial_strengths, final_strengths), "josephson_contraction")

    ring_topology, ring_vertices, ring_stream = collapsing_ring(n_square_side)
    ring_currents = list(ring_stream)
    save_animation(render.current_frames(ring_topology, ring_vertices, ring_currents),
                   "josephson_ring", frame_duration_ms)

    wide_topology, wide_vertices, line_stream = edge_reflection(n_wide_columns, n_wide_rows)
    line_currents = list(line_stream)
    save_animation(render.current_frames(wide_topology, wide_vertices, line_currents),
                   "josephson_reflection", frame_duration_ms)

    # checks
    assert ((pendulum_states[-1].norm() - 1).abs() < 1e-12).all()
    assert (mv.a | pendulum_states[-1]).min() < -0.9
    lorentz_factors = 1 / np.cosh(rapidities)
    for strengths in (initial_strengths, final_strengths):
        peaks = strengths.batch().max(axis=-1).square_root()
        assert ((peaks[0] / peaks - lorentz_factors).abs() < 0.03 * lorentz_factors).all()
    ring_distances = ring_vertices.antinorm().batch()
    ring_radii = [ring_distances[ring_currents[index].batch().norm() > 0.5].max() for index in (0, 30)]
    assert ring_radii[1] < ring_radii[0] - 2
    middle_mask = (mv.y & wide_vertices.batch()).abs() < 1e-9
    middle_x = mv.x & wide_vertices.batch()[middle_mask]
    current_dual = mv.ab.dual()
    start_current = line_currents[0].batch()[middle_mask] & current_dual
    end_current = line_currents[-1].batch()[middle_mask] & current_dual
    start_span = middle_x[start_current.argmax()] - middle_x[start_current.argmin()]
    end_span = middle_x[end_current.argmax()] - middle_x[end_current.argmin()]
    assert start_span * end_span < 0


if __name__ == "__main__":
    main()
