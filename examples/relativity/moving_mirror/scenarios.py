"""Spacetime triangles, causal light-cone strips, and relativistic moving-mirror worldsheets."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.relativity.moving_mirror.core import (
    Plane,
    Point,
    causal_wake,
    cavity_slices,
    corner_interpolation,
    edge_whitney_forms,
    moving_mirror_wave,
    mv,
    space,
    spacetime_operators,
    time,
    triangle,
)

n_triangle_subdivisions = 7
n_wake_space = 81
n_wake_time = 140
wake_duration = 0.78
source_x, source_t = 0.5, 0.12
source_width = 0.032

n_mirror_space = 85
n_mirror_time = 380
mirror_duration = 1.25
initial_length = 1.0
stroke = 0.36
inflection_time = 0.54
transition_time = 0.42
pulse_centre = 0.28
pulse_width = 0.085
wavelength = 0.17
before_row = 72
after_row = 310
n_frames = 66
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Regular lattice of sample events (`[...] Point-`) covering `corners` (`Point-[3]`) and its boundary."""
    steps = np.arange(n_subdivisions + 1)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second <= n_subdivisions
    factors = np.stack(
        (n_subdivisions - first[inside] - second[inside], first[inside], second[inside]),
        axis=-1,
    ) / n_subdivisions
    return (corners.batch() * factors).sum(axis=-1)                            # [...] Point-


def single_triangle(space_axis: Point, time_axis: Point, n_subdivisions: int) -> tuple[Point, Point]:
    """Scalene spacetime triangle vertices (`Point-[3]`) and regular sample events (`[...] Point-`)."""
    corners = stack(
        (
            0 * space_axis + 0 * time_axis,
            1.15 * space_axis + 0.16 * time_axis,
            0.30 * space_axis + 0.92 * time_axis,
        )
    ).field()                                                                  # Point-[3]
    return corners, triangle_samples(corners, n_subdivisions)


def mirror_trajectory(
    times: np.ndarray,
    length: float,
    wall_stroke: float,
    inflection: float,
    transition: float,
) -> np.ndarray:
    """Smooth timelike right-mirror spatial positions (`[n_time]`) advancing inward by `wall_stroke`."""
    return length - wall_stroke * (1 + np.tanh((times - inflection) / transition)) / 2


def spacetime_strip(
    space_axis: Point,
    time_axis: Point,
    n_space: int,
    n_time: int,
    wall_positions: np.ndarray,
    duration: float,
) -> tuple[Topology, Point, float]:
    """Triangulated spacetime strip `Topology`, event field (`Point-[V]`), and time step `dt` between `0` and `wall_positions`."""
    reference_x = np.linspace(0, 1, n_space)
    times = np.linspace(0, duration, n_time)
    dt = float(times[1] - times[0])
    positions_x = reference_x[None, :] * wall_positions[:, None]
    positions_t = np.broadcast_to(times[:, None], positions_x.shape)
    n_vertices = n_space * n_time
    vertices = (positions_x.ravel() * space_axis + positions_t.ravel() * time_axis).field()  # Point-[V]

    columns, rows = np.meshgrid(np.arange(n_space - 1), np.arange(n_time - 1), indexing="xy")
    v00 = rows * n_space + columns
    v10 = rows * n_space + (columns + 1)
    v01 = (rows + 1) * n_space + columns
    v11 = (rows + 1) * n_space + (columns + 1)
    faces = np.concatenate(
        [
            np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
            np.stack([v00, v11, v01], axis=-1).reshape(-1, 3),
        ]
    )
    return Topology(n_vertices, faces), vertices, dt


def main() -> None:
    """Renders the light-cone figure, Doppler figure, and cavity animation to `plots/` and verifies causality and Doppler gain."""
    from examples.relativity.moving_mirror import render
    from examples.util.animation import save_animation, save_figure

    # 1. Simplex gradients, edge 1-forms, Lorentzian dual, and causal light-cone wake.
    corners, points = single_triangle(space, time, n_triangle_subdivisions)    # Point-[3], [...] Point-
    weights, gradients, _ = corner_interpolation(corners, points)
    edge_forms = edge_whitney_forms(corners, points)                           # [3, ...] Plane+
    wake_topology, wake_vertices, _ = spacetime_strip(
        space, time, n_wake_space, n_wake_time, np.ones(n_wake_time), wake_duration,
    )
    _, wake_wave_operator, wake_areas = spacetime_operators(wake_topology, wake_vertices)
    source_event = source_x * space + source_t * time                          # [] Point-
    wake = causal_wake(
        wake_topology,
        wake_vertices,
        wake_wave_operator,
        wake_areas,
        n_wake_space,
        n_wake_time,
        source_event,
        source_width,
    )
    save_figure(
        render.light_cone_figure(
            corners, points, gradients, edge_forms, wake_topology, wake_vertices, wake, source_event,
        ),
        "moving_mirror_light_cone",
    )

    # 2. Relativistic moving-mirror cavity and Doppler compression.
    mirror_times = np.linspace(0, mirror_duration, n_mirror_time)
    wall_positions = mirror_trajectory(
        mirror_times, initial_length, stroke, inflection_time, transition_time,
    )
    mirror_topology, mirror_vertices, dt = spacetime_strip(
        space, time, n_mirror_space, n_mirror_time, wall_positions, mirror_duration,
    )
    mirror_gradient, mirror_wave_operator, mirror_areas = spacetime_operators(
        mirror_topology, mirror_vertices,
    )
    _, _, wave_rate, slice_energies = moving_mirror_wave(
        mirror_topology,
        mirror_vertices,
        mirror_gradient,
        mirror_wave_operator,
        mirror_areas,
        n_mirror_space,
        n_mirror_time,
        dt,
        pulse_centre,
        pulse_width,
        wavelength,
    )
    save_figure(
        render.doppler_figure(
            mirror_topology,
            mirror_vertices,
            n_mirror_space,
            n_mirror_time,
            wave_rate,
            slice_energies,
            before_row,
            after_row,
        ),
        "moving_mirror_doppler",
    )

    # 3. Animated spacetime worldsheet slice and live 1D moving-mirror cavity.
    recorded_rows: list[int] = []

    def streaming_slices():
        for row in cavity_slices(n_mirror_time, n_frames):
            recorded_rows.append(row)
            yield row

    save_animation(
        render.cavity_animation(
            mirror_topology, mirror_vertices, n_mirror_space, n_mirror_time, wave_rate, streaming_slices(),
        ),
        "moving_mirror_cavity",
        frame_duration_ms,
    )

    # checks
    assert len(recorded_rows) == n_frames
    corner_events = corners.batch()                                            # [3] Point-
    corner_weights, _, _ = corner_interpolation(corners, corner_events)
    assert ((corner_weights - np.eye(3)).abs() < 1e-10).all()
    assert ((weights.sum(axis=0) - 1).abs() < 1e-10).all()
    gradient_sum = gradients.sum(axis=0)                                       # [] Plane+
    assert ((gradient_sum & space).abs() < 1e-10) and ((gradient_sum & time).abs() < 1e-10)

    tails, heads = triangle.corners[2, 1]
    classic_forms = (
        weights[tails] * gradients[heads, None] - weights[heads] * gradients[tails, None]
    ) * triangle.orientation[1].T
    form_residual = edge_forms - classic_forms                                 # [3, ...] Plane+
    assert ((form_residual & space).abs() < 1e-10).all()
    assert ((form_residual & time).abs() < 1e-10).all()
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners)).batch()[:, 0]  # [3] Point-
    midpoints = (triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * corners)).batch()[:, 0]  # [3] Point-
    midpoint_readings = edge_whitney_forms(corners, midpoints) & sides[None, :]  # [3, 3] Scalar+
    assert ((midpoint_readings - np.eye(3)).abs() < 1e-10).all()

    right_null, left_null = mv.x - mv.t, mv.x + mv.t                           # [] Plane+, Plane+
    assert (right_null & Plane.dual()(right_null)).abs() < 1e-10
    assert (left_null & Plane.dual()(left_null)).abs() < 1e-10

    displacements = wake_vertices.batch() - source_event
    distances = (mv.x & displacements).abs()
    delays = mv.t & displacements
    outside_cone = (distances > delays + 0.14) | (delays < -0.10)
    wake_batch = wake.batch()
    assert wake_batch[outside_cone].abs().max() < 0.02 * wake_batch.abs().max()

    wave_rate_grid = wave_rate.batch().reshape(n_mirror_time, n_mirror_space)
    assert wave_rate_grid[after_row].abs().max() > 1.85 * wave_rate_grid[before_row].abs().max()
    assert slice_energies[after_row] > 2.15 * slice_energies[before_row]


if __name__ == "__main__":
    main()
