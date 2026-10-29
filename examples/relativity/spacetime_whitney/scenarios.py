"""Spacetime triangles, light-cone wake meshes, and relativistic moving-mirror worldsheets."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.relativity.spacetime_whitney.core import (
    Mesh, Point, Plane, mv, space, time,
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
    """A barycentric lattice covering a spacetime triangle, including its boundary."""
    steps = np.arange(n_subdivisions + 1)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second <= n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-


def single_triangle(space_axis: Point, time_axis: Point, n_subdivisions: int) -> tuple[Point, Point]:
    """A scalene spacetime triangle and its interior sample events."""
    corners = stack((0.0 * space_axis + 0.0 * time_axis,
                     1.15 * space_axis + 0.16 * time_axis,
                     0.30 * space_axis + 0.92 * time_axis))                  # [corners] Point-
    return corners, triangle_samples(corners, n_subdivisions)


def mirror_trajectory(times: np.ndarray, length: float, wall_stroke: float,
                      inflection: float, transition: float) -> np.ndarray:
    """Right-mirror spatial position along a smooth timelike worldline."""
    return length - wall_stroke * (1.0 + np.tanh((times - inflection) / transition)) / 2.0


def spacetime_strip(space_axis: Point, time_axis: Point, n_space: int, n_time: int,
                    wall_positions: np.ndarray, duration: float) -> tuple[Mesh, Point, float]:
    """A triangulated strip of spacetime between a wall at rest at the origin and a wall at `wall_positions` at each time."""
    reference_x = np.linspace(0.0, 1.0, n_space)
    times = np.linspace(0.0, duration, n_time)
    dt = float(times[1] - times[0])
    positions_x = reference_x[None, :] * wall_positions[:, None]
    positions_t = np.broadcast_to(times[:, None], positions_x.shape)
    vertices = (positions_x.ravel() * space_axis + positions_t.ravel() * time_axis).field()  # Point-[vertices]

    columns, rows = np.meshgrid(np.arange(n_space - 1), np.arange(n_time - 1), indexing="xy")
    v00 = rows * n_space + columns
    v10 = rows * n_space + (columns + 1)
    v01 = (rows + 1) * n_space + columns
    v11 = (rows + 1) * n_space + (columns + 1)
    faces = np.concatenate([
        np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
        np.stack([v00, v11, v01], axis=-1).reshape(-1, 3),
    ])
    return Mesh(len(vertices.batch()), faces), vertices, dt


def main() -> None:
    from examples.relativity.spacetime_whitney.core import (
        Operators, barycentric_hats, causal_wake, edge_whitney_forms, moving_mirror_wave,
    )
    from examples.relativity.spacetime_whitney import render
    from examples.util.animation import save_animation, save_figure

    # 1. Spacetime Whitney forms, the Lorentzian dual, and the causal light-cone wake.
    corners, points = single_triangle(space, time, n_triangle_subdivisions)
    hats, gradients, _ = barycentric_hats(corners, points)
    edge_forms = edge_whitney_forms(corners, points)                         # [corners, samples] Plane+
    wake_mesh, wake_vertices, _ = spacetime_strip(
        space, time, n_wake_space, n_wake_time, np.ones(n_wake_time), wake_duration,
    )
    wake_operators = Operators(wake_mesh, wake_vertices)
    source_event = source_x * space + source_t * time                        # [] Point-
    wake = causal_wake(wake_operators, n_wake_space, n_wake_time, source_event, source_width)
    save_figure(
        render.draw_light_cone_whitney(
            corners, points, gradients, edge_forms,
            wake_mesh, wake_vertices, wake, source_event,
        ),
        "spacetime_whitney_cone",
    )

    # 2. Relativistic moving-mirror cavity and Einstein Doppler frequency compression.
    mirror_times = np.linspace(0.0, mirror_duration, n_mirror_time)
    wall_positions = mirror_trajectory(
        mirror_times, initial_length, stroke, inflection_time, transition_time,
    )
    mirror_mesh, mirror_vertices, dt = spacetime_strip(
        space, time, n_mirror_space, n_mirror_time, wall_positions, mirror_duration,
    )
    mirror_operators = Operators(mirror_mesh, mirror_vertices)
    _, _, wave_rate, slice_energies = moving_mirror_wave(
        mirror_operators, n_mirror_space, n_mirror_time, dt,
        pulse_centre, pulse_width, wavelength,
    )
    save_figure(
        render.draw_moving_mirror_doppler(
            mirror_mesh, mirror_vertices, n_mirror_space, n_mirror_time,
            wave_rate, slice_energies, before_row, after_row,
        ),
        "spacetime_whitney_doppler",
    )

    # 3. Animated spacetime worldsheet slice and live 1D moving-mirror cavity.
    frame_rows = np.linspace(4, n_mirror_time - 4, n_frames, dtype=int)
    save_animation(
        render.animate_moving_mirror(
            mirror_mesh, mirror_vertices, n_mirror_space, n_mirror_time,
            wave_rate, frame_rows,
        ),
        "spacetime_whitney_mirror",
        frame_duration_ms,
    )

    # checks
    corner_hats, _, _ = barycentric_hats(corners, corners)
    np.testing.assert_allclose(corner_hats.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    first, second = [1, 2, 0], [2, 0, 1]
    classic_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - classic_forms).kernel, 0.0, atol=1e-14)
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    midpoints = (corners[first] + corners[second]) / 2
    midpoint_readings = edge_whitney_forms(corners, midpoints) & sides[None, :]
    np.testing.assert_allclose(midpoint_readings.kernel, np.eye(3)[..., None], atol=1e-14)

    # A null gradient has zero pairing with its Lorentzian dual.
    right_null, left_null = mv.x - mv.t, mv.x + mv.t
    np.testing.assert_allclose((right_null & Plane.dual()(right_null)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((left_null & Plane.dual()(left_null)).kernel, 0.0, atol=1e-14)

    # Causality of the Green wake: negligible outside the forward 45-degree light cone of source_event.
    displacements = wake_vertices.batch() - source_event
    distances = (mv.x & displacements).norm()
    delays = mv.t & displacements
    outside_cone = (distances > delays + 0.14) | (delays < -0.10)
    wake_batch = wake.batch()
    assert np.max(np.abs(wake_batch[outside_cone].kernel)) < 0.02 * np.max(np.abs(wake_batch.kernel))

    # Relativistic moving mirror amplifies wave energy and peak wave rate by the Doppler factor.
    wave_rate_grid = wave_rate.batch().reshape(n_mirror_time, n_mirror_space)
    assert np.max(np.abs(wave_rate_grid[after_row].kernel)) > 1.85 * np.max(np.abs(wave_rate_grid[before_row].kernel))
    assert slice_energies[after_row] > 2.15 * slice_energies[before_row]


if __name__ == "__main__":
    main()
