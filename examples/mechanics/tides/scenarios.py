"""Scenario parameters and animation driver for a star cluster stretched by a Plummer mass."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.tides import core

mass_centres = core.mv.vector([[0, 0, 0]]).dual()           # [1] Point-
mass_values = np.array([1.0])
mass_core = 0.3
n_stars = 300
cluster_radius = 0.25
seed = 1
start_position = [-3.0, 0.5, 0.0]
start_velocity = [0.5, 0.0, 0.0]
dt = 0.005
n_substeps = 8
half_width, half_height = 3.6, 2.4
n_columns, n_rows = 360, 240
n_frames = 150
frame_duration_ms = 40


def initial_cluster(n_samples: int) -> tuple[core.Point, core.Point, core.Point, core.Point]:
    """Uniform disc sampling places `n_samples` stars around `start_position` with common `start_velocity` (`[n_samples] Point-`)."""
    rng = np.random.default_rng(seed)
    # Radii grow as the square root so stars cover the disc uniformly under random rotor turns.
    radii = cluster_radius * np.sqrt(rng.uniform(size=n_samples))
    turns = (core.mv.xy * (-np.pi * rng.uniform(size=n_samples))).exp()      # [n_samples] Rotor+
    offsets = turns >> (core.mv.x.dual() * radii)                            # [n_samples] Point-
    guiding_centre = core.mv.vector(start_position).dual()                   # [] Point-
    guiding_velocity = core.mv.vector(start_velocity).dual()                 # [] Point-
    return guiding_centre + offsets, guiding_velocity.broadcast_to(offsets.shape), guiding_centre, guiding_velocity


def density_field(n_grid_columns: int, n_grid_rows: int) -> tuple[core.Point, core.Even, core.Scalar]:
    """Poisson's equation recovers the Plummer mass density from the geometric derivative of gravity (`[n_grid_rows, n_grid_columns] Scalar+`)."""
    points = core.sample_grid(half_width, half_height, n_grid_columns, n_grid_rows)  # [n_grid_rows, n_grid_columns] Point-
    field_derivative = core.gravity_derivative(
        core.tidal_map(points, mass_centres, mass_values, mass_core)
    )                                                                        # [n_grid_rows, n_grid_columns] Even+
    return points, field_derivative, -field_derivative.select[0] / (4 * np.pi)


def flyby(n_flyby_frames: int, n_step_substeps: int) -> Iterator[tuple[core.Point, core.Point, core.Spread]]:
    """Yields `(positions, star_centre, predicted_spread)` across `n_flyby_frames` frames of the cluster flyby."""
    positions, velocities, guiding_centre, guiding_velocity = initial_cluster(n_stars)
    yield from core.flyby_states(
        positions,
        velocities,
        guiding_centre,
        guiding_velocity,
        mass_centres,
        mass_values,
        mass_core,
        cluster_radius,
        dt,
        n_step_substeps,
        n_flyby_frames,
    )


def main() -> None:
    """Saves the flyby animation and verifies Poisson's equation and tidal spread tracking."""
    from examples.mechanics.tides import render
    from examples.util.animation import save_animation

    points, field_derivative, mass_density = density_field(n_columns, n_rows)
    states = []

    def streamed_states() -> Iterator[tuple[core.Point, core.Point, core.Spread]]:
        """Yields each flyby state once while recording it for downstream checks."""
        for state in flyby(n_frames, n_substeps):
            states.append(state)
            yield state

    save_animation(render.flyby_frames(points, mass_density, streamed_states()), "tides_flyby", frame_duration_ms)

    # checks
    # Gravity's geometric derivative has zero curl and recovers minus four pi times the Plummer density.
    source_offsets = mass_centres - points[..., None]                        # [n_rows, n_columns, 1] Point-
    softened_squares = source_offsets.scalar_norm_squared() + mass_core**2   # [n_rows, n_columns, 1] Scalar+
    plummer_density = (
        3 * mass_values * mass_core**2 / (4 * np.pi) / (softened_squares * softened_squares * softened_squares.square_root())
    ).sum(axis=-1)                                                           # [n_rows, n_columns] Scalar+
    assert (field_derivative.select[2].norm() < 1e-10).all()
    assert ((mass_density - plummer_density).abs() < 1e-10).all()

    # A uniform disc's second moment is a quarter of its squared radius, carried forward by the tidal deformation.
    n_sample_planes = 8
    sample_planes = (core.mv.xy * (-np.pi * np.arange(n_sample_planes) / n_sample_planes)).exp() >> core.mv.x  # [n_sample_planes] Plane+
    for state_positions, state_centre, state_spread in states:
        measured_moment = sample_planes & core.cluster_moment(state_positions, state_centre)(sample_planes)  # [n_sample_planes] Scalar+
        predicted_moment = sample_planes & (state_spread / 4)(sample_planes)  # [n_sample_planes] Scalar+
        assert ((measured_moment - predicted_moment).abs() <= 0.3 * predicted_moment.max()).all()


if __name__ == "__main__":
    main()
