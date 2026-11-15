"""Coverage tests verify Poisson's equation, tidal spread evolution, and flyby rendering on a coarse grid."""

from __future__ import annotations

import numpy as np

from examples.mechanics.tides import core, render, scenarios


def test_gravity_derivative_recovers_plummer_density_with_zero_curl():
    points = core.sample_grid(1.0, 1.0, 5, 5)                                # [5, 5] Point-
    tides = core.tidal_map(points, scenarios.mass_centres, scenarios.mass_values, scenarios.mass_core)  # [5, 5] Plane+ <- Point-
    field_derivative = core.gravity_derivative(tides)                        # [5, 5] Even+
    softened_squares = points.scalar_norm_squared() + scenarios.mass_core**2  # [5, 5] Scalar+
    plummer_density = (
        3 * scenarios.mass_values[0] * scenarios.mass_core**2 / (4 * np.pi)
        / (softened_squares * softened_squares * softened_squares.square_root())
    )                                                                        # [5, 5] Scalar+
    assert (field_derivative.select[2].norm() < 1e-10).all()
    assert ((field_derivative.select[0] + 4 * np.pi * plummer_density).abs() < 1e-10).all()


def test_flyby_steps_and_renders_frames():
    points, _, mass_density = scenarios.density_field(4, 4)
    states = list(scenarios.flyby(2, 1))
    positions, star_centre, predicted_spread = states[-1]
    moment = core.cluster_moment(positions, star_centre)                     # [] Point- <- Plane+
    assert ((core.mv.x & moment(core.mv.x)) > 0).all()
    snapshot = render.flyby_frame(points, mass_density, positions, star_centre, predicted_spread)
    frames = list(render.flyby_frames(points, mass_density, states))
    assert snapshot.ndim == 3
    assert len(frames) == 2
