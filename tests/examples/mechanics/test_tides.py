"""Tides: the derivative of gravity is Poisson's equation, the tidal map predicts the stretched
cluster's shape, and a frame draws."""

from __future__ import annotations

import numpy as np

from examples.mechanics.tides import core, render, scenarios


def test_contracting_the_tidal_map_gives_minus_four_pi_times_the_density():
    masses, _, _ = scenarios.setting()
    points = core.grid(1.0, 1.0, 5, 5)                                       # [rows, columns] Vector
    derivative = core.derivative(core.tidal(masses, points))                 # [rows, columns] Even
    softened = (points | points) + scenarios.MASS_CORE**2
    density = 3 * scenarios.MASS * scenarios.MASS_CORE**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((derivative + 4 * np.pi * density).kernel, 0.0, atol=1e-10)


def test_the_tidal_map_predicts_the_cluster_through_the_core_and_a_frame_draws():
    directions = (core.mv.xy * (-np.pi * np.arange(8) / 8)).exp() >> core.mv.x   # [directions] Vector
    for stars, shape in scenarios.flyby(75):
        measured = (directions | stars.moment()(directions)).to_array()
        predicted = (directions | (core.spread(shape, scenarios.CLUSTER_RADIUS) / 4)(directions)).to_array()
        np.testing.assert_allclose(measured, predicted, rtol=0, atol=0.3 * predicted.max())
    masses, _, _ = scenarios.setting()
    points = core.grid(1.0, 1.0, 8, 8)                                       # [rows, columns] Vector
    pixels = render.frame(points, core.derivative(core.tidal(masses, points)), stars, *scenarios.outline(stars, shape))
    assert pixels.ndim == 3
