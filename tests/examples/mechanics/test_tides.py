"""Tides: the geometric derivative of gravity is Poisson's equation; and the flyby takes a step on a coarse field and
draws."""

from __future__ import annotations

import numpy as np

from examples.mechanics.tides import core, render, scenarios


def test_the_geometric_derivative_of_gravity_is_minus_four_pi_times_the_density():
    masses, _, _ = scenarios.setting()
    points = core.grid(1.0, 1.0, 5, 5)                                       # [rows, columns] Point
    derivative = core.derivative(masses.tidal(points))                       # [rows, columns] Even
    softened = points.scalar_norm_squared() + scenarios.MASS_CORE**2
    density = 3 * scenarios.MASS * scenarios.MASS_CORE**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((derivative + 4 * np.pi * density).kernel, 0.0, atol=1e-10)


def test_the_flyby_takes_a_step_and_draws():
    points, _, density = scenarios.field(4, 4)
    drawn = ((stars, *scenarios.outline(stars, shape)) for stars, shape in scenarios.flyby(2, 1))
    list(render.animate(points, density, drawn))
