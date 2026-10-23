"""Thomas precession: two boosts leave a turn by their triangle's area, a spin carried around a
circular orbit turns back by the cap's area, the magic Lorentz factor removes the electric field from
the anomalous precession, and the animations draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.thomas_precession import core, render, scenarios

mv = core.mv


def test_two_equal_perpendicular_boosts_leave_the_closed_form_turn():
    rapidities = np.array([0.3, 1.0, 2.0])
    turn = scenarios.wigner_turn(rapidities)
    np.testing.assert_allclose(((turn >> core.time) - core.time).kernel, 0.0, atol=1e-12)
    angle = 2 * turn.log().scalar_product(mv.xy).kernel[..., 0]
    np.testing.assert_allclose(angle, -2 * np.arctan(np.tanh(rapidities / 2) ** 2), atol=1e-12)


def test_a_circular_orbit_turns_the_spin_back_by_the_caps_area():
    gammas = np.array([1.25, 1.5])
    velocities = scenarios.orbit_velocities(gammas, 360)
    # A spin along x, the plane turning about it, in each orbit's first frame.
    *_, (velocity, spin) = scenarios.orbits(velocities, core.rest_boost(velocities[0]) >> core.about(mv.x.dual()), 1)
    expected = (mv.xy * (np.pi * (gammas - 1))).exp() >> core.about(mv.x.dual())
    np.testing.assert_allclose((spin - expected).kernel, 0.0, atol=2e-3)


def test_each_round_trip_in_the_box_turns_the_spin_back_by_the_area_of_its_velocities():
    """The velocities of one round trip enclose the square of side 2 * momentum in the plane of their
    spatial parts; on the hyperboloid its area is four times the integral of 1 / sqrt(1 + x**2 + y**2)
    over a quarter of it."""
    momentum = 0.8
    positions, path = scenarios.box_path(1.0, momentum, 2, 60)
    *_, (_, spin) = scenarios.bouncing(positions, path, core.rest_boost(path[0]) >> core.about(mv.x.dual()))
    a = momentum
    area = 4 * (2 * a * np.arcsinh(a / np.sqrt(1 + a * a)) - np.arctan(a * a / np.sqrt(1 + 2 * a * a)))
    expected = (mv.xy * (area / 2)).exp() >> core.about(mv.x.dual())
    np.testing.assert_allclose((spin - expected).kernel, 0.0, atol=2e-3)


def test_the_magic_lorentz_factor_removes_the_electric_field():
    electric = np.array([-0.2, 0.0, 0.2])
    gammas = np.array([2.0, np.sqrt(1 + 1 / 0.1)])[:, None]
    ratios = scenarios.anomalous_rates(gammas, 1.0, electric, 0.1, 5.0, 100).kernel[..., 0]
    np.testing.assert_allclose(ratios[1], 1.0, atol=1e-3)
    assert ratios[0, 0] < 0.9 and ratios[0, 2] > 1.1


def test_the_animations_draw():
    two = list(render.animate_two_boosts(scenarios.two_boosts(np.array([1.0]), np.linspace(0, 1, 5))))
    velocities = scenarios.orbit_velocities(np.array([1.25, 1.5, 1.75]), 4)
    orbit = list(render.animate_orbits(velocities, scenarios.orbits(velocities, core.rest_boost(velocities[0]) >> core.about(mv.x.dual()), 4)))
    positions, path = scenarios.box_path(1.0, 0.8, 2, 2)
    walls = (mv.xy * (-np.arange(4) * np.pi / 4)).exp() >> mv.x.dual()
    box = list(render.animate_box(1.0, walls, scenarios.bouncing(positions[:2], path[:2], core.rest_boost(path[0]) >> core.about(mv.x.dual()))))
    outline = (mv.xy * (-np.linspace(0, 2 * np.pi, 8) / 2)).exp() >> mv.x.dual()
    ring = list(render.animate_ring(outline, scenarios.ring(2.0, 1.0, 0.0, 0.1, 1, 4, 4)))
    assert [len(two), len(orbit), len(box), len(ring)] == [1, 2, 2, 2]
    plt.close(render.draw_magic(np.zeros(2), mv.scalar(np.ones((2, 2, 1)))))
