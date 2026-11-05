"""Thomas precession: two boosts leave a turn by their triangle's area, a round trip in the box turns the
spin back by the area of its velocities, and each scene runs a frame or two and draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.thomas_precession import core, render, scenarios

mv = core.mv


def test_two_equal_perpendicular_boosts_leave_the_closed_form_turn():
    rapidities = np.array([0.3, 1.0, 2.0])
    turn = scenarios.wigner_turn(rapidities)
    np.testing.assert_allclose(((turn >> core.time) - core.time).kernel, 0.0, atol=1e-12)
    angle = 2 * turn.log().scalar_product(mv.xy).to_array()
    np.testing.assert_allclose(angle, -2 * np.arctan(np.tanh(rapidities / 2) ** 2), atol=1e-12)


def test_each_round_trip_in_the_box_turns_the_spin_back_by_the_area_of_its_velocities():
    """The velocities of one round trip enclose the square of side 2 * momentum in the plane of their
    spatial parts; on the hyperboloid its area is four times the integral of 1 / sqrt(1 + x**2 + y**2)
    over a quarter of it."""
    momentum = 0.8
    positions, path = scenarios.box_path(1.0, momentum, 2, 60)
    *_, (_, spin) = scenarios.bouncing(positions, path, core.rest_boost(path[0]) >> scenarios.gyroscope_spin)
    a = momentum
    area = 4 * (2 * a * np.arcsinh(a / np.sqrt(1 + a * a)) - np.arctan(a * a / np.sqrt(1 + 2 * a * a)))
    expected = (mv.xy * (area / 2)).exp() >> scenarios.gyroscope_spin
    np.testing.assert_allclose((spin - expected).kernel, 0.0, atol=2e-3)


def test_the_scenes_run_and_draw():
    assert len(list(render.animate_two_boosts(scenarios.two_boosts(np.array([1.0]), np.zeros(1))))) == 1
    velocities = scenarios.orbit_velocities(np.array([1.25, 1.5, 1.75]), 1)
    start_spin = core.rest_boost(velocities[0]) >> scenarios.gyroscope_spin
    assert len(list(render.animate_orbits(velocities, scenarios.orbits(velocities, start_spin, 1)))) == 2
    positions, path = scenarios.box_path(1.0, 0.8, 1, 1)
    walls = (mv.xy * (-np.arange(4) * np.pi / 4)).exp() >> mv.x.dual()
    box_spin = core.rest_boost(path[0]) >> scenarios.gyroscope_spin
    list(render.animate_box(1.0, walls, scenarios.bouncing(positions[:1], path[:1], box_spin)))
    outline = (mv.xy * (-np.linspace(0, 2 * np.pi, 7, endpoint=False) / 2)).exp() >> mv.x.dual()
    list(render.animate_ring(outline, scenarios.ring(2.0, 1.0, 0.0, 0.1, 1, 1, 1)))
    ratios = scenarios.anomalous_rates(np.array([[2.0]]), 1.0, np.array([0.0]), 0.1, 5.0, 1)
    plt.close(render.draw_magic(np.zeros(1), ratios))
