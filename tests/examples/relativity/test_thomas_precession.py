"""Coverage tests verify Wigner rotation, velocity-space holonomy, Bargmann–Michel–Telegdi precession, and scene rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.thomas_precession import core, render, scenarios

mv = core.mv


def test_two_equal_perpendicular_boosts_leave_closed_form_wigner_turn() -> None:
    """Two perpendicular boosts of equal rapidity leave the closed-form spatial Wigner rotation fixing laboratory time."""
    rapidities = np.array([0.3, 1.0, 2.0])
    rotors = core.wigner_rotor(rapidities)
    assert ((mv.t & ((rotors >> core.time) - core.time)).abs() < 1e-6).all()
    angles = 2 * rotors.log().scalar_product(mv.xy)
    assert ((angles + 2 * np.arctan(np.tanh(rapidities / 2) ** 2)).abs() < 1e-6).all()


def test_box_round_trip_turns_spin_by_velocity_square_area() -> None:
    """One diamond round trip in the box rotates the gyroscope spin by the hyperbolic area of its square of momenta."""
    momentum = 0.8
    positions, velocities = core.box_trajectory(1.0, momentum, 2, 60)
    initial_spin = core.step_boost(core.time, velocities[0]) >> scenarios.gyroscope_spin
    *_, (_, final_spin) = core.box_states(positions, velocities, initial_spin)
    square_area = 4 * (
        2 * momentum * np.arcsinh(momentum / np.sqrt(1 + momentum**2))
        - np.arctan(momentum**2 / np.sqrt(1 + 2 * momentum**2))
    )
    turn = (mv.xy * (square_area / 2)).exp()
    expected_spin = turn >> scenarios.gyroscope_spin
    assert ((final_spin - expected_spin).antinorm() < 2e-3).all()


def test_scenes_stream_and_render() -> None:
    """Each animation generator yields RGB frames and the magic-momentum chart returns a matplotlib figure."""
    assert len(list(render.two_boosts_animation(core.two_boost_states(np.array([1.0]), 3)))) == 1
    velocities = core.orbit_velocities(np.array([1.25, 1.5, 1.75]), 1)
    initial_spins = core.step_boost(core.time, velocities[0]) >> scenarios.gyroscope_spin
    assert len(list(render.orbits_animation(velocities, core.orbit_states(velocities, initial_spins, 1)))) == 2
    positions, box_velocities = core.box_trajectory(1.0, 0.8, 1, 1)
    wall_turns = (mv.xy * (-np.arange(4) * np.pi / 4)).exp()
    wall_midpoints = wall_turns >> mv.x.dual()
    box_spin = core.step_boost(core.time, box_velocities[0]) >> scenarios.gyroscope_spin
    assert len(list(render.box_animation(1.0, wall_midpoints, core.box_states(positions[:1], box_velocities[:1], box_spin)))) == 1
    ring_turns = (mv.xy * (-np.linspace(0, 2 * np.pi, 7, endpoint=False) / 2)).exp()
    ring_points = ring_turns >> mv.x.dual()
    assert len(list(render.ring_animation(ring_points, core.ring_states(2.0, 1.0, 0.0, 0.1, scenarios.muon_spin, 1, 1, 1)))) == 2
    rate_ratios = core.anomalous_rates(np.array([2.0]), 1.0, np.array([0.0]), 0.1, scenarios.muon_spin, 5.0, 1)
    figure = render.magic_momentum_figure(np.zeros(1), rate_ratios)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
