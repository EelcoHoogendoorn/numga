"""Two spins: the exchange swaps spin up and spin down through a fully entangled state, the singlet is
left alone, measuring one spin steers the other through the correlation map, and the pair draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.two_spins import core, render, scenarios


def test_scenarios_pass_their_checks_and_draw():
    mv = core.mv
    # The singlet is normalized, the coupling acts on it as three, and it reaches the Bell bound.
    state, reached = scenarios.singlet()
    np.testing.assert_allclose(core.expectation(core.one, state).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((core.coupling * state - 3 * state).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(reached.to_array(), 2 * np.sqrt(2), atol=1e-12)
    # The exchange keeps the state normalized; halfway the spins are fully entangled, and at the end they
    # have swapped.
    n_frames = 21
    angles, first, second, bells, states = scenarios.swap(n_frames)
    np.testing.assert_allclose(core.expectation(core.one, states).to_array(), 1.0, atol=1e-7)
    middle = n_frames // 2
    np.testing.assert_allclose(first[middle].kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose(bells.to_array()[[0, middle, -1]], [2.0, 2 * np.sqrt(2), 2.0], atol=1e-7)
    np.testing.assert_allclose((first[-1] + mv.xy).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((second[-1] - mv.XY).kernel, 0.0, atol=1e-7)
    for figure in (render.draw_state(first[0], second[0], render.SPIN_PLANES), render.draw_exchange(angles, bells, first, render.SPIN_PLANES)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_steering_passes_its_checks_and_draws():
    planes, probability, steered, states = scenarios.steering(5, 6, 3)
    # The probability is set by the first spin's plane alone, and the second spin's plane after the
    # measurement is the correlation map read backwards.
    first, second = core.spin_planes(states[:, None, None])
    correlations = core.correlation(states[:, None, None])
    np.testing.assert_allclose((probability - 0.5 * (1 + planes.dot(first))).kernel, 0.0, atol=1e-11)
    backwards = (second + correlations.adjoint()(planes)) / (1 + planes.dot(first))
    np.testing.assert_allclose((steered - backwards).kernel, 0.0, atol=1e-9)
    assert len(list(render.animate_steering(planes, probability, steered, render.SPIN_PLANES))) == 5


def test_field_and_qubit_scenes_pass_their_checks_and_draw():
    fields = np.linspace(0.0, 4.0, 9)
    first, second = core.spin_planes(scenarios.four)
    energies = scenarios.field_sweep(fields, 1.0)
    # The singlet sits below the triplet of zero spin by twice the exchange rate, and the field moves the
    # triplets with both spins in its plane or against it.
    expected = np.stack([-1.0 + 0 * fields, 1.0 + 0 * fields, 1.0 - fields, 1.0 + fields], axis=-1)
    np.testing.assert_allclose(energies.to_array(), expected, atol=1e-12)
    plt.close(render.draw_levels(fields, energies, core.field_plane.dot(first + second)))
    times = np.linspace(0.0, np.pi, 5)
    rates = np.array([0.0, 1.0])
    states, probability = scenarios.singlet_return(times, rates, 1.0)
    # The pair returns to the singlet as one less the field difference's share of the rate squared, times
    # the sine of the rate times time squared.
    rate = np.hypot(rates[:, None], 1.0)
    np.testing.assert_allclose(probability.to_array(), 1 - (1.0 / rate) ** 2 * np.sin(rate * times) ** 2, atol=1e-12)
    plt.close(render.draw_qubit(*core.qubit(states), times, probability, rates))
