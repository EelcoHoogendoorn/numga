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
    np.testing.assert_allclose(core.expectation(core.ONE, state).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((core.COUPLING * state - 3 * state).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(reached.to_array(), 2 * np.sqrt(2), atol=1e-12)
    # The exchange keeps the state normalized; halfway the spins are fully entangled, and at the end they
    # have swapped.
    n_frames = 21
    angles, first, second, _, bells, states = scenarios.swap(n_frames)
    np.testing.assert_allclose(core.expectation(core.ONE, states).to_array(), 1.0, atol=1e-7)
    middle = n_frames // 2
    np.testing.assert_allclose(first[middle].kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose(bells.to_array()[[0, middle, -1]], [2.0, 2 * np.sqrt(2), 2.0], atol=1e-7)
    np.testing.assert_allclose((first[-1] + mv.z).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((second[-1] - mv.Z).kernel, 0.0, atol=1e-7)
    for figure in (render.draw_state(first[0], second[0]), render.draw_exchange(angles, bells, first)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_steering_passes_its_checks_and_draws():
    directions, probability, steered, states = scenarios.steering(5, 6, 3)
    # The probability is set by the first spin's Bloch vector alone, and the second spin's Bloch vector
    # after the measurement is the correlation map read backwards.
    first, second = core.bloch(states[:, None, None])
    correlations = core.correlation(states[:, None, None])
    np.testing.assert_allclose((probability - 0.5 * (1 + (directions | first))).kernel, 0.0, atol=1e-11)
    backwards = (second + correlations.adjoint()(directions)) / (1 + (directions | first))
    np.testing.assert_allclose((steered - backwards).kernel, 0.0, atol=1e-9)
    assert len(list(render.animate_steering(directions, probability, steered))) == 5


def test_field_and_qubit_scenes_pass_their_checks_and_draw():
    fields = np.linspace(0.0, 4.0, 9)
    first, second = core.bloch(scenarios.FOUR)
    energies = scenarios.field_sweep(fields, 1.0)
    # The singlet sits below the triplet of zero spin by twice the exchange rate, and the field moves the
    # triplets with both spins along it or against it.
    expected = np.stack([-1.0 + 0 * fields, 1.0 + 0 * fields, 1.0 - fields, 1.0 + fields], axis=-1)
    np.testing.assert_allclose(energies.to_array(), expected, atol=1e-12)
    plt.close(render.draw_levels(fields, energies, (first | core.mv.z) + (second | core.mv.Z)))
    times = np.linspace(0.0, np.pi, 5)
    rates = np.array([0.0, 1.0])
    states, probability = scenarios.singlet_return(times, rates, 1.0)
    # The pair returns to the singlet as one less the field difference's share of the rate squared, times
    # the sine of the rate times time squared.
    rate = np.hypot(rates[:, None], 1.0)
    np.testing.assert_allclose(probability.to_array(), 1 - (1.0 / rate) ** 2 * np.sin(rate * times) ** 2, atol=1e-12)
    plt.close(render.draw_qubit(*core.qubit(states), times, probability, rates))
