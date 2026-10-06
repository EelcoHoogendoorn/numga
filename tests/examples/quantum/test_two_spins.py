"""Two spins: the exchange swaps spin up and spin down through a fully entangled state, the singlet is
left alone, measuring one spin steers the other through the correlation map, and the pair draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.two_spins import core, render, scenarios


def test_scenarios_pass_their_checks_and_draw():
    scenarios.singlet(0)
    angles, first, second, _, bells = scenarios.swap(21)
    for figure in (render.draw_state(first[0], second[0]), render.draw_exchange(angles, bells, first)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_steering_passes_its_checks_and_draws():
    directions, probability, steered = scenarios.steering(5, 6, 3, 0)
    assert len(render.animate_steering(directions, probability, steered)) == 5


def test_field_and_qubit_scenes_pass_their_checks_and_draw():
    fields = np.linspace(0.0, 4.0, 9)
    first, second = core.bloch(scenarios.FOUR)
    plt.close(render.draw_levels(fields, scenarios.field_sweep(fields, 1.0), (first | core.mv.z) + (second | core.mv.Z)))
    times = np.linspace(0.0, np.pi, 5)
    rates = np.array([0.0, 1.0])
    states, probability = scenarios.singlet_return(times, rates, 1.0)
    plt.close(render.draw_qubit(*core.qubit(states), times, probability, rates))
