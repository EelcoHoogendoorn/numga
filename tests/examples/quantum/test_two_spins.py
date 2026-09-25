"""Two spins: the exchange swaps spin up and spin down through a fully entangled state, the singlet is
left alone, and the pair draws."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.quantum.two_spins import render, scenarios


def test_scenarios_pass_their_checks_and_draw():
    scenarios.singlet(0)
    swapped = scenarios.swap(21)
    figure = render.draw_pair(*swapped, 10)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate_pair(*(part[:3] for part in swapped))) == 3
