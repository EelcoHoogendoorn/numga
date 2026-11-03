"""Fluxons in a planar Josephson junction: boosted fluxon lines contract as Lorentz says, a ring of flux collapses, and
a fluxon line comes back from the junction's edge as an antifluxon."""

import numpy as np

from examples.electromagnetism.josephson import core, render, scenarios


def test_boosted_fluxons_contract():
    speeds, before, after = scenarios.contracting()
    for strengths in (before, after):
        peaks = np.sqrt(strengths.to_array().max(axis=-1))
        np.testing.assert_allclose(peaks[0] / peaks, np.sqrt(1 - speeds ** 2), rtol=0.03)
    render.draw_contraction(speeds, before, after)
    pivots, frames = scenarios.meeting()
    assert next(render.animate_pendula(pivots, frames)).shape[-1] == 3


def test_ring_collapses_and_fluxon_reflects_as_antifluxon():
    topology, vertices, currents = scenarios.collapsing()
    currents = list(currents)
    _, places = core.junction(scenarios.n_square, scenarios.n_square, scenarios.square_side, scenarios.square_side)
    distances = np.hypot(places[:, 0], places[:, 1])
    reach = [distances[np.abs(np.asarray(currents[index].batch().kernel)[..., 0]) > 0.5].max() for index in (0, 30)]
    assert reach[1] < reach[0] - 2
    assert next(render.animate(topology, vertices, currents[:1])).shape[-1] == 3
    topology, vertices, currents = scenarios.reflecting()
    currents = list(currents)
    _, places = core.junction(scenarios.n_wide, scenarios.n_high, scenarios.wide, scenarios.high)
    middle = np.isclose(places[:, 1], 0.0, atol=1e-9)
    order = [np.sign(places[middle, 0][np.argmax(values)] - places[middle, 0][np.argmin(values)])
             for values in (np.asarray(currents[index].batch().kernel)[middle, 0] for index in (0, -1))]
    assert order[0] == -order[1]
