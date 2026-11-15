"""Coarse-grid checks verify transport symmetry, positive dissipation, transient Seebeck-Peltier equality, and thermocouple rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.thermodynamics.thermocouple import core, render, scenarios

mv = core.mv
coarse_spacing = 0.1
n_steps = 2


def test_response_symmetry_and_dissipation() -> None:
    responses = core.response(scenarios.charges, scenarios.entropies, scenarios.mobilities)   # [2] Flux- <- Drive+
    np.testing.assert_allclose(responses.adjugate().kernel, responses.kernel, atol=1e-12)
    drives = stack((mv.xt, mv.xs, mv.yt, mv.ys))                                              # [4] Drive+
    assert ((drives & responses[:, None](drives)) > 0).all()


def test_seebeck_voltage_matches_peltier_cooling() -> None:
    topology, vertices, metals, first_foot, second_foot, junction_vertices = scenarios.hairpin(coarse_spacing)
    loss_map, storage_map = scenarios.operators(topology, vertices, metals, second_foot)
    states = [
        state
        for state, _ in zip(
            scenarios.states(topology, loss_map, storage_map, first_foot, second_foot, junction_vertices),
            range(n_steps),
        )
    ]
    voltage, cooling = scenarios.readings(topology, second_foot, junction_vertices, states)
    assert ((voltage - cooling).abs() < 1e-9).all()
    assert voltage[-1] > 0
    next(iter(render.animation(topology, vertices, metals, (first_foot, second_foot), 60, states[-1], states)))
    plt.close(render.hairpin_figure(topology, vertices, metals, (first_foot, second_foot), 60))
    plt.close(render.readings_figure(np.arange(n_steps), voltage, cooling))
