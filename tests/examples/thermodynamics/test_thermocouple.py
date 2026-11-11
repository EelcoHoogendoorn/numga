"""A thermocouple: a metal's response its own adjugate and its dissipation positive; a coarse hairpin heated and
driven by a current over a few steps, the heat's voltage and the current's cooling of the junction equal; and its
pictures drawn."""

import numpy as np

from numga import stack
from examples.thermodynamics.thermocouple import core, render, scenarios

mv = core.mv
# A hairpin of few places, and a few of the scenario's steps.
coarse_spacing = 0.1
n_steps = 3


def test_response_is_its_own_adjugate_and_dissipates():
    responses = core.response(scenarios.charges, scenarios.entropies, scenarios.mobilities)   # [metals] Flux- <- Drive+
    np.testing.assert_allclose(responses.adjugate().kernel, responses.kernel, atol=1e-12)
    drives = stack((mv.xt, mv.xs, mv.yt, mv.ys))                             # [drives] Drive+
    assert ((drives & responses[:, None](drives)).to_array() > 0).all()


def test_heat_gives_the_voltage_that_current_gives_cooling():
    topology, vertices, metals, high, low, heated = scenarios.hairpin(coarse_spacing)
    losing, storing = scenarios.operators(topology, vertices, metals, low)
    states = [state for state, _ in zip(scenarios.states(topology, losing, storing, high, low, heated),
                                        range(n_steps))]
    voltage, cooling = scenarios.readings(topology, low, heated, states)
    np.testing.assert_allclose(voltage.to_array(), cooling.to_array(), atol=1e-9)
    assert voltage.to_array()[-1] > 0
    next(iter(render.animate(topology, vertices, metals, (high, low), 60, states[-1], states)))
    render.draw_hairpin(topology, vertices, metals, (high, low), 60)
    render.draw_readings(np.arange(n_steps), voltage, cooling)
