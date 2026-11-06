"""Two bodies exchanging heat: the exchange for a step, and its first frame drawn."""

import numpy as np

from examples.thermodynamics.entropy_production import core, render, scenarios


def test_exchange_runs_and_draws():
    states = list(scenarios.exchange(2, 1))
    next(render.exchange(scenarios.entropy_of, scenarios.temperatures_of, core.transfer,
                         np.array([scenarios.capacity_a, scenarios.capacity_b]), iter(states)))
