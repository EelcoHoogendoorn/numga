"""Step the two-body heat exchange once and render its first frame."""

import numpy as np

from examples.thermodynamics.entropy_production import core, render, scenarios


def test_exchange_runs_and_draws():
    states = list(scenarios.exchange(2, 1))
    next(
        render.exchange(
            scenarios.entropy_of,
            scenarios.entropy_potential_of,
            scenarios.temperatures_of,
            core.transfer,
            np.array([scenarios.capacity_a, scenarios.capacity_b]),
            iter(states),
        )
    )
