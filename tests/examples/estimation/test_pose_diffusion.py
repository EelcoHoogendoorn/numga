"""Station keeping under white-noise gusts solves the steady-state Lyapunov equation."""

from __future__ import annotations

import numpy as np

from examples.estimation.pose_diffusion import core, render, scenarios


def test_settled_covariance_zeroes_the_growth():
    """The steady-state covariance solved for stops the covariance's growth in every scene."""
    noise = core.covariance(scenarios.gusts)
    for rate in scenarios.rates.values():
        dynamics = core.Twist.commutator(rate) - scenarios.RELAXATION * core.Twist
        limit = core.settled(dynamics, noise)
        growth = dynamics(limit) + limit(dynamics.adjugate()) + noise
        np.testing.assert_allclose(growth.kernel, 0.0, atol=1e-10)


def test_one_step_of_the_fleets_draws():
    runs = {name: list(scenarios.diffuse(rate, scenarios.gusts, scenarios.DT, scenarios.DT, 4, 1, scenarios.SEED))
            for name, rate in scenarios.rates.items()}
    next(render.animate_clouds(runs, scenarios.EXTENT))
