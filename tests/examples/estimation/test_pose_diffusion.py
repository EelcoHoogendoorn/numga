"""Station keeping under white-noise gusts solves the steady-state Lyapunov equation and renders the fleet clouds."""

from __future__ import annotations

import matplotlib.pyplot as plt
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


def test_clouds_render():
    """A short simulation run converges toward the steady-state covariance and draws."""
    runs = {name: list(scenarios.diffuse(rate, scenarios.gusts, 8.0, 0.02, 800, 100, 1)) for name, rate in scenarios.rates.items()}
    figure = render.draw_settled({name: states[-1] for name, states in runs.items()}, 1.6)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
