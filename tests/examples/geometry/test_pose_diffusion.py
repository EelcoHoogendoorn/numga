"""A held pose under noise: the settled covariance zeroes the growth, and the scenario renders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.pose_diffusion import core, render, scenarios


def test_settled_covariance_zeroes_the_growth():
    """The map solved for stops the covariance's growth, in every setup."""
    for rate in scenarios.RATES.values():
        dynamics = core.drift(rate, scenarios.RELAXATION)
        noise = core.covariance(scenarios.KICKS)
        limit = core.settled(dynamics, noise)
        np.testing.assert_allclose(core.growth(dynamics, limit, noise).kernel, 0.0, atol=1e-10)


def test_clouds_render():
    """A short run passes its checks against the settled state and draws."""
    runs = {name: list(scenarios.diffuse(rate, scenarios.KICKS, 8.0, 0.02, 800, 100, 1)) for name, rate in scenarios.RATES.items()}
    figure = render.draw_settled({name: states[-1] for name, states in runs.items()}, 1.6)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
