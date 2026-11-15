"""Station keeping under white-noise gusts solves the steady-state Lyapunov equation and renders cleanly."""

from __future__ import annotations

from functools import partial

import numpy as np

from examples.estimation.pose_diffusion import core, render, scenarios


def test_settled_covariance_zeroes_the_growth():
    """The steady-state covariance balances drift and white-noise injection in every scene."""
    frame = core.mv.bivector(np.eye(3)).dual()
    noise = core.gust_covariance(scenarios.gusts)
    for rate in scenarios.turning_rates.values():
        dynamics = core.Twist.commutator(rate) - scenarios.relaxation_gain * core.Twist
        limit = core.steady_covariance(dynamics, noise)
        growth = dynamics(limit) + limit(dynamics.adjugate()) + noise
        assert (frame[:, None] & growth(frame[None, :])).abs().max() < 1e-10


def test_one_step_of_the_fleets_draws():
    """One diffusion step across both scenes renders an animation frame, a settled still, and an envelope plot."""
    streams = [
        scenarios.scene_stream(rate, scenarios.gusts, scenarios.dt, 2, 1, 4, scenarios.seed)
        for rate in scenarios.turning_rates.values()
    ]
    ellipsoid_of = partial(core.position_ellipsoid, sigmas=scenarios.sigmas)
    paired = list(zip(*streams))
    next(render.clouds_animation(scenarios.extent, ellipsoid_of, iter(paired)))
    render.settled_figure(list(paired[-1]), scenarios.extent, ellipsoid_of)
    gains = np.array([0.05, 0.5])
    controllers = -core.mv.scalar(gains[:, None]) * core.Twist
    spreads = core.position_spread(core.steady_covariance(controllers, core.gust_covariance(scenarios.gusts)))
    render.envelope_figure(gains, spreads, 2.0)

