"""Pose filtering on the motor manifold in PGA2D preserves covariance symmetry and renders cleanly."""

from __future__ import annotations

from functools import partial

import numpy as np

from examples.estimation.kalman import core, render, scenarios


def test_covariance_stays_symmetric():
    """Prediction sandwiches the covariance map through each step and `filter_update` subtracts a symmetric gain term."""
    mv = core.mv
    frame = mv.bivector(np.eye(3)).dual()
    noise = core.body_covariance(0.1, 0.2)
    estimate, sigma = mv.rotor(), noise
    for turn in (0.3, -0.2, 0.5):
        step = ((mv.xy * turn - mv.wx) * 0.05).exp()
        estimate, sigma = estimate * step, (step << sigma(step >> core.Line)) + noise
        skew = sigma.adjugate() - sigma
        assert bool((frame[:, None] & skew(frame[None, :])).abs().max() < 1e-13)
    reading = ((mv.xy * 0.2 - mv.wx) * 0.5).exp()
    estimate, sigma = core.filter_update(estimate, sigma, reading, core.body_covariance(0.3, 0.1))
    skew = sigma.adjugate() - sigma
    assert bool((frame[:, None] & skew(frame[None, :])).abs().max() < 1e-13)


def test_one_reading_draws():
    """One commanded step and reading render a single tracking frame."""
    increments = scenarios.commanded_increments(1, 1)
    motion_noise = core.body_covariance(*(np.array(scenarios.motion_std) * np.sqrt(scenarios.dt)))
    measurement_noise = core.body_covariance(*scenarios.measurement_std)
    states = scenarios.drive_states(
        core.mv.rotor(), increments, motion_noise, measurement_noise, np.random.default_rng(scenarios.seed)
    )
    ellipsoid_of = partial(core.position_ellipsoid, sigmas=scenarios.sigmas)
    next(render.tracking_animation(increments, ellipsoid_of, states))

