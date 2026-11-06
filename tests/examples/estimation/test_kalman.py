"""Pose filtering on the motor manifold in PGA2D preserves covariance symmetry."""

from __future__ import annotations

import numpy as np

from examples.estimation.kalman import core, render, scenarios


def test_covariance_stays_symmetric():
    """Prediction pushes the covariance through the step on both sides and the update subtracts a
    symmetric term, so the covariance stays its own adjugate."""
    mv = core.mv
    noise = core.covariance(0.1, 0.2)
    estimate, sigma = mv.rotor(), noise
    for turn in (0.3, -0.2, 0.5):
        step = ((mv.xy * turn - mv.wx) * 0.05).exp()
        estimate, sigma = estimate * step, (step << sigma(step >> core.Line)) + noise
        np.testing.assert_allclose(sigma.adjugate().kernel, sigma.kernel, atol=1e-14)
    estimate, sigma = core.update(estimate, sigma, ((mv.xy * 0.2 - mv.wx) * 0.5).exp(), core.covariance(0.3, 0.1))
    np.testing.assert_allclose(sigma.adjugate().kernel, sigma.kernel, atol=1e-14)


def test_one_reading_draws():
    truth, dead, estimates, covariances, readings = scenarios.tracking(n_readings=1, n_steps_per_reading=1)
    positions = estimates >> core.mv.w.dual()
    ellipsoids = core.position_ellipsoid(estimates, covariances, positions, scenarios.SIGMAS)
    next(render.animate_tracking(truth, dead, estimates, positions, ellipsoids, readings))
