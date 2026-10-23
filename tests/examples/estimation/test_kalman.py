"""Pose filtering on the motor manifold in PGA2D."""

from __future__ import annotations

import numpy as np

from examples.estimation.kalman import core, render, scenarios


def test_filter_beats_dead_reckoning_and_animates():
    """Sparse noisy pose readings keep the filtered path far closer to the truth than the
    dead-reckoned one, and the animation draws."""
    truth, dead, estimates, covariances, readings = scenarios.tracking()
    dead_error, filtered_error = scenarios.errors(truth, dead, estimates)
    assert filtered_error.mean(axis=0).to_array() < 0.25 * dead_error.mean(axis=0).to_array()

    here, variances, axes = core.position_ellipse(estimates[:3], covariances[:3], core.mv.w.dual())
    frames = list(render.animate_tracking(truth[:3], dead[:3], estimates[:3], here, variances, axes, readings[:3]))
    assert len(frames) == 3 and all(frame.shape == frames[0].shape for frame in frames)


def test_covariance_stays_symmetric():
    """Prediction pushes the covariance through the step on both sides and the update subtracts a
    symmetric term, so the covariance stays its own adjugate."""
    mv = core.mv
    noise = core.covariance(0.1, 0.2)
    estimate, sigma = mv.rotor(), noise
    for turn in (0.3, -0.2, 0.5):
        estimate, sigma = core.predict(estimate, sigma, ((mv.xy * turn - mv.wx) * 0.05).exp(), noise)
        np.testing.assert_allclose(sigma.adjugate().kernel, sigma.kernel, atol=1e-14)
    estimate, sigma = core.update(estimate, sigma, ((mv.xy * 0.2 - mv.wx) * 0.5).exp(), core.covariance(0.3, 0.1))
    np.testing.assert_allclose(sigma.adjugate().kernel, sigma.kernel, atol=1e-14)
