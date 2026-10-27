"""Pose filtering on the motor manifold in PGA2D keeps the filtered trajectory close to the truth and preserves covariance symmetry."""

from __future__ import annotations

import numpy as np

from examples.estimation.kalman import core, render, scenarios


def test_filter_beats_dead_reckoning_and_animates():
    """Sparse noisy pose readings keep the filtered path far closer to the truth than the
    dead-reckoned one, and the animation draws."""
    origin = core.mv.w.dual()
    truth, dead, estimates, covariances, readings = scenarios.tracking()
    positions = estimates >> origin
    ellipses = core.position_ellipse(estimates, covariances, positions, scenarios.SIGMAS)
    true_positions = truth >> origin
    dead_error = (true_positions & (dead >> origin)).norm()
    filtered_error = (true_positions & positions).norm()
    assert filtered_error.mean(axis=0) < 0.25 * dead_error.mean(axis=0)

    frames = list(render.animate_tracking(truth[:3], dead[:3], estimates[:3], positions[:3], ellipses[:3], readings[:3]))
    assert len(frames) == 3 and all(frame.shape == frames[0].shape for frame in frames)


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
