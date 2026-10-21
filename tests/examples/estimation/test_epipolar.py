"""Unit tests for epipolar geometry and two-view 3D reconstruction."""

from __future__ import annotations

import numpy as np

from examples.estimation.epipolar import render, scenarios


def test_reconstruct_from_noise_free_images():
    """Without noise, relative pose and world points are recovered exactly, up to the baseline scale."""
    landmarks, _, _, _, reconstructed, true_motor, est_motors = scenarios.epipolar(0.0)
    alignment, agreement, rms = scenarios.errors(landmarks, reconstructed[-1], true_motor, est_motors[-1])
    np.testing.assert_allclose(alignment.to_array(), 1.0, atol=1e-6)
    np.testing.assert_allclose(agreement.to_array(), 1.0, atol=1e-6)
    assert rms.to_array() < 1e-3


def test_search_animates():
    """The noisy scenario's search draws, one frame per state."""
    landmarks, _, _, rays, _, true_motor, est_motors = scenarios.epipolar(scenarios.NOISE)
    states = list(scenarios.easing(rays, est_motors, 1, 1))
    frames = render.animate_reconstruction(landmarks, np.array(scenarios.HOUSE_EDGES), true_motor, iter(states[-2:]))
    assert len(frames) == 2 and frames[0].shape == frames[1].shape
