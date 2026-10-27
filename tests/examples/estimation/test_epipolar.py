"""Unit tests for epipolar geometry and two-view 3D reconstruction."""

from __future__ import annotations

import numpy as np

from examples.estimation.epipolar import render, scenarios


def test_reconstruct_from_noise_free_images():
    """Without noise, relative pose and world points are recovered exactly, up to the baseline scale."""
    landmarks, _, _, _, reconstructed_landmarks, true_motor, estimated_motors = scenarios.epipolar(0.0)
    alignment, agreement, rms = scenarios.errors(landmarks, reconstructed_landmarks[-1], true_motor, estimated_motors[-1])
    np.testing.assert_allclose(alignment.kernel, 1.0, atol=1e-6)
    np.testing.assert_allclose(agreement.kernel, 1.0, atol=1e-6)
    assert rms < 1e-3


def test_search_animates():
    """The noisy scenario's search renders one frame per state."""
    landmarks, _, _, rays, _, true_motor, estimated_motors = scenarios.epipolar(scenarios.NOISE)
    states = list(scenarios.easing(rays, estimated_motors, 1, 1))
    frames = list(render.animate_reconstruction(landmarks, np.array(scenarios.HOUSE_EDGES), true_motor, iter(states[-2:])))
    assert len(frames) == 2 and frames[0].shape == frames[1].shape
