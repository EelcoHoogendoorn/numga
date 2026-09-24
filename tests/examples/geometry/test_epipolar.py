"""Unit tests for epipolar geometry and two-view 3D reconstruction."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.epipolar import core, render, scenarios
from examples.geometry.epipolar.render import euclidean


def test_reconstruct_from_noise_free_images():
    """Without noise, relative pose and world points are recovered exactly, up to the baseline scale."""
    mv = core.mv
    landmarks = core.house_landmarks()
    origin = mv.zyx
    camera = (origin & core.Point) ^ (mv.z - mv.w)

    theta = np.radians(14.0)
    true_rot = (mv.zx * (theta * 0.5)).exp()
    true_trans = (mv.xw * 0.65 + mv.yw * 0.08 + mv.zw * 0.18) * 0.5
    true_motor = (true_trans.exp() * true_rot).normalized()

    rays_1 = (origin & camera(landmarks)).normalized()
    rays_2 = (origin & camera(true_motor << landmarks)).normalized()
    est_motor, reconstructed = core.reconstruct(rays_1, rays_2, true_trans.exp().normalized(), 10)

    # Rotated axis planes agree:
    axes = mv("x y z", np.eye(3))
    turned = (est_motor >> axes).normalized() | (true_motor >> axes).normalized()
    np.testing.assert_allclose(turned.to_array(), 1.0, atol=1e-6)

    # Camera 2's centre lies on the true baseline, and the points match after rescaling:
    c2_true = euclidean(true_motor >> origin)
    c2_est = euclidean(est_motor >> origin)
    scale = np.linalg.norm(c2_true) / np.linalg.norm(c2_est)
    np.testing.assert_allclose(c2_est * scale, c2_true, atol=1e-4)
    np.testing.assert_allclose(euclidean(reconstructed) * scale, euclidean(landmarks), atol=1e-2)


def test_scenario_renders():
    """The scenario passes its checks, and its geometry renders as a figure."""
    figure = render.draw_epipolar(*scenarios.epipolar())
    assert isinstance(figure, plt.Figure)
