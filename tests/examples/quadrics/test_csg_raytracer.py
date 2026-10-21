"""Tests for Constructive Solid Geometry quadric raytracing with WebGPU / Metal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.csg_raytracer import core, render, scenarios


def test_hyperboloid_ruling_lines_lie_on_surface():
    """Points along the ruling lines p(theta) + t * v(theta) evaluate identically to zero on the quadric."""
    a, c = 0.55, 0.85
    throat, v_plus, v_minus = core.ruling_lines([a, a, c], 8)
    hyp = core.hyperboloid([a, a, c])

    for t in [-2.0, -0.5, 0.0, 1.2]:
        pts = throat + v_plus * t
        np.testing.assert_allclose(core.evaluate(hyp, pts).to_array(), 0.0, atol=1e-7)


def test_csg_scene_renders_figure_and_frames():
    """The GPU CSG raytracer renders a Figure and animation frames offscreen."""
    scene = scenarios.make_scene()
    fig = render.draw_scene(scene, scenarios.PRIMARY_CAM, scenarios.TARGET, 160, 160)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    cam_orbit = scenarios.orbit_cameras(4)
    anim_frames = render.frames(scene, cam_orbit, scenarios.TARGET, 120, 120)
    assert len(anim_frames) == 4
    assert anim_frames[0].shape == (120, 120, 3)
    assert anim_frames[0].dtype == np.uint8
