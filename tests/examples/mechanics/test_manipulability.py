"""The arm's velocity and force ellipsoids are each other's polar in the unit sphere about the tip, and
a unit force lies on the force ellipsoid exactly when its joint torques have unit squared sum; the
figure and the sweep render."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.manipulability import core, render, scenarios


def test_the_ellipsoids_are_reciprocal_and_the_force_ellipsoid_holds_unit_torques():
    _, tip, axes = core.arm_axes((0.1, 0.7, 0.9, -0.3))
    velocity, force = core.velocity_ellipsoid(tip, axes), core.force_ellipsoid(tip, axes)
    sphere = core.unit_sphere(tip)
    np.testing.assert_allclose(force.kernel, sphere(velocity(sphere(core.Point))).kernel, rtol=1e-8, atol=1e-8)
    along = core.mv.direction([[0.0, 0.6, 0.8]])
    torques = (axes[:, None] & (tip & along)).to_array()
    held = tip + along
    np.testing.assert_allclose((held & force(held)).to_array(), 1 - (torques**2).sum(axis=0), atol=1e-8)


def test_figure_renders():
    poses = [scenarios.ellipsoids(angles, scenarios.VELOCITY_SCALE, scenarios.FORCE_SCALE)
             for angles in (scenarios.REACHING, scenarios.NEARLY_STRAIGHT)]
    figure = render.draw_ellipsoids(poses, scenarios.VIEW)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test_sweep_renders():
    frames = [render.frame(pose, scenarios.VIEW, scenarios.CENTRE, scenarios.EXTENT, 40) for pose in islice(scenarios.sweep(12), 3)]
    assert frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
