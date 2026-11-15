"""A six-dimensional rotation splits into its three planes by its spectrum and by its wedge powers, a rigid motion
splits into its placed planes, and the turned direction's orbit draws at the fewest samples."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.invariant_decomposition import core, render, scenarios


def test_the_rotation_and_the_motion_split_and_the_orbit_draws():
    bivector, start, motion = scenarios.samples(scenarios.seed)
    squares, parts, points, circles = scenarios.orbit_tangle(bivector, start, 2)
    wedge_squares, wedge_parts = core.wedge_decomposition(bivector)
    planes = core.placed_planes(motion)
    assert ((wedge_squares - squares).abs() < 1e-11).all()
    assert ((wedge_parts - parts).scalar_norm_squared() < 1e-22).all()
    np.testing.assert_allclose((planes ^ planes).kernel, 0.0, atol=1e-11)
    np.testing.assert_allclose((motion - planes.sum(axis=-1)).kernel, 0.0, atol=1e-10)
    plt.close(render.orbit_figure(points, circles, parts, 1))
    list(render.orbit_animation(points, circles, parts, 1))
