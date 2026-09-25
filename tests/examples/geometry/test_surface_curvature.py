"""Curvature of quadric surfaces: against the textbook formula, and independent of placement."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.surface_curvature import core, render, scenarios

SEMI_AXES = np.array([3.0, 2.0, 1.2])
TOWARD = np.array([[10.0, 6.0, 4.0], [-7.0, 9.0, 3.0], [2.0, -8.0, -9.0]])


def surface_points(surface: core.Quadric) -> core.Point:
    """Where rays toward the centre from a few directions meet the surface."""
    heading = core.direction(-TOWARD / np.linalg.norm(TOWARD, axis=-1, keepdims=True))
    points, _ = core.hit(surface, core.point(TOWARD), heading)
    return points


def test_gaussian_curvature_matches_the_ellipsoid_formula():
    """K = 1 / (a b c)^2 / (x^2/a^4 + y^2/b^4 + z^2/c^4)^2 on the ellipsoid with semi-axes a, b, c."""
    ellipsoid = core.quadric(1 / SEMI_AXES**2)
    points = surface_points(ellipsoid)
    k = core.principal(ellipsoid, points).to_array()
    xyz = render.euclidean(points)
    expected = 1 / np.prod(SEMI_AXES) ** 2 / ((xyz**2 / SEMI_AXES**4).sum(axis=-1)) ** 2
    np.testing.assert_allclose(k[:, 0] * k[:, 1], expected, rtol=1e-6)


def test_curvature_and_confocal_parameters_do_not_depend_on_placement():
    """Moving the quadric and its points by one motor changes neither the curvatures nor the
    confocal parameters: nothing refers to an origin."""
    ellipsoid = core.quadric(1 / SEMI_AXES**2)
    points = surface_points(ellipsoid)
    motor = ((core.mv.xy * 0.4 + core.mv.yz * 0.3) * 0.5).exp() * ((core.mv.xw * 1.5 - core.mv.zw * 0.7) * 0.5).exp()
    placed = motor >> ellipsoid(motor << core.Point)
    moved = motor >> points
    np.testing.assert_allclose(core.principal(placed, moved).to_array(), core.principal(ellipsoid, points).to_array(), rtol=1e-6)
    np.testing.assert_allclose(core.confocal(placed, moved).to_array(), core.confocal(ellipsoid, points).to_array(), rtol=1e-6)


def test_figure_renders():
    """The scenario passes its checks, and each stage draws."""
    scenes, curvatures, parameters = scenarios.curvature_lines(48)
    heights = (np.inf, 2.0)
    for draw in (render.draw_surfaces(scenes, heights), render.draw_gaussian_curvature(scenes, curvatures, heights, 1.2),
                 render.draw_curvature_lines(scenes, curvatures, parameters, heights, 1.2)):
        assert isinstance(draw, plt.Figure)
        plt.close(draw)
