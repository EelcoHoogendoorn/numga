"""Curvature of quadric surfaces matches the ellipsoid formula, is invariant to rigid placement, and closes at the umbilics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.surface_curvature import core, render, scenarios

SEMI_AXES = np.array([3.0, 2.0, 1.2])
ORIGINS = np.array([[10.0, 6.0, 4.0], [-7.0, 9.0, 3.0], [2.0, -8.0, -9.0]])


def surface_points(surface: core.Quadric) -> core.Point:
    """Where rays toward the centre from three sample points meet the surface."""
    heading = core.mv.direction(-ORIGINS / np.linalg.norm(ORIGINS, axis=-1, keepdims=True))
    points, _ = core.hit(surface, core.mv.point(ORIGINS), heading)
    return points


def test_gaussian_curvature_matches_the_ellipsoid_formula():
    """K = 1 / (a b c)^2 / (x^2/a^4 + y^2/b^4 + z^2/c^4)^2 on the ellipsoid with semi-axes a, b, c."""
    ellipsoid = core.quadric(1 / SEMI_AXES**2)
    points = surface_points(ellipsoid)
    curvatures = core.principal(ellipsoid, points).to_array()
    xyz = render.coordinates(points)
    expected = 1 / np.prod(SEMI_AXES) ** 2 / ((xyz**2 / SEMI_AXES**4).sum(axis=-1)) ** 2
    np.testing.assert_allclose(curvatures[:, 0] * curvatures[:, 1], expected, rtol=1e-6)


def test_curvature_and_confocal_parameters_do_not_depend_on_placement():
    """Moving the quadric and its points by one motor changes neither the curvatures nor the
    confocal parameters: nothing refers to an origin."""
    ellipsoid = core.quadric(1 / SEMI_AXES**2)
    points = surface_points(ellipsoid)
    turn = ((core.mv.xy * 0.4 + core.mv.yz * 0.3) * 0.5).exp()
    shift = ((core.mv.xw * 1.5 - core.mv.zw * 0.7) * 0.5).exp()
    motor = turn * shift
    posed_ellipsoid = motor >> ellipsoid(motor << core.Point)
    posed_points = motor >> points
    np.testing.assert_allclose(core.principal(posed_ellipsoid, posed_points).kernel, core.principal(ellipsoid, points).kernel, rtol=1e-6)
    np.testing.assert_allclose(core.confocal(posed_ellipsoid, posed_points).kernel, core.confocal(ellipsoid, points).kernel, rtol=1e-6)


def test_the_net_of_curvature_lines_closes_up_at_the_umbilics():
    """At the ellipsoid's umbilics the two principal curvatures agree and the two confocal parameters meet."""
    ellipsoid, _ = scenarios.surfaces()
    curvatures, parameters = core.principal(ellipsoid, scenarios.umbilics()), core.confocal(ellipsoid, scenarios.umbilics())
    np.testing.assert_allclose(curvatures[:, 0].kernel, curvatures[:, 1].kernel, rtol=1e-8)
    np.testing.assert_allclose(parameters[:, 0].kernel, parameters[:, 1].kernel, rtol=1e-6)


def test_figure_renders():
    """Each stage renders a matplotlib figure."""
    scenes, curvatures, parameters = scenarios.curvature_lines(48)
    heights = scenarios.HEIGHTS
    for draw in (render.draw_surfaces(scenes, heights), render.draw_gaussian_curvature(scenes, curvatures, heights, 1.2),
                 render.draw_curvature_lines(scenes, curvatures, parameters, heights, 1.2)):
        assert isinstance(draw, plt.Figure)
        plt.close(draw)
