"""Curvature of quadric surfaces matches the ellipsoid formula, stays invariant under rigid placement, and closes at the umbilics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.surface_curvature import core, render, scenarios

semi_axes = np.array([3.0, 2.0, 1.2])
origins = np.array([[10.0, 6.0, 4.0], [-7.0, 9.0, 3.0], [2.0, -8.0, -9.0]])


def sample_surface_points(surface: core.Quadric) -> core.Point:
    """Intersects rays aimed toward the origin from three exterior points (`[3] Point-`) with `surface`."""
    heading = core.mv.direction(-origins / np.linalg.norm(origins, axis=-1, keepdims=True))  # [3] Direction-
    points, _ = core.ray_intersections(surface, core.mv.point(origins), heading)        # [3] Point-, [3] Scalar+
    return points


def test_gaussian_curvature_matches_the_ellipsoid_formula():
    """Gaussian curvature matches the analytic ellipsoid formula at arbitrary surface points."""
    ellipsoid = core.central_quadric(1 / semi_axes**2)                                  # [] Plane+ <- Point-
    points = sample_surface_points(ellipsoid)                                           # [3] Point-
    curvatures = core.principal_curvatures(ellipsoid, points)                           # [3, 2] Scalar+
    xyz = render.coordinates(points)
    expected = 1 / np.prod(semi_axes) ** 2 / ((xyz**2 / semi_axes**4).sum(axis=-1)) ** 2
    assert ((curvatures[:, 0] * curvatures[:, 1] - expected).abs() < 1e-6 * expected).all()


def test_curvature_and_confocal_parameters_do_not_depend_on_placement():
    """Moving the quadric and its points by one rigid motor leaves both principal curvatures and confocal parameters unchanged."""
    ellipsoid = core.central_quadric(1 / semi_axes**2)                                  # [] Plane+ <- Point-
    points = sample_surface_points(ellipsoid)                                           # [3] Point-
    turn = ((core.mv.xy * 0.4 + core.mv.yz * 0.3) * 0.5).exp()                          # [] Rotor+
    shift = ((core.mv.xw * 1.5 - core.mv.zw * 0.7) * 0.5).exp()                         # [] Translator+
    motor = turn * shift                                                                # [] Motor+
    posed_ellipsoid = motor >> ellipsoid(motor << core.Point)                           # [] Plane+ <- Point-
    posed_points = motor >> points                                                      # [3] Point-
    assert ((core.principal_curvatures(posed_ellipsoid, posed_points) - core.principal_curvatures(ellipsoid, points)).abs() < 1e-6).all()
    assert ((core.confocal_parameters(posed_ellipsoid, posed_points) - core.confocal_parameters(ellipsoid, points)).abs() < 1e-6).all()


def test_the_net_of_curvature_lines_closes_up_at_the_umbilics():
    """At the ellipsoid's four umbilic points the two principal curvatures agree and the two confocal parameters meet."""
    ellipsoid, _ = scenarios.quadric_pair()                                             # [] Plane+ <- Point-, [] Plane+ <- Point-
    umbilics = scenarios.ellipsoid_umbilics()                                           # [4] Point-
    curvatures = core.principal_curvatures(ellipsoid, umbilics)                         # [4, 2] Scalar+
    parameters = core.confocal_parameters(ellipsoid, umbilics)                          # [4, 2] Scalar+
    assert ((curvatures[:, 0] - curvatures[:, 1]).abs() < 1e-8).all()
    assert ((parameters[:, 0] - parameters[:, 1]).abs() < 1e-6).all()


def test_figure_renders():
    """Each rendering stage produces a matplotlib figure on a coarse pixel grid."""
    scenes, curvatures, parameters = scenarios.curvature_scenes(16)
    heights = scenarios.heights
    for fig in (
        render.surfaces_figure(scenes, heights),
        render.gaussian_curvature_figure(scenes, curvatures, heights, 1.2),
        render.curvature_lines_figure(scenes, curvatures, parameters, heights, 1.2),
    ):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
