"""Coverage tests for least-squares fitting of projective primitives and its figure renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.fitting import render, scenarios
from examples.geometry.fitting.core import Line, Plane, Point, ctx, fitted_primitive, mv
from examples.geometry.fitting.render import euclidean_coordinates
from examples.geometry.fitting.scenarios import (
    line_segment,
    noisy_points,
    origin,
    planar_patch,
    point_cloud,
    pose,
    ray_bundle,
    segment_end_planes,
)


def projective_match(first, second, atol: float) -> bool:
    """Two nullary extensors represent the same projective element when their kernels agree up to scale and sign."""
    first_k, second_k = first.kernel, second.kernel
    first_k = first_k / np.abs(first_k).max()
    second_k = second_k / np.abs(second_k).max()
    return np.allclose(first_k, second_k, atol=atol) or np.allclose(first_k, -second_k, atol=atol)


def test_unit_forms_are_degenerate_on_the_free_coefficients():
    ranks = {}
    for name, Unknown in (("point", Point), ("line", Line), ("plane", Plane)):
        unit = Unknown.scalar_norm_squared()
        ranks[name] = np.linalg.matrix_rank(unit.materialize(ctx).kernel[0])
    assert ranks == {"point": 1, "line": 3, "plane": 3}


def test_exact_samples_recover_line_and_plane_exactly():
    assert projective_match(fitted_primitive(Line, pose >> line_segment(12, 2.0)), pose >> mv.xz, atol=1e-10)
    assert projective_match(
        fitted_primitive(Plane, pose >> planar_patch(16, 2.0, np.random.default_rng(1))),
        pose >> mv.z,
        atol=1e-10,
    )


def test_point_fit_is_the_centroid():
    rng = np.random.default_rng(2)
    points = noisy_points(pose >> point_cloud(30, 0.5, rng), 0.05, rng)
    centroid = fitted_primitive(Point, points)
    np.testing.assert_allclose(euclidean_coordinates(centroid), euclidean_coordinates(points).mean(axis=0), atol=1e-10)
    assert projective_match(centroid, points.sum(), atol=1e-10)


def test_line_fit_matches_principal_axis_and_is_a_line():
    rng = np.random.default_rng(3)
    points = noisy_points(pose >> line_segment(30, 2.0), 0.05, rng)
    line = fitted_primitive(Line, points)
    assert (line & line).abs() < 1e-10

    xyz = euclidean_coordinates(points)
    _, _, vt = np.linalg.svd(xyz - xyz.mean(axis=0))
    ends = euclidean_coordinates(line ^ (pose >> segment_end_planes(2.0)))
    axis = ends[1] - ends[0]
    axis = axis / np.linalg.norm(axis)
    assert np.isclose(abs(axis @ vt[0]), 1.0, atol=1e-6)


def test_plane_fit_matches_svd_normal_through_centroid():
    rng = np.random.default_rng(4)
    points = noisy_points(pose >> planar_patch(30, 2.0, rng), 0.05, rng)
    plane = fitted_primitive(Plane, points)
    xyz = euclidean_coordinates(points)
    _, _, vt = np.linalg.svd(xyz - xyz.mean(axis=0))
    normal = (stack((mv.x, mv.y, mv.z)) & plane.dual()).to_array()
    normal = normal / np.linalg.norm(normal)
    assert np.isclose(abs(normal @ vt[-1]), 1.0, atol=1e-6)
    assert (plane & mv.point(xyz.mean(axis=0))).abs() < 1e-10


def test_point_fitted_to_a_bundle_is_the_point_of_closest_approach():
    rng = np.random.default_rng(5)
    exact = pose >> ray_bundle(10, 0.0, rng)
    assert projective_match(fitted_primitive(Point, exact), pose >> origin, atol=1e-10)

    rays = pose >> ray_bundle(16, 0.05, rng)
    meet = fitted_primitive(Point, rays)
    directions = (stack((mv.x, mv.y, mv.z)) & (rays ^ mv.w)[..., None]).to_array()
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projector = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    normals = mv.x * directions[:, 0] + mv.y * directions[:, 1] + mv.z * directions[:, 2]
    on_line = euclidean_coordinates(rays ^ normals)
    system = projector.sum(axis=0)
    rhs = np.einsum("nij,nj->i", projector, on_line)
    np.testing.assert_allclose(euclidean_coordinates(meet), np.linalg.solve(system, rhs), atol=1e-8)


def test_fitting_figures_render():
    plt.close(render.point_fit_figure(*scenarios.point_to_points()))
    plt.close(render.line_fit_figure(*scenarios.line_to_points()))
    plt.close(render.plane_fit_figure(*scenarios.plane_to_points()))
    plt.close(render.bundle_fit_figure(*scenarios.point_to_lines()))
