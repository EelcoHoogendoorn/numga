"""Unit tests for least-squares fitting of PGA3D primitives."""

from __future__ import annotations

import numpy as np


from examples.geometry.fitting import (
    fit,
    ga,
    ctx,
    Line,
    Plane,
    Point,
    bundle,
    cloud,
    euclidean,
    jitter,
    line_caps,
    mv,
    patch,
    point,
    segment,
)

POSE = (mv.xw * 0.4 + mv.yw * -0.3 + mv.zw * 0.6).exp() * (mv.yz * 0.3).exp() * (mv.xy * 0.5).exp()


def same_element(a, b, atol: float) -> bool:
    """Whether two nullary extensors agree as projective elements, up to scale and sign."""
    ka, kb = a.kernel, b.cast(a.gatype.output_subspace).kernel
    ka, kb = ka / np.abs(ka).max(), kb / np.abs(kb).max()
    return np.allclose(ka, kb, atol=atol) or np.allclose(ka, -kb, atol=atol)

def test_unit_forms_are_degenerate_on_the_free_coefficients():
    ranks = {}
    for name, Unknown in (("point", Point), ("line", Line), ("plane", Plane)):
        unit = ga.operator.reverse(Unknown.output_subspace) | Unknown
        ranks[name] = np.linalg.matrix_rank(np.asarray(ctx.lower(unit).kernel).squeeze())
    assert ranks == {"point": 1, "line": 3, "plane": 3}


def test_exact_samples_recover_line_and_plane_exactly():
    assert same_element(fit(Line, POSE >> segment(50, 2.0)), POSE >> mv.xz, atol=1e-10)
    assert same_element(fit(Plane, POSE >> patch(50, 2.0, np.random.default_rng(1))), POSE >> mv.z, atol=1e-10)


def test_point_fit_is_the_centroid():
    rng = np.random.default_rng(2)
    points = jitter(POSE >> cloud(300, 0.5, rng), 0.05, rng)
    centroid = fit(Point, points)
    np.testing.assert_allclose(euclidean(centroid), euclidean(points).mean(axis=0), atol=1e-10)
    assert same_element(centroid, points.sum(), atol=1e-10)


def test_line_fit_matches_principal_axis_and_is_a_line():
    rng = np.random.default_rng(3)
    points = jitter(POSE >> segment(200, 2.0), 0.05, rng)
    line = fit(Line, points)
    scale = float(np.abs(line.kernel).max()) ** 2
    np.testing.assert_allclose(line.wedge(line).kernel / scale, 0.0, atol=1e-14)

    xyz = euclidean(points)
    _, _, vt = np.linalg.svd(xyz - xyz.mean(axis=0))
    ends = euclidean(line.wedge(POSE >> line_caps(2.0)))
    d = ends[1] - ends[0]
    d = d / np.linalg.norm(d)
    assert np.isclose(abs(d @ vt[0]), 1.0, atol=1e-6)


def test_plane_fit_matches_svd_normal_through_centroid():
    rng = np.random.default_rng(4)
    points = jitter(POSE >> patch(300, 2.0, rng), 0.05, rng)
    plane = fit(Plane, points)
    xyz = euclidean(points)
    _, _, vt = np.linalg.svd(xyz - xyz.mean(axis=0))
    n = plane.cast(Plane.output_subspace).kernel[:3]
    n = n / np.linalg.norm(n)
    assert np.isclose(abs(n @ vt[-1]), 1.0, atol=1e-6)
    np.testing.assert_allclose(point(xyz.mean(axis=0)).regressive(plane).kernel, 0.0, atol=1e-10)


def test_point_fitted_to_a_bundle_is_the_point_of_closest_approach():
    rng = np.random.default_rng(5)
    exact = POSE >> bundle(20, 0.0, rng)
    assert same_element(fit(Point, exact), POSE >> mv.zyx, atol=1e-10)

    rays = POSE >> bundle(40, 0.05, rng)
    meet = fit(Point, rays)
    # Compare with the classical closed form: the least-squares intersection of lines.
    directions = rays.wedge(mv.w).cast(Point.output_subspace).kernel[:, :3]
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    projector = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    # A point on each line: meet the line with the plane through the origin perpendicular to it.
    normals = mv.vector(np.concatenate([directions, np.zeros((40, 1))], axis=-1))
    on_line = euclidean(rays.wedge(normals))
    A = projector.sum(axis=0)
    b = np.einsum("nij,nj->i", projector, on_line)
    np.testing.assert_allclose(euclidean(meet), np.linalg.solve(A, b), atol=1e-8)


def test_tutorial_runs_and_saves(tmp_path, monkeypatch):
    from examples.geometry import fitting
    monkeypatch.setattr(fitting, "PLOT_DIR", tmp_path)
    fitting.point_to_points()
    fitting.line_to_points()
    fitting.plane_to_points()
    fitting.point_to_lines()
    assert (tmp_path / "fitting_point_to_points.png").exists()
    assert (tmp_path / "fitting_line_to_points.png").exists()
    assert (tmp_path / "fitting_plane_to_points.png").exists()
    assert (tmp_path / "fitting_point_to_lines.png").exists()
