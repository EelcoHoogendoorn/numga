"""Unit tests for Garland–Heckbert QEM in PGA3D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.qem import core, render, scenarios
from examples.geometry.qem.core import Point, mv, point
from examples.geometry.qem.render import euclidean


def test_plane_dyad_evaluates_to_squared_perpendicular_distance():
    """A rank-1 plane dyad P * (P & Point) evaluates to squared distance (P & X)^2."""
    # The plane y = 0.5:
    p_on_plane = np.array([1.0, 0.5, -0.5])
    p = mv.y - mv.w * 0.5

    # Build dyad
    q = p * (p & Point)

    # Test points at distance +2, -2, and 0
    q1 = point(np.array([1.0, 2.5, -0.5]))
    q2 = point(np.array([0.0, -1.5, 3.0]))
    q_on = point(p_on_plane)

    np.testing.assert_allclose((q(q1) & q1).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q2) & q2).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q_on) & q_on).to_array(), 0.0, atol=1e-12)


def test_edge_collapse_minimizes_joint_error():
    """Edge collapse combines incident quadrics and finds the joint minimum."""
    verts, faces = core.ridge_patch()
    coords = euclidean(verts)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_planes = (v0 & v1 & v2).normalized()

    # Flanking planes pass through both edge endpoints a and b:
    flank_planes = face_planes[[0, 1]]
    pa = verts[0]
    pb = verts[1]
    np.testing.assert_allclose((flank_planes & pa).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((flank_planes & pb).to_array(), 0.0, atol=1e-12)

    planes_a = face_planes[[0, 1, 2, 3]]
    planes_b = face_planes[[0, 1, 4, 5]]

    qa, qb, q_edge, v_edge = core.edge_collapse(planes_a, planes_b)

    # Edge quadric is the extensor sum qa + qb
    test_pt = point(np.array([3.0, 4.0, 5.0]))
    ea = (qa(test_pt) & test_pt).to_array()
    eb = (qb(test_pt) & test_pt).to_array()
    e_edge = (q_edge(test_pt) & test_pt).to_array()
    np.testing.assert_allclose(e_edge, ea + eb, atol=1e-12)

    # Optimal collapse point preserves the ridge line (y = 0) and is biased toward corner a
    v_opt_xyz = euclidean(v_edge)
    np.testing.assert_allclose(v_opt_xyz[1], 0.0, atol=1e-12)
    assert 0.0 < v_opt_xyz[0] < coords[0, 0]

    # Joint error at v_edge is lower than at either endpoint
    e_opt = (q_edge(v_edge) & v_edge).to_array()
    e_at_a = (q_edge(pa) & pa).to_array()
    e_at_b = (q_edge(pb) & pb).to_array()
    assert e_opt < e_at_a
    assert e_opt < e_at_b


def test_scenario_renders():
    """The scenario passes its checks, and its geometry renders as a figure."""
    figure = render.draw_qem(*scenarios.qem())
    assert isinstance(figure, plt.Figure)
