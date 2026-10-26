"""Unit tests for Garland–Heckbert QEM in PGA3D."""

from __future__ import annotations

import numpy as np
import pytest

from numga import stack
from examples.surfaces.qem import core
from examples.surfaces.qem.core import Point, mv


def test_plane_dyad_evaluates_to_squared_perpendicular_distance():
    """A rank-1 plane dyad P * (P & Point) evaluates to squared distance (P & X)^2."""
    # The plane y = 0.5:
    p_on_plane = np.array([1.0, 0.5, -0.5])
    p = mv.y - mv.w * 0.5

    # Build dyad
    q = p * (p & Point)

    # Test points at distance +2, -2, and 0
    q1 = mv.point(np.array([1.0, 2.5, -0.5]))
    q2 = mv.point(np.array([0.0, -1.5, 3.0]))
    q_on = mv.point(p_on_plane)

    np.testing.assert_allclose((q(q1) & q1).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q2) & q2).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((q(q_on) & q_on).to_array(), 0.0, atol=1e-12)


def test_edge_collapse_minimizes_joint_error():
    """Edge collapse combines incident quadrics and finds the joint minimum."""
    from examples.surfaces.qem import scenarios

    verts, faces, _, _, qa, qb, q_edge, v_edge, _, _ = scenarios.qem()
    face_planes = (verts[faces[:, 0]] & verts[faces[:, 1]] & verts[faces[:, 2]]).normalized()

    # Flanking planes pass through both edge endpoints a and b:
    flank_planes = face_planes[[0, 1]]
    pa = verts[0]
    pb = verts[1]
    np.testing.assert_allclose((flank_planes & pa).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((flank_planes & pb).to_array(), 0.0, atol=1e-12)

    # Edge quadric is the extensor sum qa + qb
    test_pt = mv.point(np.array([3.0, 4.0, 5.0]))
    ea = (qa(test_pt) & test_pt).to_array()
    eb = (qb(test_pt) & test_pt).to_array()
    e_edge = (q_edge(test_pt) & test_pt).to_array()
    np.testing.assert_allclose(e_edge, ea + eb, atol=1e-12)

    # Optimal collapse point preserves the ridge line (y = 0) and is biased toward corner a
    np.testing.assert_allclose((mv.y & v_edge).to_array(), 0.0, atol=1e-12)
    assert 0.0 < (mv.x & v_edge).to_array() < (mv.x & pa).to_array()

    # Joint error at v_edge is lower than at either endpoint
    e_opt = (q_edge(v_edge) & v_edge).to_array()
    e_at_a = (q_edge(pa) & pa).to_array()
    e_at_b = (q_edge(pb) & pb).to_array()
    assert e_opt < e_at_a
    assert e_opt < e_at_b


@pytest.mark.parametrize("height", [0.0, 1.0])
def test_planar_collapse_preserves_affine_weight_and_unconstrained_position(height):
    plane = mv.z - height * mv.w
    midpoint = mv.point([2.0, -3.0, height + 1.4])
    quadric = plane * (plane & Point)
    optimum = core.collapse_position(quadric, midpoint)

    np.testing.assert_allclose((mv.w & optimum).to_array(), 1, atol=1e-12)
    np.testing.assert_allclose((optimum - mv.point([2.0, -3.0, height])).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((plane & optimum).to_array(), 0, atol=1e-12)


def test_crease_collapse_preserves_position_along_the_free_direction():
    planes = stack((mv.x - 2 * mv.w, mv.z - mv.w))
    quadric = (planes * (planes & Point)).sum(axis=0)
    optimum = core.collapse_position(quadric, mv.point([4.0, -3.0, 5.0]))

    np.testing.assert_allclose((mv.w & optimum).to_array(), 1, atol=1e-12)
    np.testing.assert_allclose((optimum - mv.point([2.0, -3.0, 1.0])).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((planes & optimum).to_array(), 0, atol=1e-12)


def icosahedron():
    golden = (1 + np.sqrt(5)) / 2
    positions = np.array([
        [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
        [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
        [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
    ])
    # An asymmetric convex surface avoids equivalent edge choices in this small fixture.
    positions *= [1, 1.3, .7]
    positions += .03 * np.sin(np.arange(positions.size).reshape(positions.shape))
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    return mv.point(positions), faces


def test_repeated_collapses_preserve_closed_mesh_and_original_plane_error():
    vertices, faces = icosahedron()
    initial_faces = faces.copy()
    initial_planes = (vertices[faces[:, 0]] & vertices[faces[:, 1]] & vertices[faces[:, 2]]).normalized()
    clusters = [{index} for index in range(len(vertices))]
    targets = (18, 16, 14, 12)
    snapshots = list(core.simplify(vertices, faces, targets))
    assert [len(faces) for _, faces in snapshots] == [20, *targets]

    for (before, old_faces), (after, new_faces) in zip(snapshots, snapshots[1:]):
        np.testing.assert_allclose((mv.w & after).to_array(), 1, atol=1e-12)
        assert np.isfinite(after.kernel).all()
        assert new_faces.min() == 0 and new_faces.max() == len(after) - 1
        assert len(np.unique(new_faces)) == len(after)
        assert len(np.unique(np.sort(new_faces, axis=1), axis=0)) == len(new_faces)
        assert np.all(np.diff(np.sort(new_faces, axis=1), axis=1) > 0)
        edges, incidence = np.unique(np.sort(np.stack((new_faces, np.roll(new_faces, -1, axis=1)), axis=-1)
                                            .reshape(-1, 2), axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(incidence, 2)
        assert len(after) - len(edges) + len(new_faces) == 2

        # Unmoved vertices identify the two merged clusters even after compact renumbering.
        equal = np.all(before.kernel[:, None, :] == after.kernel[None, :, :], axis=-1)
        removed = np.flatnonzero(~equal.any(axis=1))
        added = np.flatnonzero(~equal.any(axis=0))
        assert len(removed) == 2 and len(added) == 1
        merged = clusters[removed[0]] | clusters[removed[1]]
        next_clusters = [set() for _ in range(len(after))]
        old_to_new = np.empty(len(before), dtype=int)
        for old_index, new_index in zip(*np.nonzero(equal)):
            old_to_new[old_index] = new_index
            next_clusters[new_index] = clusters[old_index]
        old_to_new[removed] = added[0]
        next_clusters[added[0]] = merged

        # Every surviving triangle keeps its winding and a nonzero area through the contraction.
        mapped = old_to_new[old_faces]
        survives = np.all(np.diff(np.sort(mapped, axis=1), axis=1) > 0, axis=1)
        assert set(map(tuple, np.sort(mapped[survives], axis=1))) == set(map(tuple, np.sort(new_faces, axis=1)))
        old_planes = before[old_faces[survives, 0]] & before[old_faces[survives, 1]] & before[old_faces[survives, 2]]
        new_planes = after[mapped[survives, 0]] & after[mapped[survives, 1]] & after[mapped[survives, 2]]
        assert np.all((old_planes | new_planes).to_array() > 0)

        # The new vertex minimizes all original planes inherited by its cluster, including
        # multiplicities. Rebuilding quadrics from the current triangles loses this condition.
        inherited = np.concatenate([np.flatnonzero(np.any(initial_faces == vertex, axis=1)) for vertex in merged])
        planes = initial_planes[inherited]
        optimum = after[added[0]]
        step = 1e-4
        directions = mv.point(np.eye(3)) - mv.point([0, 0, 0])
        forward = ((planes[:, None] & (optimum + step * directions)).squared()).sum(axis=0)
        backward = ((planes[:, None] & (optimum - step * directions)).squared()).sum(axis=0)
        np.testing.assert_allclose(((forward - backward) / (2 * step)).to_array(), 0, atol=1e-9)
        clusters = next_clusters


def test_scenario_renders():
    """The scenario's geometry renders as a figure."""
    import matplotlib.pyplot as plt
    from examples.surfaces.qem import render, scenarios

    figure = render.draw_qem(*scenarios.qem())
    assert isinstance(figure, plt.Figure)
