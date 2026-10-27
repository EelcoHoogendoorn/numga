"""Test Garland–Heckbert quadric error metrics in PGA3D."""

from __future__ import annotations

import numpy as np
import pytest

from numga import stack
from examples.surfaces.qem import core
from examples.surfaces.qem.core import Point, mv


def test_plane_dyad_evaluates_to_squared_perpendicular_distance():
    """Evaluate a rank-1 plane dyad `plane * (plane & Point)` as the squared distance `(plane & point)**2`."""
    # The plane y = 0.5:
    plane = mv.y - mv.w * 0.5
    quadric = plane * (plane & Point)

    # Test points at distance +2, -2, and 0
    above = mv.point([1.0, 2.5, -0.5])
    below = mv.point([0.0, -1.5, 3.0])
    on_plane = mv.point([1.0, 0.5, -0.5])

    np.testing.assert_allclose((quadric(above) & above).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((quadric(below) & below).to_array(), 4.0, atol=1e-12)
    np.testing.assert_allclose((quadric(on_plane) & on_plane).to_array(), 0.0, atol=1e-12)


def test_edge_collapse_minimizes_joint_error():
    """Verify that edge collapse combines incident quadrics and finds the joint minimum."""
    from examples.surfaces.qem import scenarios

    vertices, faces, _, _, quadric_a, quadric_b, edge_quadric, optimum, _, _ = scenarios.qem()
    face_planes = (vertices[faces[:, 0]] & vertices[faces[:, 1]] & vertices[faces[:, 2]]).normalized()

    # Flanking planes pass through both edge endpoints a and b:
    flank_planes = face_planes[[0, 1]]
    vertex_a = vertices[0]
    vertex_b = vertices[1]
    np.testing.assert_allclose((flank_planes & vertex_a).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((flank_planes & vertex_b).to_array(), 0.0, atol=1e-12)

    # Edge quadric is the extensor sum quadric_a + quadric_b
    sample_point = mv.point([3, 4, 5])
    error_a = (quadric_a(sample_point) & sample_point).to_array()
    error_b = (quadric_b(sample_point) & sample_point).to_array()
    error_edge = (edge_quadric(sample_point) & sample_point).to_array()
    np.testing.assert_allclose(error_edge, error_a + error_b, atol=1e-12)

    # Optimal collapse point preserves the ridge line (y = 0) and is biased toward corner a
    np.testing.assert_allclose((mv.y & optimum).to_array(), 0.0, atol=1e-12)
    assert 0.0 < (mv.x & optimum) < (mv.x & vertex_a)

    # Joint error at optimum is lower than at either endpoint
    assert (edge_quadric(optimum) & optimum) < (edge_quadric(vertex_a) & vertex_a)
    assert (edge_quadric(optimum) & optimum) < (edge_quadric(vertex_b) & vertex_b)


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
    positions *= [1, 1.3, 0.7]
    positions += 0.03 * np.sin(np.arange(positions.size).reshape(positions.shape))
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

    for (old_vertices, old_faces), (new_vertices, new_faces) in zip(snapshots, snapshots[1:]):
        np.testing.assert_allclose((mv.w & new_vertices).to_array(), 1, atol=1e-12)
        assert np.isfinite(new_vertices.kernel).all()
        assert new_faces.min() == 0 and new_faces.max() == len(new_vertices) - 1
        assert len(np.unique(new_faces)) == len(new_vertices)
        assert len(np.unique(np.sort(new_faces, axis=1), axis=0)) == len(new_faces)
        assert np.all(np.diff(np.sort(new_faces, axis=1), axis=1) > 0)
        edges, incidence = np.unique(np.sort(np.stack((new_faces, np.roll(new_faces, -1, axis=1)), axis=-1)
                                             .reshape(-1, 2), axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(incidence, 2)
        assert len(new_vertices) - len(edges) + len(new_faces) == 2

        # Unmoved vertices identify the two merged clusters even after compact renumbering.
        equal = np.all(old_vertices.kernel[:, None, :] == new_vertices.kernel[None, :, :], axis=-1)
        removed_indices = np.flatnonzero(~equal.any(axis=1))
        added_indices = np.flatnonzero(~equal.any(axis=0))
        assert len(removed_indices) == 2 and len(added_indices) == 1
        merged_cluster = clusters[removed_indices[0]] | clusters[removed_indices[1]]
        next_clusters = [set() for _ in range(len(new_vertices))]
        old_to_new = np.empty(len(old_vertices), dtype=int)
        for old_index, new_index in zip(*np.nonzero(equal)):
            old_to_new[old_index] = new_index
            next_clusters[new_index] = clusters[old_index]
        old_to_new[removed_indices] = added_indices[0]
        next_clusters[added_indices[0]] = merged_cluster

        # Every surviving triangle keeps its winding and a nonzero area through the contraction.
        mapped_faces = old_to_new[old_faces]
        surviving_mask = np.all(np.diff(np.sort(mapped_faces, axis=1), axis=1) > 0, axis=1)
        assert set(map(tuple, np.sort(mapped_faces[surviving_mask], axis=1))) == set(map(tuple, np.sort(new_faces, axis=1)))
        old_planes = old_vertices[old_faces[surviving_mask, 0]] & old_vertices[old_faces[surviving_mask, 1]] & old_vertices[old_faces[surviving_mask, 2]]
        new_planes = new_vertices[mapped_faces[surviving_mask, 0]] & new_vertices[mapped_faces[surviving_mask, 1]] & new_vertices[mapped_faces[surviving_mask, 2]]
        assert np.all((old_planes | new_planes) > 0)

        # The replacement vertex minimizes all initial planes inherited by its cluster, including
        # multiplicities. Rebuilding quadrics from the current triangles loses this condition.
        inherited_faces = np.concatenate([np.flatnonzero(np.any(initial_faces == vertex, axis=1)) for vertex in merged_cluster])
        planes = initial_planes[inherited_faces]
        optimum = new_vertices[added_indices[0]]
        step = 1e-4
        directions = mv.point(np.eye(3)) - mv.point([0, 0, 0])
        forward = ((planes[:, None] & (optimum + step * directions)).squared()).sum(axis=0)
        backward = ((planes[:, None] & (optimum - step * directions)).squared()).sum(axis=0)
        np.testing.assert_allclose(((forward - backward) / (2 * step)).to_array(), 0, atol=1e-9)
        clusters = next_clusters


def test_scenario_renders():
    """Verify that the scenario's geometry renders as a figure."""
    import matplotlib.pyplot as plt
    from examples.surfaces.qem import render, scenarios

    figure = render.draw_qem(*scenarios.qem())
    assert isinstance(figure, plt.Figure)
