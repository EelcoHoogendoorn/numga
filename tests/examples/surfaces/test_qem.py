"""Test Garland–Heckbert quadric error metrics in PGA3D and rendering of the ridge patch and bunny simplification."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from examples.surfaces.qem import core, render, scenarios
from examples.surfaces.qem.core import Point, mv


def test_plane_dyad_evaluates_to_squared_perpendicular_distance():
    """A rank-1 plane dyad `plane * (plane & Point)` evaluates to `(plane & point)**2`."""
    plane = mv.y - mv.w * 0.5                                                   # [] Plane+
    quadric = plane * (plane & Point)                                           # [] Plane+ <- Point-

    above = mv.point([1.0, 2.5, -0.5])                                          # [] Point-
    below = mv.point([0.0, -1.5, 3.0])                                          # [] Point-
    on_plane = mv.point([1.0, 0.5, -0.5])                                       # [] Point-

    assert ((quadric(above) & above) - 4).abs() < 1e-12
    assert ((quadric(below) & below) - 4).abs() < 1e-12
    assert (quadric(on_plane) & on_plane).abs() < 1e-12


def test_edge_collapse_minimizes_joint_error():
    """Edge contraction sums incident quadrics and finds the joint minimum along the ridge."""
    ridge, vertices, _, _, quadric_a, quadric_b, edge_quadric, optimum, _, _ = scenarios.qem()
    corners = ridge.incidence[2, 0] * vertices                                  # [3] Point-[F]
    face_planes = (corners[0] & corners[1] & corners[2]).normalized().batch()   # [F] Plane-

    flank_planes = face_planes[[0, 1]]                                          # [2] Plane-
    vertex_a, vertex_b = (ridge.reading(scenarios.ridge_edge, 0) * vertices).batch()    # [] Point-, Point-
    assert ((flank_planes & vertex_a).abs() < 1e-12).all()
    assert ((flank_planes & vertex_b).abs() < 1e-12).all()

    sample_point = mv.point([3, 4, 5])                                          # [] Point-
    error_sum = (quadric_a(sample_point) & sample_point) + (quadric_b(sample_point) & sample_point)
    assert ((edge_quadric(sample_point) & sample_point) - error_sum).abs() < 1e-12

    assert (mv.y & optimum).abs() < 1e-12
    assert 0 < (mv.x & optimum) < (mv.x & vertex_a)
    assert (edge_quadric(optimum) & optimum) < (edge_quadric(vertex_a) & vertex_a)
    assert (edge_quadric(optimum) & optimum) < (edge_quadric(vertex_b) & vertex_b)


@pytest.mark.parametrize("height", [0.0, 1.0])
def test_planar_collapse_preserves_affine_weight_and_unconstrained_position(height):
    plane = mv.z - height * mv.w                                                # [] Plane+
    midpoint = mv.point([2.0, -3.0, height + 1.4])                              # [] Point-
    quadric = plane * (plane & Point)                                           # [] Plane+ <- Point-
    optimum = core.collapse_position(quadric, midpoint)                         # [] Point-

    assert ((mv.w & optimum) - 1).abs() < 1e-12
    assert (optimum - mv.point([2.0, -3.0, height])).antinorm() < 1e-12
    assert (plane & optimum).abs() < 1e-12


def test_crease_collapse_preserves_position_along_the_free_direction():
    planes = stack((mv.x - 2 * mv.w, mv.z - mv.w))                              # [2] Plane+
    quadric = (planes * (planes & Point)).sum(axis=0)                           # [] Plane+ <- Point-
    optimum = core.collapse_position(quadric, mv.point([4.0, -3.0, 5.0]))       # [] Point-

    assert ((mv.w & optimum) - 1).abs() < 1e-12
    assert (optimum - mv.point([2.0, -3.0, 1.0])).antinorm() < 1e-12
    assert ((planes & optimum).abs() < 1e-12).all()


def icosahedron():
    golden = (1 + np.sqrt(5)) / 2
    positions = np.array([
        [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
        [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
        [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
    ])
    # Perturb the convex polyhedron slightly so edge costs are distinct across contractions.
    positions *= [1, 1.3, 0.7]
    positions += 0.03 * np.sin(np.arange(positions.size).reshape(positions.shape))
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    return core.Topology(len(positions), faces), mv.point(positions).field()


def test_repeated_collapses_preserve_closed_surface_and_initial_plane_error():
    topology, vertices = icosahedron()
    V, E, F = topology.n_elements
    initial_corners = topology.incidence[2, 0] * vertices                       # [3] Point-[F]
    initial_planes = (initial_corners[0] & initial_corners[1] & initial_corners[2]).normalized()    # Plane-[F]
    initial_quadrics = initial_planes * (initial_planes & Point)                # (Plane+ <- Point-)[F]
    clusters = [{index} for index in range(V)]
    targets = np.array([18, 16, 14, 12])
    snapshots = list(core.simplify(topology, vertices, targets))
    assert [step_topology.n_elements[2] for step_topology, _ in snapshots] == [F, *targets]

    for (old_topology, old_vertices), (new_topology, new_vertices) in zip(snapshots, snapshots[1:]):
        V_new, E_new, F_new = new_topology.n_elements
        old_faces, new_faces = old_topology.elements[2], new_topology.elements[2]
        old_batch, new_batch = old_vertices.batch(), new_vertices.batch()
        assert (((mv.w & new_batch) - 1).abs() < 1e-12).all()
        assert np.isfinite(new_batch.kernel).all()
        assert new_topology.is_closed and new_topology.is_manifold and new_topology.euler_characteristic == 2

        equal = np.all(old_batch.kernel[:, None, :] == new_batch.kernel[None, :, :], axis=-1)
        removed_indices = np.flatnonzero(~equal.any(axis=1))
        added_indices = np.flatnonzero(~equal.any(axis=0))
        assert len(removed_indices) == 2 and len(added_indices) == 1
        merged_cluster = clusters[removed_indices[0]] | clusters[removed_indices[1]]
        next_clusters = [set() for _ in range(V_new)]
        old_to_new = np.empty(len(old_batch), dtype=int)
        for old_index, new_index in zip(*np.nonzero(equal)):
            old_to_new[old_index] = new_index
            next_clusters[new_index] = clusters[old_index]
        old_to_new[removed_indices] = added_indices[0]
        next_clusters[added_indices[0]] = merged_cluster

        mapped_faces = old_to_new[old_faces]
        surviving_mask = np.all(np.diff(np.sort(mapped_faces, axis=1), axis=1) > 0, axis=1)
        assert set(map(tuple, np.sort(mapped_faces[surviving_mask], axis=1))) == set(map(tuple, np.sort(new_faces, axis=1)))
        old_corners = old_topology.subset(surviving_mask, -1).incidence[2, 0] * old_vertices
        new_corners = core.Topology(V_new, mapped_faces[surviving_mask]).incidence[2, 0] * new_vertices
        old_planes = old_corners[0] & old_corners[1] & old_corners[2]
        new_planes = new_corners[0] & new_corners[1] & new_corners[2]
        assert ((old_planes | new_planes) > 0).all()

        # The replacement vertex minimizes the initial face quadrics inherited with multiplicity by its vertex cluster.
        cluster_vertices = topology.reading(np.array(sorted(merged_cluster)), 0)
        cluster_quadric = ((cluster_vertices * (topology.incidence[2, 0].sum(axis=0).adjugate() * initial_quadrics)).batch()).sum(axis=0)
        optimum = new_batch[added_indices[0]]
        assert cluster_quadric(optimum).norm() < 1e-9
        clusters = next_clusters


def test_the_scenes_run_and_draw():
    plt.close(render.draw_qem(*scenarios.qem()))
    topology, vertices = scenarios.bunny()
    V, E, F = topology.n_elements
    one_pass = np.array([F - 2])
    assert len(list(render.animate_surface(vertices, core.simplify(topology, vertices, one_pass)))) == 2
