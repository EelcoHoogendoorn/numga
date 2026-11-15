"""Tests for discrete K-surfaces verify edge-length and star-planarity preservation, Bäcklund tangency and Bianchi permutability, mirror equivariance, and GPU rendering."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from examples.surfaces.k_surfaces import core, render, scenarios

tolerance = 1e-10


def crossing_normals(n_half: int, extent: float, corner_angles: np.ndarray) -> tuple[core.Plane, core.Plane]:
    """Turning a reference normal by `extent` in two planes separated by `corner_angles` builds crossing normal arcs (`[n_vertices] Plane+`, `[..., n_vertices] Plane+`)."""
    n_vertices = 2 * n_half + 1
    angles = np.linspace(-extent, extent, n_vertices)                                    # [n_vertices]
    alternating = 1 - 2 * ((np.arange(n_vertices) - n_half) % 2)                         # [n_vertices]
    reference_normal = core.mv.vector([0, 0, 1])                                         # [] Plane+
    corner_turns = (core.mv.xy * (-corner_angles / 2)).exp()                             # [...] Rotor+
    planes_v = corner_turns >> core.mv.yz                                                # [...] Bivector+
    turns_u = (core.mv.yz * (-angles / 2)).exp()                                         # [n_vertices] Rotor+
    mid_u = turns_u >> reference_normal                                                  # [n_vertices] Plane+
    turns_v = (planes_v[..., None] * (-angles / 2)).exp()                                # [..., n_vertices] Rotor+
    mid_v = alternating * (turns_v >> reference_normal)                                  # [..., n_vertices] Plane+
    return mid_u, mid_v


@pytest.mark.parametrize("n_half", [3, 6])
def test_saddle_deformation_preserves_edge_lengths_and_planar_stars(n_half: int):
    extent = 0.75
    corner_angles = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])                      # [3]
    radius = core.mv.xyz.dual()                                                          # [] Antiscalar-
    origin = core.mv.z.dual() * 0                                                        # [] Point-
    mid_u, mid_v = crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.quad_normal)                            # [3, n_vertices, n_vertices] Plane+
    points = core.surface_points(normals, radius, origin)                                # [3, n_vertices, n_vertices] Point-
    edges_u = radius * normals[..., 1:, :].commutator(normals[..., :-1, :])              # [3, n_vertices - 1, n_vertices] Point-
    edges_v = radius * normals[..., :, 1:].commutator(normals[..., :, :-1])              # [3, n_vertices, n_vertices - 1] Point-
    closure = edges_u[..., :, :-1] + edges_v[..., 1:, :] - edges_u[..., :, 1:] - edges_v[..., :-1, :]
    centre = points[..., 1:-1, 1:-1]
    star = stack([points[..., 2:, 1:-1] - centre, points[..., :-2, 1:-1] - centre,
                  points[..., 1:-1, 2:] - centre, points[..., 1:-1, :-2] - centre], axis=-1)
    expected_length = np.sin(extent / n_half)
    diagonal_lengths = (points[..., -1, -1] - points[..., 0, 0]).antinorm()              # [3] Scalar+

    assert ((normals.scalar_norm_squared() - 1).abs() < tolerance).all()
    assert (closure.antinorm() < tolerance).all()
    assert ((normals[..., 1:-1, 1:-1, None] & star).abs() < tolerance).all()
    assert ((edges_u.antinorm() - expected_length).abs() < tolerance).all()
    assert ((edges_v.antinorm() - expected_length).abs() < tolerance).all()
    assert ((points[..., 1:, :] - points[..., :-1, :] - edges_u).antinorm() < tolerance).all()
    assert ((points[..., :, 1:] - points[..., :, :-1] - edges_v).antinorm() < tolerance).all()
    assert diagonal_lengths.max() - diagonal_lengths.min() > 0.2


def test_backlund_connectors_and_bianchi_completion_integrate_to_same_surface():
    n_half, extent = 5, 0.65
    corner_angles = np.array([1.15, 1.55])                                               # [2]
    radius = core.mv.xyz.dual()                                                          # [] Antiscalar-
    origin = core.mv.z.dual() * 0                                                        # [] Point-
    mid_u, mid_v = crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.quad_normal)                            # [2, n_vertices, n_vertices] Plane+
    points = core.surface_points(normals, radius, origin)                                # [2, n_vertices, n_vertices] Point-
    turn_a = (core.mv.yz * 0.4).exp()                                                    # [] Rotor+
    turn_b = ((core.mv.xy - core.mv.zx) * 0.35).exp()                                    # [] Rotor+
    starts = stack([turn_a >> normals[..., 0, 0], turn_b >> normals[..., 0, 0]])         # [2, 2] Plane+
    transformed = core.backlund_normals(normals, starts, core.quad_normal)               # [2, 2, n_vertices, n_vertices] Plane+
    connectors = radius * transformed.commutator(normals)                                # [2, 2, n_vertices, n_vertices] Point-
    transformed_points = points + connectors                                             # [2, 2, n_vertices, n_vertices] Point-
    independent = core.surface_points(transformed, radius, transformed_points[..., 0, 0])
    normal_a, normal_b = transformed
    point_a, point_b = transformed_points
    normal_ab = core.quad_normal(normals, normal_a, normal_b)
    point_ab_a = point_a + radius * normal_ab.commutator(normal_a)
    point_ab_b = point_b + radius * normal_ab.commutator(normal_b)
    independent_ab = core.surface_points(normal_ab, radius, point_ab_a[..., 0, 0])
    propagated_ab_a = core.backlund_normals(normal_a, normal_ab[..., 0, 0], core.quad_normal)
    propagated_ab_b = core.backlund_normals(normal_b, normal_ab[..., 0, 0], core.quad_normal)

    assert ((transformed.scalar_norm_squared() - 1).abs() < tolerance).all()
    assert ((connectors.antinorm() - connectors[..., :1, :1].antinorm()).abs() < tolerance).all()
    assert ((normals & connectors).abs() < tolerance).all()
    assert ((transformed & connectors).abs() < tolerance).all()
    assert ((independent - transformed_points).antinorm() < tolerance).all()
    assert ((point_ab_a - point_ab_b).antinorm() < tolerance).all()
    assert ((independent_ab - point_ab_a).antinorm() < tolerance).all()
    assert ((propagated_ab_a - normal_ab).norm() < tolerance).all()
    assert ((propagated_ab_b - normal_ab).norm() < tolerance).all()


def test_reflecting_input_frame_reflects_saddle_and_backlund_surface():
    n_half, extent = 5, 0.65
    corner_angles = np.array([1.2, 1.6])                                                 # [2]
    radius = core.mv.xyz.dual()                                                          # [] Antiscalar-
    origin = core.mv.vector([0.2, -0.1, 0.3]).dual()                                     # [] Point-
    mirror = core.mv.vector([2, -1, 3]).normalized()                                     # [] Plane+
    mid_u, mid_v = crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.quad_normal)                            # [2, n_vertices, n_vertices] Plane+
    points = core.surface_points(normals, radius, origin)                                # [2, n_vertices, n_vertices] Point-
    seed_turn = (core.mv.yz * 0.4).exp()                                                 # [] Rotor+
    start = seed_turn >> normals[..., 0, 0]                                              # [2] Plane+
    transformed = core.backlund_normals(normals, start, core.quad_normal)                # [2, n_vertices, n_vertices] Plane+
    transformed_points = points + radius * transformed.commutator(normals)               # [2, n_vertices, n_vertices] Point-
    reflected_normals = core.normal_net(mirror >> mid_u, mirror >> mid_v, core.quad_normal)
    reflected_points = core.surface_points(reflected_normals, mirror >> radius, mirror >> origin)
    reflected_transformed = core.backlund_normals(reflected_normals, mirror >> start, core.quad_normal)
    reflected_transformed_points = reflected_points + (mirror >> radius) * reflected_transformed.commutator(reflected_normals)

    assert ((reflected_normals - (mirror >> normals)).norm() < tolerance).all()
    assert ((reflected_points - (mirror >> points)).antinorm() < tolerance).all()
    assert ((reflected_transformed_points - (mirror >> transformed_points)).antinorm() < tolerance).all()


def test_saddle_and_backlund_renderers_produce_figures_and_frames():
    normals, points, transformed_normals, transformed_points = scenarios.backlund_pair()
    del normals, transformed_normals
    saddle_fig = render.saddle_figure(points, points[None])
    plt.close(saddle_fig)
    pair_fig = render.backlund_figure(transformed_points, points, points)
    plt.close(pair_fig)
    orbit_frame = next(render.backlund_orbit_animation(transformed_points, points, np.array([0.0]), scenarios.orbit_elevation))
    assert orbit_frame.ndim == 3 and orbit_frame.shape[2] == 3
