"""Compatible saddle nets, tangent Bäcklund connectors and Bianchi completion."""

import numpy as np
import pytest

from numga import stack
from examples.surfaces.k_surfaces import core


ROUND_OFF = 1e-10


def _crossing_normals(n_half, extent, corner_angles):
    n_vertices = 2 * n_half + 1
    angles = np.linspace(-extent, extent, n_vertices)
    alternating = 1 - 2 * ((np.arange(n_vertices) - n_half) % 2)
    reference_normal = core.mv.vector([0, 0, 1])
    planes_v = (core.mv.xy * (-corner_angles / 2)).exp() >> core.mv.yz
    mid_u = (core.mv.yz * (-angles / 2)).exp() >> reference_normal
    mid_v = alternating * ((planes_v[..., None] * (-angles / 2)).exp() >> reference_normal)
    return mid_u, mid_v


@pytest.mark.parametrize("n_half", [7, 12])
def test_saddle_deformation_preserves_edge_lengths_and_planar_stars(n_half):
    extent = 0.75
    corner_angles = np.array([np.pi / 3, np.pi / 2, 2 * np.pi / 3])
    radius = (core.mv.x ^ core.mv.y ^ core.mv.z).dual()
    origin = core.mv.z.dual() * 0
    mid_u, mid_v = _crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.complete)
    points = core.positions(normals, radius, origin)
    edges_u = radius * normals[..., 1:, :].commutator(normals[..., :-1, :])
    edges_v = radius * normals[..., :, 1:].commutator(normals[..., :, :-1])
    closure = edges_u[..., :, :-1] + edges_v[..., 1:, :] - edges_u[..., :, 1:] - edges_v[..., :-1, :]
    centre = points[..., 1:-1, 1:-1]
    star = stack([points[..., 2:, 1:-1] - centre, points[..., :-2, 1:-1] - centre,
                  points[..., 1:-1, 2:] - centre, points[..., 1:-1, :-2] - centre], axis=-1)
    expected_length = np.sin(extent / n_half)
    diagonal_lengths = (points[..., -1, -1] - points[..., 0, 0]).norm()

    # checks
    np.testing.assert_allclose(normals.scalar_norm_squared().kernel, 1, atol=ROUND_OFF)
    np.testing.assert_allclose(closure.kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((normals[..., 1:-1, 1:-1, None] & star).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose(edges_u.norm().kernel, expected_length, atol=ROUND_OFF)
    np.testing.assert_allclose(edges_v.norm().kernel, expected_length, atol=ROUND_OFF)
    np.testing.assert_allclose((points[..., 1:, :] - points[..., :-1, :] - edges_u).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((points[..., :, 1:] - points[..., :, :-1] - edges_v).kernel, 0, atol=ROUND_OFF)
    assert np.ptp(diagonal_lengths.kernel) > 0.5


def test_backlund_connectors_and_bianchi_completion_integrate_to_the_same_surface():
    n_half, extent = 12, 0.65
    corner_angles = np.array([1.15, 1.55])
    radius = (core.mv.x ^ core.mv.y ^ core.mv.z).dual()
    origin = core.mv.z.dual() * 0
    mid_u, mid_v = _crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.complete)
    points = core.positions(normals, radius, origin)
    turn_a = (core.mv.yz * 0.4).exp()
    turn_b = ((core.mv.xy + core.mv.xz) * 0.35).exp()
    starts = stack([turn_a >> normals[..., 0, 0], turn_b >> normals[..., 0, 0]])
    partners = core.partner(normals, starts, core.complete)
    connectors = radius * partners.commutator(normals)
    partner_points = points + connectors
    independent_partners = core.positions(partners, radius, partner_points[..., 0, 0])
    normal_a, normal_b = partners
    point_a, point_b = partner_points
    normal_ab = core.complete(normals, normal_a, normal_b)
    point_ab_a = point_a + radius * normal_ab.commutator(normal_a)
    point_ab_b = point_b + radius * normal_ab.commutator(normal_b)
    independent_ab = core.positions(normal_ab, radius, point_ab_a[..., 0, 0])
    propagated_ab_a = core.partner(normal_a, normal_ab[..., 0, 0], core.complete)
    propagated_ab_b = core.partner(normal_b, normal_ab[..., 0, 0], core.complete)

    # checks
    np.testing.assert_allclose(partners.scalar_norm_squared().kernel, 1, atol=ROUND_OFF)
    np.testing.assert_allclose((connectors.norm() - connectors[..., :1, :1].norm()).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((normals & connectors).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((partners & connectors).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((independent_partners - partner_points).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((point_ab_a - point_ab_b).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((independent_ab - point_ab_a).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((propagated_ab_a - normal_ab).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((propagated_ab_b - normal_ab).kernel, 0, atol=ROUND_OFF)


def test_reflecting_the_input_frame_reflects_the_saddle_and_its_partner():
    n_half, extent = 12, 0.65
    corner_angles = np.array([1.2, 1.6])
    radius = (core.mv.x ^ core.mv.y ^ core.mv.z).dual()
    origin = core.mv.vector([0.2, -0.1, 0.3]).dual()
    mirror = core.mv.vector([2, -1, 3]).normalized()
    mid_u, mid_v = _crossing_normals(n_half, extent, corner_angles)
    normals = core.normal_net(mid_u, mid_v, core.complete)
    points = core.positions(normals, radius, origin)
    start = (core.mv.yz * 0.4).exp() >> normals[..., 0, 0]
    partner_normals = core.partner(normals, start, core.complete)
    partner_points = points + radius * partner_normals.commutator(normals)
    reflected_normals = core.normal_net(mirror >> mid_u, mirror >> mid_v, core.complete)
    reflected_points = core.positions(reflected_normals, mirror >> radius, mirror >> origin)
    reflected_partner = core.partner(reflected_normals, mirror >> start, core.complete)
    reflected_partner_points = reflected_points + (mirror >> radius) * reflected_partner.commutator(reflected_normals)

    # checks
    np.testing.assert_allclose((reflected_normals - (mirror >> normals)).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((reflected_points - (mirror >> points)).kernel, 0, atol=ROUND_OFF)
    np.testing.assert_allclose((reflected_partner_points - (mirror >> partner_points)).kernel, 0, atol=ROUND_OFF)
