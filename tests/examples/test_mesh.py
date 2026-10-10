"""Sparse mesh derivatives, geometric reconstruction, and the boundary agree."""

import numpy as np
import pytest

from examples.mesh import Mesh, as_scalar, at_sites, mv


@pytest.mark.parametrize("mesh", (Mesh.disk(1.7, 8, 48), Mesh.triangular_disk(1.7, 10),
                                  Mesh.concentric_disk(np.array([0.34, 1.445, 1.7]), 10)),
                         ids=("rings", "triangular", "concentric"))
def test_affine_field_reconstruction_and_energy_follow_a_turned_mesh(mesh: Mesh):
    turn = (mv.xz * 0.31).exp()
    mesh = mesh.copy(vertices=turn >> mesh.vertices)
    gradient = turn >> (mv.x + 2 * mv.y)
    potential = mesh.vertices | gradient
    differences = mesh.d0 * potential

    # Integrating a gradient around a triangle vanishes, and edge integrals reconstruct
    # the same constant vector on every face, including after turning the whole mesh.
    np.testing.assert_allclose((mesh.d1 * mesh.d0).cells.kernel, 0.0, atol=0.0)
    np.testing.assert_allclose((mesh.d1 * differences).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((mesh.reconstruction(differences) - gradient).kernel, 0.0, atol=1e-12)

    # The cotangent Hodge measures the same energy as integrating the squared gradient.
    edge_energy = (mesh.edge_ratio * differences.squared()).batch().sum()
    face_energy = mesh.triangle_areas.batch().sum() * gradient.scalar_norm_squared()
    np.testing.assert_allclose((edge_energy - face_energy).kernel, 0.0, atol=1e-11)


@pytest.mark.parametrize("mesh", (Mesh.disk(2.0, 8, 48), Mesh.triangular_disk(2.0, 10),
                                  Mesh.concentric_disk(np.array([0.4, 1.7, 2.0]), 10)),
                         ids=("rings", "triangular", "concentric"))
def test_boundary_integral_matches_oriented_face_area(mesh: Mesh):
    turn = (mv.yz * -0.24).exp()
    mesh = mesh.copy(vertices=turn >> mesh.vertices + mv.x)

    # Interior edges cancel in the sum of face boundaries. The surviving edge pairs
    # enclose exactly the same oriented area as the sum of triangular faces.
    boundary = ~mesh.d1 * as_scalar(np.ones(len(mesh.faces))).field()
    edge_area = at_sites(mesh.vertices, mesh.edges[:, 0]) ^ at_sites(mesh.vertices, mesh.edges[:, 1])
    boundary_area = (boundary * edge_area).batch().sum() / 2
    face_area = (mesh.triangle_areas * mesh.face_planes).batch().sum()
    np.testing.assert_allclose((boundary_area - face_area).kernel, 0.0, atol=1e-11)
    np.testing.assert_array_equal(np.flatnonzero(boundary.kernel[:, 0]), mesh.boundary_edges)


@pytest.mark.parametrize("divisions", (6, 10, 16, 24, 32))
def test_triangular_disk_has_regular_interior_and_resolved_circular_boundary(divisions: int):
    radius = 0.1
    mesh = Mesh.triangular_disk(radius, divisions)
    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    radii = mesh.vertices.norm()
    degrees = np.bincount(mesh.edges.ravel(), minlength=len(mesh.vertices.batch()))
    interior = np.setdiff1d(np.arange(len(mesh.vertices.batch())), boundary)
    angles = np.rad2deg(np.arccos(np.clip(mesh.corner_cosines().kernel, -1, 1)))
    lengths = mesh.edge_vectors.norm().kernel[:, 0]
    areas = mesh.triangle_areas.kernel[:, 0]
    rim_areas = areas[mesh.face_centers.norm().kernel[:, 0] > 0.75 * radius]

    # Every interior vertex has six neighbours; the boundary follows the circle.
    assert np.all(degrees[interior] == 6)
    np.testing.assert_allclose((radii.batch()[boundary] - radius).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((mesh.face_planes - mv.xy).kernel, 0, atol=1e-12)
    # Edge lengths and areas remain uniform through the rim without thin triangles.
    assert lengths.std() < 0.1 * lengths.mean()
    assert areas.std() < 0.1 * areas.mean()
    assert rim_areas.std() < 0.15 * rim_areas.mean()
    assert angles.min() > 35
    assert angles.max() < 95


@pytest.mark.parametrize("divisions", (10, 24, 32))
def test_concentric_disk_follows_material_interfaces_without_thin_cells(divisions: int):
    boundaries = np.array([0.02, 0.085, 0.1])
    mesh = Mesh.concentric_disk(boundaries, divisions)
    radii = mesh.vertices.norm().kernel[:, 0]
    edge_radii = radii[mesh.edges]
    angles = np.rad2deg(np.arccos(np.clip(mesh.corner_cosines().kernel, -1, 1)))
    areas = mesh.triangle_areas.kernel[:, 0]
    tolerance = 1e-12

    # Every prescribed circle is an unbroken chain of mesh edges; no face straddles it.
    for radius in boundaries:
        on_ring = np.abs(radii - radius) < tolerance
        assert np.count_nonzero(on_ring) >= 6
        ring_edges = mesh.edges[np.all(on_ring[mesh.edges], axis=-1)]
        degrees = np.bincount(ring_edges.ravel(), minlength=len(radii))
        assert np.all(degrees[on_ring] == 2)
        assert not np.any((edge_radii.min(axis=-1) < radius - tolerance)
                          & (edge_radii.max(axis=-1) > radius + tolerance))
    np.testing.assert_allclose((mesh.face_planes - mv.xy).kernel, 0, atol=tolerance)
    assert angles.min() > 30
    assert angles.max() < 105
    assert areas.std() < 0.2 * areas.mean()
