"""Mesh topology: the incidences compose to nothing, the boundary encloses the faces, and the disk constructors give
regular triangles."""

import numpy as np
import pytest

from numga.algebras import VGA3D
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", VGA3D)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv


def sides(mesh: Mesh, vertices):
    """The edge facing each corner of each face `[3] Point[F]`, one of the edges signed as the face runs along it."""
    return ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()


def corner_angles(mesh: Mesh, vertices) -> np.ndarray:
    """Each corner's angle in degrees `[3, F]`: one edge's slice read along the other, over both lengths."""
    edges = sides(mesh, vertices)
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
    return np.rad2deg(np.arccos(np.clip((-(after.dual_inverse() & before) / (after.norm() * before.norm())).to_array(), -1, 1)))


@pytest.mark.parametrize("built", (Mesh.disk(2.0, 8, 48), Mesh.triangular_disk(2.0, 10),
                                   Mesh.concentric_disk(np.array([0.4, 1.7, 2.0]), 10)),
                         ids=("rings", "triangular", "concentric"))
def test_boundary_integral_matches_oriented_face_area(built):
    mesh, vertices = built
    turn = (mv.yz * -0.24).exp()
    vertices = turn >> vertices + mv.x.dual()

    # Circulation around a face of a difference vanishes.
    np.testing.assert_allclose((mesh.d1 * mesh.d0).cells.kernel, 0.0, atol=0.0)
    # Interior edges cancel in the sum of face boundaries. The surviving edge pairs
    # enclose exactly the same oriented area as the sum of triangular faces.
    boundary = ~mesh.d1 * as_scalar(np.ones(len(mesh.faces))).field()
    tail, head = vertices.batch()[mesh.edges.T]                               # [E] Point each
    boundary_area = (boundary.batch() * (tail & head)).sum() / 2
    edges = sides(mesh, vertices)
    face_area = (edges[0] & edges[1]).sites.sum() / 2
    np.testing.assert_allclose((boundary_area - face_area).kernel, 0.0, atol=1e-11)
    np.testing.assert_array_equal(np.flatnonzero(boundary.kernel[:, 0]), mesh.boundary_edges)


@pytest.mark.parametrize("divisions", (6, 10, 16, 24, 32))
def test_triangular_disk_has_regular_interior_and_resolved_circular_boundary(divisions: int):
    radius = 0.1
    mesh, vertices = Mesh.triangular_disk(radius, divisions)
    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    degrees = np.bincount(mesh.edges.ravel(), minlength=mesh.n_vertices)
    interior = np.setdiff1d(np.arange(mesh.n_vertices), boundary)
    lengths = (mesh.d0 * vertices).norm().kernel[:, 0]
    edges = sides(mesh, vertices)
    planes = (edges[1].commutator(edges[0])).normalized()                     # Bivector[F]
    areas = (edges[0] & edges[1]).norm().kernel[:, 0] / 2
    centres = vertices.batch()[mesh.faces.T].sum(axis=0) / 3                 # [F] Point
    rim_areas = areas[centres.norm().kernel[:, 0] > 0.75 * radius]
    angles = corner_angles(mesh, vertices)

    # Every interior vertex has six neighbours; the boundary follows the circle.
    assert np.all(degrees[interior] == 6)
    np.testing.assert_allclose((vertices.norm().batch()[boundary] - radius).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((planes - mv.xy).kernel, 0, atol=1e-12)
    # Edge lengths and areas remain uniform through the rim without thin triangles.
    assert lengths.std() < 0.1 * lengths.mean()
    assert areas.std() < 0.1 * areas.mean()
    assert rim_areas.std() < 0.15 * rim_areas.mean()
    assert angles.min() > 35
    assert angles.max() < 95


@pytest.mark.parametrize("divisions", (10, 24, 32))
def test_concentric_disk_follows_material_interfaces_without_thin_cells(divisions: int):
    boundaries = np.array([0.02, 0.085, 0.1])
    mesh, vertices = Mesh.concentric_disk(boundaries, divisions)
    radii = vertices.norm().kernel[:, 0]
    edge_radii = radii[mesh.edges]
    edges = sides(mesh, vertices)
    planes = (edges[1].commutator(edges[0])).normalized()                     # Bivector[F]
    areas = (edges[0] & edges[1]).norm().kernel[:, 0] / 2
    angles = corner_angles(mesh, vertices)
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
    np.testing.assert_allclose((planes - mv.xy).kernel, 0, atol=tolerance)
    assert angles.min() > 30
    assert angles.max() < 105
    assert areas.std() < 0.2 * areas.mean()
