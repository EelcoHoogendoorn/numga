"""Topology: the incidences compose to nothing, the boundary encloses the faces, the disk shapes give regular
triangles, a closed surface is wound outward, and a topology's border, closedness, orientation, manifoldness and
Euler characteristic are what its shape says."""

import numpy as np
import pytest

from numga import NumpyContext, stack

from numga.algebras import VGA3D
from examples.util import shapes
from examples.util.topology import Topology, turned

mv = NumpyContext(VGA3D).multivector
# The planes that read a shape's coordinates: of x and of y for a disk, all three for a surface in space.
Upright = VGA3D.gatype.from_blades("x y")
Vector = VGA3D.gatype.vector()


def as_scalar(values: np.ndarray):
    return mv.scalar(np.asarray(values)[..., None])


def placed(shape, planes):
    """A shape's topology with its vertices at its coordinates, read by the planes given."""
    topology, coordinates = shape
    return topology, mv(planes, coordinates).dual().field()


def corner_angles(topology: Topology, vertices) -> np.ndarray:
    """Each corner's angle in degrees `[3, F]`: one edge's slice read along the other, over both lengths."""
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)
    tails, heads = topology.corners[2, 1]
    after, before = edges[tails], edges[heads]
    return np.rad2deg(np.arccos(np.clip((-(after.dual_inverse() & before) / (after.norm() * before.norm())).to_array(), -1, 1)))


@pytest.mark.parametrize("built", (placed(shapes.disk(2.0, 8, 48), Upright), placed(shapes.triangular_disk(2.0, 10), Upright),
                                   placed(shapes.concentric_disk(np.array([0.4, 1.7, 2.0]), 10), Upright)),
                         ids=("rings", "triangular", "concentric"))
def test_boundary_integral_matches_oriented_face_area(built):
    topology, vertices = built
    turn = (mv.yz * -0.24).exp()
    vertices = turn >> vertices + mv.x.dual()

    # Circulation around a face of a difference vanishes.
    np.testing.assert_allclose((topology.d[1] * topology.d[0]).cells, 0.0, atol=0.0)
    # Interior edges cancel in the sum of face boundaries. The surviving edge pairs
    # enclose exactly the same oriented area as the sum of triangular faces.
    boundary = ~topology.d[1] * as_scalar(np.ones(topology.n_elements[2])).field()
    tail, head = (topology.incidence[1, 0] * vertices).batch()                      # [E] Point-
    boundary_area = (boundary.batch() * (tail & head)).sum() / 2
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)
    face_area = (edges[0] & edges[1]).sites.sum() / 2
    np.testing.assert_allclose((boundary_area - face_area).kernel, 0.0, atol=1e-11)
    np.testing.assert_array_equal(np.flatnonzero(boundary.kernel[:, 0]), topology.exterior[1])


@pytest.mark.parametrize("divisions", (6, 10, 16, 24, 32))
def test_triangular_disk_has_regular_interior_and_resolved_circular_boundary(divisions: int):
    radius = 0.1
    topology, vertices = placed(shapes.triangular_disk(radius, divisions), Upright)
    boundary, interior = topology.reading(topology.exterior[0]), topology.reading(topology.interior[0])
    # Each vertex's degree: every edge at it, summed back onto its ends.
    degrees = ~topology.incidence[1, 0].sum(axis=0) * as_scalar(np.ones(topology.n_elements[1])).field()   # Scalar+[vertices]
    lengths = (topology.d[0] * vertices).norm().kernel[:, 0]
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)
    planes = (edges[1].commutator(edges[0])).normalized()                     # Bivector+[F]
    areas = (edges[0] & edges[1]).norm().kernel[:, 0] / 2
    centres = topology.incidence[2, 0].mean(axis=0) * vertices               # Point-[F]
    rim_areas = areas[centres.norm().kernel[:, 0] > 0.75 * radius]
    angles = corner_angles(topology, vertices)

    # Every interior vertex has six neighbours; the boundary follows the circle.
    assert np.all((interior * degrees).kernel == 6)
    np.testing.assert_allclose(((boundary * vertices).norm() - radius).kernel, 0, atol=1e-12)
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
    topology, vertices = placed(shapes.concentric_disk(boundaries, divisions), Upright)
    radii = vertices.norm().kernel[:, 0]
    edge_radii = (topology.incidence[1, 0] * vertices.norm()).kernel[..., 0]   # [ends, edges]
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)
    planes = (edges[1].commutator(edges[0])).normalized()                     # Bivector[F]
    areas = (edges[0] & edges[1]).norm().kernel[:, 0] / 2
    angles = corner_angles(topology, vertices)
    tolerance = 1e-12

    # Every prescribed circle is an unbroken chain of mesh edges; no face straddles it.
    for radius in boundaries:
        on_ring = np.abs(radii - radius) < tolerance
        assert np.count_nonzero(on_ring) >= 6
        # An edge lies on the ring where both its ends do; along those, every vertex on the ring has two neighbours.
        tails, heads = topology.incidence[1, 0] * as_scalar(on_ring * 1.0).field()     # Scalar+[edges], Scalar+[edges]
        degrees = (~topology.incidence[1, 0].sum(axis=0) * (tails * heads)).kernel[:, 0]   # [vertices]
        assert np.all(degrees[on_ring] == 2)
        assert not np.any((edge_radii.min(axis=0) < radius - tolerance)
                          & (edge_radii.max(axis=0) > radius + tolerance))
    np.testing.assert_allclose((planes - mv.xy).kernel, 0, atol=tolerance)
    assert angles.min() > 30
    assert angles.max() < 105
    assert areas.std() < 0.2 * areas.mean()


def test_boundaries_of_boundaries_vanish_in_every_dimension():
    # A cube of six tetrahedra around its diagonal, and a path of edges, oriented each its own way.
    solid = Topology(8, np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]))
    path = Topology(4, np.array([[0, 1], [2, 1], [3, 2]]))

    np.testing.assert_allclose((solid.d[1] * solid.d[0]).cells, 0.0, atol=0.0)
    np.testing.assert_allclose((solid.d[2] * solid.d[1]).cells, 0.0, atol=0.0)
    # The cube's boundary: its six squares, each two triangles, their twelve edges and the six diagonals, every corner.
    assert [len(solid.exterior[k]) for k in (2, 1, 0)] == [12, 18, 8]
    # A tet's edges are signed by whether they run the way its own vertex order runs them, and its faces by their
    # order in it, which is their outward orientation up to the sign of the corner they face.
    tails, heads = np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3])
    ones = as_scalar(np.ones(solid.n_elements[1])).field()
    np.testing.assert_array_equal((solid.oriented[3, 1] * ones).kernel[..., 0],
                                  np.where(solid.elements[3][:, tails] < solid.elements[3][:, heads], 1.0, -1.0).T)
    facing = (-1.0) ** np.arange(4)[:, None, None]
    np.testing.assert_array_equal(solid.oriented[3, 2].cells, facing[..., 0] * solid.boundary[3, 2].cells)
    # Each edge's derivative is its head less its tail.
    heights = as_scalar(np.arange(4.0)).field()
    np.testing.assert_array_equal((path.d[0] * heights).kernel[:, 0], [1.0, -1.0, -1.0])


@pytest.mark.parametrize("built", (placed(shapes.sphere(2), Vector), placed(shapes.torus(2.0, 0.5, 16, 8), Vector)),
                         ids=("sphere", "torus"))
def test_closed_mesh_is_wound_outward_whatever_the_given_winding(built):
    topology, vertices = built
    faces = topology.elements[2]
    coordinates = (stack((mv.x, mv.y, mv.z)) & vertices.batch()[..., None]).to_array()
    # Turn a scattered half of the faces, then all of them: either way the closed mesh recovers the outward winding.
    scattered = np.where((np.arange(len(faces)) % 3 == 0)[:, None], faces[:, [1, 0, 2]], faces)
    for given in (scattered, faces[:, [1, 0, 2]]):
        wound = Topology.closed(given, coordinates).elements[2]
        same_corners = np.sort(wound, axis=-1) == np.sort(faces, axis=-1)
        same_sense = np.any([np.all(np.roll(faces, shift, axis=-1) == wound, axis=-1) for shift in range(3)], axis=0)
        assert same_corners.all() and same_sense.all()


def test_corners_operator_indexing_and_shapes():
    triangle = Topology(3, [[0, 1, 2]])
    assert triangle.corners[2, 1].shape == (2, 3)
    tails, heads = triangle.corners[2, 1]
    np.testing.assert_array_equal(tails, [1, 0, 0])
    np.testing.assert_array_equal(heads, [2, 2, 1])

    tet = Topology(4, [[0, 1, 2, 3]])
    assert tet.corners[3, 1].shape == (2, 6)
    assert tet.corners[3, 2].shape == (3, 4)
    first, second, third = tet.corners[3, 2]
    assert len(first) == 4 and len(second) == 4 and len(third) == 4

    # corners must strictly use bracket indexing; call syntax is forbidden with no backwards compatibility.
    with pytest.raises(TypeError):
        triangle.corners(2, 1)


# The six tetrahedra of a unit cube around its diagonal, its corners numbered by their bits along x, y and z.
cube_corners = np.array([[(index >> axis) & 1 for axis in range(3)] for index in range(8)], dtype=float)
cube_tets = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])


@pytest.mark.parametrize("built, euler", ((shapes.sphere(2), 2), (shapes.torus(2.0, 0.5, 16, 8), 0)),
                         ids=("sphere", "torus"))
def test_closed_surfaces_have_no_border_and_their_euler_characteristic(built, euler):
    topology, _ = built
    assert topology.is_closed and topology.is_oriented and topology.is_manifold and topology.n_components == 1
    assert len(topology.border.elements[-1]) == 0
    assert topology.euler_characteristic == euler


def test_a_disk_is_bordered_by_a_closed_loop():
    topology, _ = shapes.disk(1.0, 4, 12)
    rim = topology.border
    assert not topology.is_closed and topology.euler_characteristic == 1
    assert rim.n_dim == 1 and rim.n_elements[1] == 12
    assert rim.is_closed and rim.is_oriented and rim.is_manifold and rim.euler_characteristic == 0


def test_a_solids_border_is_its_closed_surface_facing_out():
    # Each tetrahedron wound to positive volume, its edges from its first corner a right-handed frame.
    edges = cube_corners[cube_tets[:, 1:]] - cube_corners[cube_tets[:, :1]]     # [tets, 3, 3]
    solid = Topology(8, np.where((np.linalg.det(edges) > 0)[:, None], cube_tets, turned(cube_tets)))
    surface = solid.border
    assert solid.euler_characteristic == 1 and not solid.is_closed and solid.is_manifold
    assert surface.n_dim == 2 and surface.n_elements[2] == 12
    assert surface.is_closed and surface.is_oriented and surface.euler_characteristic == 2
    assert len(surface.border.elements[-1]) == 0
    # Each triangle faces away from the cube's middle: the cone it spans from there has positive volume.
    about_middle = mv.vector(cube_corners - 0.5).dual().field()                  # Point-[vertices]
    corners = surface.incidence[2, 0] * about_middle                             # [3] Point-[faces]
    assert np.all((corners[0] & corners[1] & corners[2]) > 0)


def test_pinched_and_folded_topologies_are_told_apart():
    # Two triangles touching only at a vertex: a pinched star, and two pieces.
    pinched = Topology(5, np.array([[0, 1, 2], [0, 3, 4]]))
    assert not pinched.is_manifold and pinched.n_components == 2
    # Three triangles on one edge: a facet where three meet.
    fin = Topology(5, np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]))
    assert not fin.is_manifold
    # Two triangles running their shared edge the same way: not oriented, until one is turned.
    folded = Topology(4, np.array([[0, 1, 2], [0, 1, 3]]))
    assert not folded.is_oriented and Topology(4, np.array([[0, 1, 2], [1, 0, 3]])).is_oriented


@pytest.mark.parametrize("name, n_vertices, euler", (
    ("tetrahedron", 4, 2), ("cube", 8, 2), ("octahedron", 6, 2), ("dodecahedron", 20, 2), ("icosahedron", 12, 2),
    ("simplex_4", 5, 0), ("tesseract", 16, 0), ("cell_16", 8, 0), ("cell_24", 24, 0), ("cell_120", 600, 0),
    ("cell_600", 120, 0)))
def test_regular_polytopes_are_closed_surfaces_of_their_counts(name, n_vertices, euler):
    topology, coordinates = getattr(shapes, name)()
    assert len(coordinates) == n_vertices and topology.n_dim == coordinates.shape[1] - 1
    assert topology.is_closed and topology.is_oriented and topology.is_manifold
    assert topology.euler_characteristic == euler
    np.testing.assert_allclose(np.linalg.norm(coordinates, axis=-1), 1.0, atol=1e-12)


def test_the_600_cell_is_its_600_tetrahedra():
    topology, _ = shapes.cell_600()
    assert topology.n_elements[3] == 600


@pytest.mark.parametrize("solid, euler, border_euler", ((shapes.ball(1, 2), 1, 2), (shapes.solid_torus(1.0, 0.3, 10, 2), 0, 0)),
                         ids=("ball", "solid torus"))
def test_a_solids_border_is_the_surface_of_its_name(solid, euler, border_euler):
    topology, coordinates = solid
    assert topology.n_dim == 3 and topology.is_manifold and not topology.is_closed
    assert topology.euler_characteristic == euler
    border = topology.border
    assert border.is_closed and border.is_oriented and border.is_manifold
    assert border.euler_characteristic == border_euler
    # Every tetrahedron is wound to positive volume: the join of its edges from its first corner.
    corners = topology.incidence[3, 0] * mv.vector(coordinates).dual().field()   # [4] Point-[tets]
    edges = corners[1:] - corners[0]                                             # [3] Point-[tets]
    assert np.all((edges[0] & edges[1] & edges[2]) > 0)


@pytest.mark.parametrize("shape", (shapes.sphere(1), shapes.disk(1.0, 3, 12), shapes.ball(1, 2)), ids=("sphere", "disk", "ball"))
def test_the_dual_runs_the_transposed_derivatives_and_its_edges_end_on_both_sides(shape):
    topology, _ = shape
    dual = topology.dual
    assert dual.n_elements == topology.n_elements[::-1]
    rng = np.random.default_rng(0)
    # Its derivatives follow each other to nothing, as the primal's do.
    for k in range(topology.n_dim - 1):
        values = as_scalar(rng.normal(size=dual.n_elements[k])).field()
        np.testing.assert_allclose((dual.d[k + 1] * (dual.d[k] * values)).kernel, 0, atol=1e-12)
    # The ends of each dual edge differ by its derivative, and a constant does not change across an interior facet.
    values = as_scalar(rng.normal(size=dual.n_elements[0])).field()
    tails, heads = dual.incidence[1, 0] * values
    np.testing.assert_allclose((heads - tails - dual.d[0] * values).kernel, 0, atol=1e-12)
    constant = as_scalar(np.ones(dual.n_elements[0])).field()
    inner = dual.reading(topology.interior[-2], 1)
    np.testing.assert_allclose((inner * (dual.d[0] * constant)).kernel, 0, atol=1e-12)


def test_loops_close_their_runs_and_a_subset_keeps_its_vertices():
    loops = Topology.loops([3, 5])
    assert loops.is_closed and loops.n_components == 2 and loops.euler_characteristic == 0
    np.testing.assert_array_equal(loops.elements[1][[2, 7]], [[2, 0], [7, 3]])
    topology, coordinates = shapes.disk(1.0, 3, 12)
    truss = topology.subset(topology.interior[1], 1)
    assert truss.n_dim == 1 and truss.n_vertices == topology.n_vertices
    assert truss.n_elements[1] == len(topology.interior[1])
    # A mask of the faces and the indices it picks give one topology.
    near = np.arange(topology.n_elements[2]) % 3 == 0
    np.testing.assert_array_equal(topology.subset(near).elements[2], topology.subset(np.flatnonzero(near)).elements[2])
