"""Reference triangles, vertex stars, edge diamonds, disk meshes and a reference tetrahedron for Whitney forms."""

from __future__ import annotations

from functools import reduce
from operator import and_

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag
from examples.util import shapes
from examples.util.topology import Topology
from examples.surfaces.whitney_forms.core import SpacePoint, space_mv

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                  # Vector+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 8
n_star_divisions = 2
n_tet_subdivisions = 6
# The level surfaces of each edge form drawn, and the cuts of each side of a face whose parts' middles its face form's
# field lines run through.
n_pages = 5
n_face_cuts = 4
n_disk_divisions = 9
disk_radius = 1.0
electrode_threshold = 0.8
fibre_angle = np.pi / 6
principal_conductivities = np.array([[1, 1], [10, 1]])                       # [cases, axes]


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Interior barycentric lattice points inside a triangle."""
    steps = np.arange(1, n_subdivisions)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-


def single_triangle(n_subdivisions: int) -> tuple[Point, Point]:
    """A scalene triangle and its interior sample points."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    corners = stack((horizontal * 0.0 + vertical * 0.0,
                     horizontal * 1.2 + vertical * 0.12,
                     horizontal * 0.38 + vertical * 0.96))                   # [corners] Point-
    return corners, triangle_samples(corners, n_subdivisions)


def edge_diamond(n_subdivisions: int) -> tuple[Topology, Point, Point]:
    """Two triangles sharing a vertical interior edge, and the interior sample points of each."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    bottom, top = -vertical * 0.72, vertical * 0.72                          # [] Point-, [] Point-
    left_apex = -horizontal * 0.95 + vertical * 0.14                         # [] Point-
    right_apex = horizontal * 0.92 - vertical * 0.12                         # [] Point-
    vertices = stack((bottom, top, left_apex, right_apex)).field()           # Point-[vertices]
    topology = Topology(4, np.array([[0, 1, 2], [1, 0, 3]]))
    return topology, vertices, triangle_samples(topology.incidence[2, 0] * vertices, n_subdivisions)  # ..., [samples] Point-[faces]


def tetrahedron_samples(corners: SpacePoint, n_subdivisions: int) -> SpacePoint:
    """Interior barycentric lattice points inside each tetrahedron of corners `[corners] Point-[tets]`,
    `[samples] Point-[tets]`."""
    steps = np.arange(1, n_subdivisions)
    first, second, third = np.meshgrid(steps, steps, steps, indexing="ij")
    inside = first + second + third < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside] - third[inside],
                        first[inside], second[inside], third[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-[tets]


def single_tetrahedron(n_subdivisions: int) -> tuple[Topology, SpacePoint, SpacePoint]:
    """A mesh of one positively oriented scalene tetrahedron in space, its vertices, and its interior sample points."""
    vertices = space_mv.point([[-0.56, -0.38, -0.34],
                               [0.62, -0.28, -0.34],
                               [-0.05, 0.68, -0.34],
                               [0.03, 0.02, 0.66]]).field()                  # Point-[vertices]
    topology = Topology(4, [[0, 1, 2, 3]])
    return topology, vertices, tetrahedron_samples(topology.incidence[3, 0] * vertices, n_subdivisions)


def deformed_tetrahedron() -> SpacePoint:
    """The vertices of a strongly sheared, stretched and shifted tetrahedron of the same sense, `Point-[vertices]`."""
    return space_mv.point([[0.40, -1.10, 0.25],
                           [2.30, -0.35, -0.15],
                           [0.85, 1.65, 0.55],
                           [1.15, 0.20, 2.45]]).field()                      # Point-[vertices]


def obtuse_diamond() -> tuple[Topology, Point]:
    """Two obtuse triangles sharing a long horizontal edge where circumcentric DEC's Hodge weight is negative."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    vertices = stack((horizontal * 0.0 + vertical * 0.0,
                      horizontal * 2.0 + vertical * 0.0,
                      horizontal * 1.0 + vertical * 0.4,
                      horizontal * 1.0 - vertical * 0.4)).field()            # Point-[vertices]
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return Topology(4, faces), vertices


def main() -> None:
    from examples.surfaces.whitney_forms.core import (
        Operators, barycentric_hats, complements, conduction, edge_conductance, edge_whitney_forms, simplex_parity,
        tetrahedral_edge_pages, tetrahedral_face_lines, tetrahedral_forms, tetrahedral_helicity, triangle,
    )
    from examples.surfaces.whitney_forms import render
    from examples.util.animation import save_figure

    # 1. Barycentric hat functions and piecewise-constant face gradients.
    corners, points = single_triangle(n_triangle_subdivisions)
    hats, gradients = barycentric_hats(corners, points)
    star_topology, star_coordinates = shapes.hexagonal_disk(disk_radius, n_star_divisions)
    star_vertices = mv.vector(star_coordinates).dual().field()               # Point-[vertices]
    vertex_hat = mv.scalar((np.arange(star_topology.n_vertices) == 0)[..., None]).field()  # Scalar+[vertices]
    save_figure(
        render.draw_hat_0forms(corners, gradients, star_topology, star_vertices, vertex_hat),
        "whitney_hat_0form",
    )

    # 2. Edge Whitney 1-forms and tangential continuity across a shared edge.
    edge_forms = edge_whitney_forms(corners, points)                         # [corners, samples] Plane+
    diamond_topology, diamond_vertices, _ = edge_diamond(n_triangle_subdivisions)
    diamond = Operators(diamond_topology, diamond_vertices)
    # Unit circulation on the one interior edge, read on both faces through their edge forms,
    # each signed by the sense in which its face runs along the edge.
    shared_circulation = mv.scalar(np.isin(np.arange(diamond_topology.n_elements[1]), diamond_topology.interior[1])[..., None]).field()  # Scalar+[edges]
    corner_forms = edge_whitney_forms(corners, corners)                      # [edges, corners] Plane+
    corner_offsets = (diamond.corners[:, None] - diamond.corners[None, :]) / (2 * diamond.patches)
    corner_whitney1 = (spdiag(corner_offsets) * diamond_topology.boundary[2, 1]).sum(axis=1) * Scalar
    shared_corner_forms = corner_whitney1(shared_circulation)                 # [corners] Plane+[faces]
    save_figure(
        render.draw_edge_1forms(corners, corner_forms, diamond_topology, diamond_vertices, shared_corner_forms),
        "whitney_edge_1form",
    )

    # 3. Whitney 0-, 1- and 2-forms on a tetrahedron, and the volume pairing that links opposite edges.
    tetrahedron, tet_vertices, tet_points = single_tetrahedron(n_tet_subdivisions)
    # The forms of each degree, one quotient each: the corners' ramps, the edges' forms and the faces' forms.
    ramps, tet_edge_forms, tet_face_forms = (tetrahedral_forms(tetrahedron, tet_vertices, m) for m in range(3))  # [corners] Plane+[tets], [tet_edges] Bivector+[tets], [corners] Trivector+[tets]
    tet_pages = tetrahedral_edge_pages(tetrahedron, tet_vertices, n_pages)     # [tet_edges, pages, 3] Point-[tets]
    tet_lines = tetrahedral_face_lines(tetrahedron, tet_vertices, n_face_cuts)  # [corners, lines, 2] Point-[tets]
    save_figure(
        render.draw_tet_whitney(tetrahedron, tet_vertices, tet_pages, tet_lines),
        "whitney_tet_forms",
    )

    # 4. Exact midpoint quadrature and anisotropic conduction across a disk.
    topology, coordinates = shapes.hexagonal_disk(disk_radius, n_disk_divisions)
    vertices = mv.vector(coordinates).dual().field()                         # Point-[vertices]
    operators = Operators(topology, vertices)
    fibre = (mv.xy * (-fibre_angle / 2)).exp() >> mv.x.dual()                # [] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    directions = stack((fibre, quarter_turn >> fibre))                       # [axes] Point-
    conductivity = (directions * (Plane & directions) * principal_conductivities).sum(axis=-1)  # [cases] Point- <- Plane+

    boundary = topology.exterior[0]
    abscissa = (mv.x & vertices).batch()                                     # [vertices] Scalar+
    electrodes = boundary[abscissa[boundary].norm() > electrode_threshold]
    pinned = topology.reading(electrodes, 0)
    prescribed = ~pinned * (pinned * (mv.x & vertices))                      # Scalar+[vertices]
    potential, current, streamfunction = conduction(
        operators, conductivity, electrodes, prescribed,
    )
    save_figure(
        render.draw_anisotropic_flow(topology, vertices, potential, current, fibre),
        "whitney_anisotropic_flow",
    )

    # checks
    corner_hats, _ = barycentric_hats(corners, corners)
    np.testing.assert_allclose(corner_hats.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    first, second = complements(triangle, 0).T
    alternating_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - alternating_forms).kernel, 0.0, atol=1e-14)
    sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners.field())).batch()[:, 0]
    midpoints = (corners[first] + corners[second]) / 2
    midpoint_readings = edge_whitney_forms(corners, midpoints) & sides[None, :]
    np.testing.assert_allclose(midpoint_readings.kernel, np.eye(3)[..., None], atol=1e-14)

    tails, heads = diamond_topology.incidence[1, 0] * diamond_vertices
    shared = diamond_topology.reading(diamond_topology.interior[1], 1)            # Scalar+[shared <- edges]
    shared_edge, = (shared * (diamond_topology.d[0] * diamond_vertices)).batch()
    seam_point, = (shared * (tails * 0.35 + heads * 0.65)).batch()
    seam_forms = (seam_point - diamond.corners) / (2 * diamond.patches)
    seam_whitney1 = (spdiag(seam_forms) * diamond_topology.boundary[2, 1]).sum(axis=0) * Scalar
    np.testing.assert_allclose((seam_whitney1(shared_circulation) & shared_edge).kernel, 1.0, atol=1e-14)

    # Each degree's forms read one on their own simplex's join, in the sense it is stored, and nothing on the others.
    for m, forms in enumerate((ramps, tet_edge_forms, tet_face_forms)):
        spans = reduce(and_, tetrahedron.on[3, m] * tet_vertices)
        np.testing.assert_allclose((forms[:, None] & spans[None, :]).kernel, np.eye(spans.shape[0])[..., None, None], atol=1e-13)
    tet_hats = ramps[:, None] & tet_points[None, :]
    np.testing.assert_allclose(tet_hats.sum(axis=0).kernel, 1.0, atol=1e-13)

    # Whitney's formulas from the ramps: the edge forms, and the face forms, which read one on the join of a face's
    # corners, twice the face; without a point, each form is the wedge of the ramps at its simplex's corners.
    tails, heads = tetrahedron.corners[3, 1]
    alternating_edge_forms = tet_hats[tails] * ramps[heads, None] - tet_hats[heads] * ramps[tails, None]
    np.testing.assert_allclose(((tet_edge_forms[:, None] & tet_points[None, :]) - alternating_edge_forms).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((tet_edge_forms - (ramps[tails] ^ ramps[heads])).kernel, 0.0, atol=1e-13)
    first, second, third = tetrahedron.corners[3, 2]
    alternating_face_forms = 2 * (tet_hats[first] * (ramps[second, None] ^ ramps[third, None])
                                  + tet_hats[second] * (ramps[third, None] ^ ramps[first, None])
                                  + tet_hats[third] * (ramps[first, None] ^ ramps[second, None]))
    np.testing.assert_allclose((2 * (tet_face_forms[:, None] & tet_points[None, :]) - alternating_face_forms).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((tet_face_forms - (ramps[first] ^ ramps[second] ^ ramps[third])).kernel, 0.0, atol=1e-13)

    tet_helicity = tetrahedral_helicity(tetrahedron, tet_vertices)
    deformed_helicity = tetrahedral_helicity(tetrahedron, deformed_tetrahedron())
    np.testing.assert_allclose((tet_helicity - deformed_helicity).kernel, 0.0, atol=1e-13)
    # Two edges link where they share no corner, with the sense of the tetrahedron's corners in the order they list
    # them, over six.
    edge_corners = tetrahedron.corners[3, 1].T
    edge_pairs = np.concatenate(np.broadcast_arrays(edge_corners[:, None], edge_corners[None, :]), axis=-1)
    disjoint = (edge_corners[:, None, :, None] != edge_corners[None, :, None, :]).all(axis=(-1, -2))
    linking = np.where(disjoint, simplex_parity(edge_pairs), 0) / 6
    np.testing.assert_allclose(tet_helicity.kernel[..., 0, 0], linking, atol=1e-13)

    linear_potential = (mv.x & vertices) * 0.7 - (mv.y & vertices) * 1.1
    np.testing.assert_allclose(operators.d1(operators.d0(linear_potential)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((operators.whitney1(operators.d0(linear_potential)) - operators.gradient(linear_potential)).kernel, 0.0, atol=1e-14)
    test_cochain = operators.d0(linear_potential) + mv.scalar(np.sin(np.arange(topology.n_elements[1]))[..., None]).field()
    np.testing.assert_allclose((operators.curl(test_cochain) - operators.whitney2(operators.d1(test_cochain))).kernel, 0.0, atol=1e-12)

    edge_conductances = edge_conductance(operators, conductivity)
    conductance = operators.gradient.adjugate()((operators.areas * conductivity)(operators.gradient))
    quadratic_potential = (mv.x & vertices).squared() - 0.5 * (mv.y & vertices)
    np.testing.assert_allclose(
        (operators.d0.adjugate()(edge_conductances(operators.d0(quadratic_potential))) - conductance(quadratic_potential)).kernel,
        0.0, atol=1e-13,
    )
    np.testing.assert_allclose((conductance - conductance.adjugate()).cells.kernel, 0.0, atol=1e-13)

    obtuse_topology, obtuse_vertices = obtuse_diamond()
    obtuse_ops = Operators(obtuse_topology, obtuse_vertices)
    obtuse_edge_conductance = edge_conductance(obtuse_ops, Plane.dual())
    shared_cochain = mv.scalar(np.isin(np.arange(obtuse_topology.n_elements[1]), obtuse_topology.interior[1])[..., None]).field()
    assert (obtuse_edge_conductance(shared_cochain) & shared_cochain).sites.sum() > 0

    free_vertices = np.setdiff1d(np.arange(topology.n_vertices), electrodes)
    free = ~topology.reading(free_vertices, 0) * Scalar
    np.testing.assert_allclose(free.adjugate()(conductance(potential)).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
