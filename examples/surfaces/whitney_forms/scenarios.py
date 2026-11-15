"""Reference triangles, vertex stars, two-triangle diamonds, triangulated disks, and scalene tetrahedra provide test domains for Whitney forms."""

from __future__ import annotations

from functools import reduce
from operator import and_

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga
from numga.topology import Topology, simplex_parity
from examples.surfaces.whitney_forms.core import SpacePoint, space_mv
from examples.util import shapes

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 8
n_star_divisions = 2
n_tet_subdivisions = 6
n_pages = 5
n_face_cuts = 4
n_disk_divisions = 9
disk_radius = 1.0
electrode_threshold = 0.8
fibre_angle = np.pi / 6
principal_conductivities = np.array([[1, 1], [10, 1]])                       # [2, 2]


# --- plumbing -------------------------------------------------------------------------
def face_part_middles(n_cuts: int) -> np.ndarray:
    """Subdividing each triangle edge into `n_cuts` equal segments partitions the face into equal-area subtriangles with centre weights (`[n_cuts * n_cuts, 3]`)."""
    steps = np.arange(n_cuts)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    upright = first + second < n_cuts
    inverted = first + second < n_cuts - 1
    upright_corners = np.stack((first[upright], second[upright], n_cuts - 1 - first[upright] - second[upright]), axis=-1)
    inverted_corners = np.stack((first[inverted], second[inverted], n_cuts - 2 - first[inverted] - second[inverted]),
                                axis=-1)
    return np.concatenate((upright_corners + 1 / 3, inverted_corners + 2 / 3)) / n_cuts


def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Uniform lattice weights inside a reference triangle place interior sample points (`[...] Point-[F]`) across each face of `corners` (`[3] Point-[F]`)."""
    steps = np.arange(1, n_subdivisions)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [..., 3]
    return (corners * weights).sum(axis=-1)                                  # [...] Point-[F]


def single_triangle(n_subdivisions: int) -> tuple[Topology, Point, Point]:
    """Three non-collinear planar points define a single scalene triangle topology, its vertices (`Point-[V]`), and interior sample points (`[...] Point-[F]`)."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    vertices = stack((horizontal * 0.0 + vertical * 0.0,
                      horizontal * 1.2 + vertical * 0.12,
                      horizontal * 0.38 + vertical * 0.96)).field()          # Point-[V]
    topology = Topology(3, [[0, 1, 2]])
    return topology, vertices, triangle_samples(topology.incidence[2, 0] * vertices, n_subdivisions)


def edge_diamond() -> tuple[Topology, Point]:
    """Four planar vertices define two triangles sharing a vertical interior edge (`Point-[V]`)."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    bottom, top = -vertical * 0.72, vertical * 0.72                          # [] Point-, [] Point-
    left_apex = -horizontal * 0.95 + vertical * 0.14                         # [] Point-
    right_apex = horizontal * 0.92 - vertical * 0.12                         # [] Point-
    vertices = stack((bottom, top, left_apex, right_apex)).field()           # Point-[V]
    topology = Topology(4, np.array([[0, 1, 2], [1, 0, 3]]))
    return topology, vertices


def tetrahedron_samples(corners: SpacePoint, n_subdivisions: int) -> SpacePoint:
    """Uniform lattice weights inside a reference tetrahedron place interior sample points (`[...] Point-[T]`) across each element of `corners` (`[4] Point-[T]`)."""
    steps = np.arange(1, n_subdivisions)
    first, second, third = np.meshgrid(steps, steps, steps, indexing="ij")
    inside = first + second + third < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside] - third[inside],
                        first[inside], second[inside], third[inside]), axis=-1) / n_subdivisions  # [..., 4]
    return (corners * weights).sum(axis=-1)                                  # [...] Point-[T]


def single_tetrahedron(n_subdivisions: int) -> tuple[Topology, SpacePoint, SpacePoint]:
    """Four non-coplanar projective points define a positively oriented scalene tetrahedron, its vertices (`Point-[V]`), and interior sample points (`[...] Point-[T]`)."""
    vertices = space_mv.point([[-0.56, -0.38, -0.34],
                               [0.62, -0.28, -0.34],
                               [-0.05, 0.68, -0.34],
                               [0.03, 0.02, 0.66]]).field()                  # Point-[V]
    topology = Topology(4, [[0, 1, 2, 3]])
    return topology, vertices, tetrahedron_samples(topology.incidence[3, 0] * vertices, n_subdivisions)


def deformed_tetrahedron() -> SpacePoint:
    """Sheared, stretched, and translated projective points define a deformed tetrahedron (`Point-[V]`) of positive orientation."""
    return space_mv.point([[0.40, -1.10, 0.25],
                           [2.30, -0.35, -0.15],
                           [0.85, 1.65, 0.55],
                           [1.15, 0.20, 2.45]]).field()                      # Point-[V]


def obtuse_diamond() -> tuple[Topology, Point]:
    """Two obtuse triangles sharing a long horizontal edge form a diamond (`Point-[V]`) whose circumcentric Hodge weight is negative."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    vertices = stack((horizontal * 0.0 + vertical * 0.0,
                      horizontal * 2.0 + vertical * 0.0,
                      horizontal * 1.0 + vertical * 0.4,
                      horizontal * 1.0 - vertical * 0.4)).field()            # Point-[V]
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return Topology(4, faces), vertices


def main() -> None:
    from examples.surfaces.whitney_forms import render
    from examples.surfaces.whitney_forms.core import (
        cochain_maps,
        edge_conductance,
        edge_whitney_forms,
        edge_whitney_map,
        simplex_gradient_map,
        simplex_gradients,
        steady_conduction,
        tetrahedral_edge_pages,
        tetrahedral_face_lines,
        tetrahedral_forms,
        tetrahedral_helicity,
        vertex_whitney_forms,
    )
    from examples.util.animation import save_figure

    # 1. Simplex gradients and piecewise-linear vertex 0-forms.
    triangle, triangle_vertices, sample_points = single_triangle(n_triangle_subdivisions)
    patches, gradients = simplex_gradients(triangle, triangle_vertices)      # Patch-[F], [3] Plane+[F]
    vertex_forms = vertex_whitney_forms(triangle, triangle_vertices, sample_points)  # [3, ...] Scalar+[F]
    star_topology, star_coordinates = shapes.hexagonal_disk(disk_radius, n_star_divisions)
    V_star, E_star, F_star = star_topology.n_elements
    star_vertices = mv.vector(star_coordinates).dual().field()               # Point-[V_star]
    vertex_form = mv.scalar((np.arange(V_star) == 0)[..., None]).field()     # Scalar+[V_star]
    save_figure(
        render.vertex_forms_figure(triangle, triangle_vertices, gradients, star_topology, star_vertices, vertex_form),
        "whitney_vertex_0form",
    )

    # 2. Edge Whitney 1-forms and tangential continuity across a shared interior edge.
    triangle_corners = triangle.incidence[2, 0] * triangle_vertices          # [3] Point-[F]
    edge_forms = edge_whitney_forms(triangle, triangle_vertices, sample_points)  # [3, ...] Plane+[F]
    corner_forms = edge_whitney_forms(triangle, triangle_vertices, triangle_corners)  # [3, 3] Plane+[F]
    diamond_topology, diamond_vertices = edge_diamond()
    V_diamond, E_diamond, F_diamond = diamond_topology.n_elements
    diamond_corners = diamond_topology.incidence[2, 0] * diamond_vertices    # [3] Point-[F_diamond]
    shared_circulation = mv.scalar(np.isin(np.arange(E_diamond), diamond_topology.interior[1])[..., None]).field()  # Scalar+[E_diamond]
    shared_corner_forms = edge_whitney_map(diamond_topology, diamond_vertices, diamond_corners)(shared_circulation)  # [3] Plane+[F_diamond]
    save_figure(
        render.edge_forms_figure(triangle, triangle_vertices, corner_forms, diamond_topology, diamond_vertices, shared_corner_forms),
        "whitney_edge_1form",
    )

    # 3. Whitney 0-, 1-, and 2-forms on a tetrahedron, and opposite-edge helicity linkage.
    tetrahedron, tet_vertices, tet_points = single_tetrahedron(n_tet_subdivisions)
    tet_vertex_forms, tet_edge_forms, tet_face_forms = (
        tetrahedral_forms(tetrahedron, tet_vertices, degree) for degree in range(3)
    )
    tet_helicity = tetrahedral_helicity(tetrahedron, tet_vertices)           # [6, 6] Scalar+[T]
    tet_pages = tetrahedral_edge_pages(tetrahedron, tet_vertices, n_pages)   # [6, n_pages, 3] Point-[T]
    tet_lines = tetrahedral_face_lines(tetrahedron, tet_vertices, face_part_middles(n_face_cuts))  # [4, ..., 2] Point-[T]
    save_figure(
        render.tetrahedron_forms_figure(tetrahedron, tet_vertices, tet_pages, tet_lines),
        "whitney_tet_forms",
    )

    # 4. Commuting cochain maps, exact midpoint quadrature, and anisotropic conduction across a disk.
    topology, coordinates = shapes.hexagonal_disk(disk_radius, n_disk_divisions)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                         # Point-[V]
    fibre_turn = (mv.xy * (-fibre_angle / 2)).exp()                          # [] Rotor+
    fibre = fibre_turn >> mv.x.dual()                                        # [] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    directions = stack((fibre, quarter_turn >> fibre))                       # [2] Point-
    conductivity = (directions * (Plane & directions) * principal_conductivities).sum(axis=-1)  # [2] Point- <- Plane+

    boundary = topology.exterior[0]
    exterior = topology.reading(boundary, 0)                                 # Scalar+[V_exterior <- V]
    electrodes = boundary[(exterior * (mv.x & vertices)).abs() > electrode_threshold]
    pinned = topology.reading(electrodes, 0)                                 # Scalar+[V_pinned <- V]
    prescribed = pinned.adjugate() * (pinned * (mv.x & vertices))            # Scalar+[V]
    potential, current = steady_conduction(topology, vertices, conductivity, electrodes, prescribed)  # [2] Scalar+[V], [2] Point-[F]
    save_figure(
        render.conduction_figure(topology, vertices, potential, current, fibre),
        "whitney_anisotropic_flow",
    )

    # checks
    corner_values = vertex_whitney_forms(triangle, triangle_vertices, triangle_corners)
    assert ((corner_values - np.eye(3)).abs() < 1e-10).all()
    assert ((vertex_forms.sum(axis=0) - 1).abs() < 1e-10).all()
    assert (gradients.sum(axis=0).norm() < 1e-10).all()

    tails, heads = triangle.corners[2, 1]
    alternating_forms = triangle.orientation[1][0, :, None] * (
        vertex_forms[tails] * gradients[heads, None] - vertex_forms[heads] * gradients[tails, None]
    )
    assert ((edge_forms - alternating_forms).norm() < 1e-10).all()
    sides = triangle.boundary[2, 1] * (triangle.d[0] * triangle_vertices)
    triangle_midpoints = triangle.incidence[2, 1] * (triangle.incidence[1, 0].mean(axis=0) * triangle_vertices)
    midpoint_readings = edge_whitney_forms(triangle, triangle_vertices, triangle_midpoints) & sides[None, :]
    assert ((midpoint_readings - np.eye(3)).abs() < 1e-10).all()

    diamond_edges = diamond_topology.d[0] * diamond_vertices
    edge_samples = ((diamond_topology.incidence[1, 0] * np.array([[0.15], [0.85]])).sum(axis=0)) * diamond_vertices
    shared_reading = diamond_topology.reading(diamond_topology.interior[1], 1)
    face_from_shared = (diamond_topology.incidence[2, 1] * shared_reading.adjugate()).sum(axis=0)
    seam_points = face_from_shared * (shared_reading * edge_samples)
    seam_edges = face_from_shared * (shared_reading * diamond_edges)
    seam_readings = edge_whitney_map(diamond_topology, diamond_vertices, seam_points[None])[0](shared_circulation)
    assert (((seam_readings & seam_edges) - 1).abs() < 1e-10).all()

    for degree, forms in enumerate((tet_vertex_forms, tet_edge_forms, tet_face_forms)):
        spans = reduce(and_, tetrahedron.on[3, degree] * tet_vertices)
        assert (((forms[:, None] & spans[None, :]) - np.eye(spans.shape[0])).abs() < 1e-10).all()
    tet_weights = tet_vertex_forms[:, None] & tet_points[None, :]
    assert ((tet_weights.sum(axis=0) - 1).abs() < 1e-10).all()

    tet_corners = tetrahedron.incidence[3, 0] * tet_vertices
    tet_tails, tet_heads = tetrahedron.corners[3, 1]
    alternating_tet_edges = tet_weights[tet_tails] * tet_vertex_forms[tet_heads, None] - tet_weights[tet_heads] * tet_vertex_forms[tet_tails, None]
    edge_spans = reduce(and_, tetrahedron.on[3, 1] * tet_vertices)
    assert ((((tet_edge_forms[:, None] & tet_points[None, :]) - alternating_tet_edges)[:, :, None] & tet_corners[None, None, :]).abs() < 1e-10).all()
    assert (((tet_edge_forms - (tet_vertex_forms[tet_tails] ^ tet_vertex_forms[tet_heads]))[:, None] & edge_spans[None, :]).abs() < 1e-10).all()
    face_first, face_second, face_third = tetrahedron.corners[3, 2]
    face_spans = reduce(and_, tetrahedron.on[3, 2] * tet_vertices)
    assert (((tet_face_forms - (tet_vertex_forms[face_first] ^ tet_vertex_forms[face_second] ^ tet_vertex_forms[face_third]))[:, None] & face_spans[None, :]).abs() < 1e-10).all()

    deformed_helicity = tetrahedral_helicity(tetrahedron, deformed_tetrahedron())
    assert ((tet_helicity - deformed_helicity).abs() < 1e-10).all()
    edge_corners = tetrahedron.corners[3, 1].T
    edge_pairs = np.concatenate(np.broadcast_arrays(edge_corners[:, None], edge_corners[None, :]), axis=-1)
    disjoint = (edge_corners[:, None, :, None] != edge_corners[None, :, None, :]).all(axis=(-1, -2))
    linking = np.where(disjoint, simplex_parity(edge_pairs), 0) / 6
    assert ((tet_helicity - linking).abs() < 1e-10).all()

    d0 = topology.d[0] * Scalar
    d1 = topology.d[1] * Scalar
    gradient = simplex_gradient_map(topology, vertices)
    whitney1, curl, whitney2 = cochain_maps(topology, vertices)
    linear_potential = (mv.x & vertices) * 0.7 - (mv.y & vertices) * 1.1
    assert (d1(d0(linear_potential)).abs() < 1e-10).all()
    assert ((whitney1(d0(linear_potential)) - gradient(linear_potential)).norm() < 1e-10).all()
    trial_cochain = d0(linear_potential) + mv.scalar(np.sin(np.arange(E))[..., None]).field()
    assert ((curl(trial_cochain) - whitney2(d1(trial_cochain))).norm() < 1e-10).all()

    disk_patches, disk_gradients = simplex_gradients(topology, vertices)
    conductance = gradient.adjugate()((disk_patches.antinorm() * conductivity)(gradient))
    edge_conductances = edge_conductance(topology, vertices, conductivity)
    quadratic_potential = (mv.x & vertices).squared() - 0.5 * (mv.y & vertices)
    assert ((d0.adjugate()(edge_conductances(d0(quadratic_potential))) - conductance(quadratic_potential)).antinorm() < 1e-10).all()
    assert (((linear_potential & conductance(quadratic_potential)) - (quadratic_potential & conductance(linear_potential))).sites.sum().abs() < 1e-10).all()

    obtuse_topology, obtuse_vertices = obtuse_diamond()
    V_obtuse, E_obtuse, F_obtuse = obtuse_topology.n_elements
    obtuse_conductance = edge_conductance(obtuse_topology, obtuse_vertices, Plane.dual())
    shared_cochain = mv.scalar(np.isin(np.arange(E_obtuse), obtuse_topology.interior[1])[..., None]).field()
    assert ((shared_cochain & obtuse_conductance(shared_cochain)).sites.sum() > 0).all()

    free_vertices = np.setdiff1d(np.arange(V), electrodes)
    free = topology.reading(free_vertices, 0).adjugate() * Scalar
    assert (free.adjugate()(conductance(potential)).antinorm() < 1e-9).all()


if __name__ == "__main__":
    main()
