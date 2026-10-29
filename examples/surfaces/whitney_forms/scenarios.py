"""Reference triangles, vertex stars, edge diamonds, disk meshes and a reference tetrahedron for Whitney forms."""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag
from examples import instantiate
from examples.surfaces.whitney_forms.core import SpaceMesh, SpacePoint, space_mv

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                  # Vector+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 8
n_star_divisions = 2
n_tet_subdivisions = 6
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


def edge_diamond(n_subdivisions: int) -> tuple[Mesh, Point, Point]:
    """Two triangles sharing a vertical interior edge, and the interior sample points of each."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    bottom, top = -vertical * 0.72, vertical * 0.72                          # [] Point-, [] Point-
    left_apex = -horizontal * 0.95 + vertical * 0.14                         # [] Point-
    right_apex = horizontal * 0.92 - vertical * 0.12                         # [] Point-
    vertices = stack((bottom, top, left_apex, right_apex)).field()           # Point-[vertices]
    mesh = Mesh(4, np.array([[0, 1, 2], [1, 0, 3]]))
    return mesh, vertices, triangle_samples(mesh.incidence[2, 0] * vertices, n_subdivisions)  # ..., [samples] Point-[faces]


def tetrahedron_samples(corners: SpacePoint, n_subdivisions: int) -> SpacePoint:
    """Interior barycentric lattice points inside each tetrahedron of corners `[corners] Point-[tets]`,
    `[samples] Point-[tets]`."""
    steps = np.arange(1, n_subdivisions)
    first, second, third = np.meshgrid(steps, steps, steps, indexing="ij")
    inside = first + second + third < n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside] - third[inside],
                        first[inside], second[inside], third[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-[tets]


def single_tetrahedron(n_subdivisions: int) -> tuple[SpaceMesh, SpacePoint, SpacePoint]:
    """A mesh of one positively oriented scalene tetrahedron in space, its vertices, and its interior sample points."""
    x, y, z = space_mv.x.dual(), space_mv.y.dual(), space_mv.z.dual()        # [] Point-, [] Point-, [] Point-
    vertices = stack((-x * 0.56 - y * 0.38 - z * 0.34,
                      x * 0.62 - y * 0.28 - z * 0.34,
                      -x * 0.05 + y * 0.68 - z * 0.34,
                      x * 0.03 + y * 0.02 + z * 0.66)).field()               # Point-[vertices]
    mesh = SpaceMesh(4, [[0, 1, 2, 3]])
    return mesh, vertices, tetrahedron_samples(mesh.incidence[3, 0] * vertices, n_subdivisions)


def deformed_tetrahedron() -> SpacePoint:
    """The vertices of a strongly sheared, stretched and shifted tetrahedron of the same sense, `Point-[vertices]`."""
    x, y, z = space_mv.x.dual(), space_mv.y.dual(), space_mv.z.dual()        # [] Point-, [] Point-, [] Point-
    return stack((x * 0.40 - y * 1.10 + z * 0.25,
                  x * 2.30 - y * 0.35 - z * 0.15,
                  x * 0.85 + y * 1.65 + z * 0.55,
                  x * 1.15 + y * 0.20 + z * 2.45)).field()                   # Point-[vertices]


def disk(radius: float, n_divisions: int) -> tuple[Mesh, Point]:
    """A circular disk of nearly equilateral triangles with its first vertex at the origin."""
    spacing = radius / n_divisions
    indices = np.arange(-n_divisions, n_divisions + 1)
    columns, rows = np.meshgrid(indices, indices, indexing="ij")
    layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
    inside = (layers > 0) & (layers <= n_divisions)
    lattice = np.stack((columns[inside], rows[inside]), axis=-1)             # [nodes, axes]
    unit = mv.x.dual()                                                       # [] Point-
    axes = stack((unit, (mv.xy * (-np.pi / 6)).exp() >> unit))               # [axes] Point-
    positions = (axes * lattice).sum(axis=-1)                                # [nodes] Point-
    positions = spacing * layers[inside] * positions.normalized()            # [nodes] Point-
    vertices = concatenate((unit[None] * 0, positions)).field()              # Point-[vertices]
    return Mesh.triangulate(vertices), vertices


def obtuse_diamond() -> tuple[Mesh, Point]:
    """Two obtuse triangles sharing a long horizontal edge where circumcentric DEC's Hodge weight is negative."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    vertices = stack((horizontal * 0.0 + vertical * 0.0,
                      horizontal * 2.0 + vertical * 0.0,
                      horizontal * 1.0 + vertical * 0.4,
                      horizontal * 1.0 - vertical * 0.4)).field()            # Point-[vertices]
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return Mesh(4, faces), vertices


def main() -> None:
    from examples.surfaces.whitney_forms.core import (
        Operators, barycentric_hats, complements, conduction, edge_conductance, edge_whitney_forms, simplex_parity,
        tetrahedral_edge_forms, tetrahedral_face_forms, tetrahedral_hats, tetrahedral_helicity,
    )
    from examples.surfaces.whitney_forms import render
    from examples.util.animation import save_figure

    # 1. Barycentric hat functions and piecewise-constant face gradients.
    corners, points = single_triangle(n_triangle_subdivisions)
    hats, gradients = barycentric_hats(corners, points)
    star_mesh, star_vertices = disk(disk_radius, n_star_divisions)
    star_operators = Operators(star_mesh, star_vertices)
    vertex_hat = as_scalar(np.arange(star_mesh.n_vertices) == 0).field()     # Scalar+[vertices]
    star_gradients = star_operators.gradient(vertex_hat)                     # Plane+[faces]
    save_figure(
        render.draw_hat_0forms(corners, gradients, star_mesh, star_vertices, vertex_hat, star_gradients),
        "whitney_hat_0form",
    )

    # 2. Edge Whitney 1-forms and tangential continuity across a shared edge.
    edge_forms = edge_whitney_forms(corners, points)                         # [corners, samples] Plane+
    diamond_mesh, diamond_vertices, diamond_points = edge_diamond(n_triangle_subdivisions)
    diamond = Operators(diamond_mesh, diamond_vertices)
    # Unit circulation on the one interior edge, read on both faces through their edge forms,
    # each signed by the sense in which its face runs along the edge.
    shared_circulation = as_scalar(np.isin(np.arange(len(diamond_mesh.edges)), diamond_mesh.interior_edges)).field()  # Scalar+[edges]
    sample_forms = (diamond_points[:, None] - diamond.corners[None, :]) / (2 * diamond.patches)  # [samples, corners] Plane+[faces]
    sample_whitney1 = (spdiag(sample_forms) * diamond_mesh.boundary[2, 1]).sum(axis=1) * Scalar  # [samples] (Plane+ <- Scalar+)[faces <- edges]
    shared_forms = sample_whitney1(shared_circulation)                       # [samples] Plane+[faces]
    save_figure(
        render.draw_edge_1forms(corners, points, edge_forms,
                                diamond_mesh, diamond_vertices, diamond_points, shared_forms),
        "whitney_edge_1form",
    )

    # 3. Whitney 0-, 1- and 2-forms on a tetrahedron, and the volume pairing that links opposite edges.
    tetrahedron, tet_vertices, tet_points = single_tetrahedron(n_tet_subdivisions)
    tet_hats, tet_gradients, face_patches, volume = tetrahedral_hats(tetrahedron, tet_vertices, tet_points)
    tet_edge_forms, tet_curls = tetrahedral_edge_forms(tetrahedron, tet_vertices, tet_points)
    tet_face_forms = tetrahedral_face_forms(tetrahedron, tet_vertices, tet_points)
    save_figure(
        render.draw_tet_whitney(tetrahedron, tet_vertices, tet_points, tet_edge_forms, tet_face_forms, volume),
        "whitney_tet_forms",
    )

    # 4. Exact midpoint quadrature and anisotropic conduction across a disk.
    mesh, vertices = disk(disk_radius, n_disk_divisions)
    operators = Operators(mesh, vertices)
    fibre = (mv.xy * (-fibre_angle / 2)).exp() >> mv.x.dual()                # [] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    directions = stack((fibre, quarter_turn >> fibre))                       # [axes] Point-
    conductivity = (directions * (Plane & directions) * principal_conductivities).sum(axis=-1)  # [cases] Point- <- Plane+

    boundary = mesh.exterior(0)
    abscissa = (mv.x & vertices).batch()                                     # [vertices] Scalar+
    electrodes = boundary[abscissa[boundary].norm() > electrode_threshold]
    pinned = mesh.reading(electrodes)
    prescribed = ~pinned * (pinned * (mv.x & vertices))                      # Scalar+[vertices]
    potential, current, streamfunction = conduction(
        operators, conductivity, electrodes, prescribed,
    )
    save_figure(
        render.draw_anisotropic_flow(mesh, vertices, operators.centres, potential, streamfunction, current, fibre),
        "whitney_anisotropic_flow",
    )

    # checks
    corner_hats, _ = barycentric_hats(corners, corners)
    np.testing.assert_allclose(corner_hats.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-14)

    first, second = [1, 2, 0], [2, 0, 1]
    alternating_forms = hats[first] * gradients[second, None] - hats[second] * gradients[first, None]
    np.testing.assert_allclose((edge_forms - alternating_forms).kernel, 0.0, atol=1e-14)
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]
    midpoints = (corners[first] + corners[second]) / 2
    midpoint_readings = edge_whitney_forms(corners, midpoints) & sides[None, :]
    np.testing.assert_allclose(midpoint_readings.kernel, np.eye(3)[..., None], atol=1e-14)

    tails, heads = diamond_mesh.incidence[1, 0] * diamond_vertices
    shared = diamond_mesh.reading(diamond_mesh.interior_edges, 1)                 # Scalar+[shared <- edges]
    shared_edge, = (shared * (diamond_mesh.d[0] * diamond_vertices)).batch()
    seam_point, = (shared * (tails * 0.35 + heads * 0.65)).batch()
    seam_forms = (seam_point - diamond.corners) / (2 * diamond.patches)
    seam_whitney1 = (spdiag(seam_forms) * diamond_mesh.boundary[2, 1]).sum(axis=0) * Scalar
    np.testing.assert_allclose((seam_whitney1(shared_circulation) & shared_edge).kernel, 1.0, atol=1e-14)

    tet_corners = tetrahedron.incidence[3, 0] * tet_vertices
    corner_tet_hats, _, _, _ = tetrahedral_hats(tetrahedron, tet_vertices, tet_corners)
    np.testing.assert_allclose(corner_tet_hats.kernel, np.eye(4)[..., None, None], atol=1e-13)
    np.testing.assert_allclose(tet_hats.sum(axis=0).kernel, 1.0, atol=1e-13)
    np.testing.assert_allclose(tet_gradients.sum(axis=0).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(face_patches.sum(axis=0).kernel, 0.0, atol=1e-13)

    tails, heads = tetrahedron.corners(3, 1).T
    alternating_edge_forms = tet_hats[tails] * tet_gradients[heads, None] - tet_hats[heads] * tet_gradients[tails, None]
    alternating_curls = 2 * (tet_gradients[tails] ^ tet_gradients[heads])
    np.testing.assert_allclose((tet_edge_forms - alternating_edge_forms).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((tet_curls - alternating_curls).kernel, 0.0, atol=1e-13)

    tet_edges = tetrahedron.oriented[3, 1] * (tetrahedron.d[0] * tet_vertices)
    edge_midpoints = tetrahedron.incidence[3, 1] * (tetrahedron.incidence[1, 0].mean(axis=0) * tet_vertices)
    midpoint_edge_forms, _ = tetrahedral_edge_forms(tetrahedron, tet_vertices, edge_midpoints)
    np.testing.assert_allclose((midpoint_edge_forms & tet_edges[None, :]).kernel, np.eye(6)[..., None, None], atol=1e-13)

    first, second, third = complements(tetrahedron, 0).T
    alternating_face_forms = 2 * (tet_hats[first] * (tet_gradients[second, None] ^ tet_gradients[third, None])
                                  + tet_hats[second] * (tet_gradients[third, None] ^ tet_gradients[first, None])
                                  + tet_hats[third] * (tet_gradients[first, None] ^ tet_gradients[second, None]))
    np.testing.assert_allclose((tet_face_forms - alternating_face_forms).kernel, 0.0, atol=1e-13)
    face_centres = tetrahedron.incidence[3, 2] * (tetrahedron.incidence[2, 0].mean(axis=0) * tet_vertices)
    centre_face_forms = tetrahedral_face_forms(tetrahedron, tet_vertices, face_centres)
    np.testing.assert_allclose((centre_face_forms & face_patches[None, :]).kernel, np.eye(4)[..., None, None], atol=1e-13)

    tet_helicity = tetrahedral_helicity(tetrahedron, tet_vertices)
    deformed_helicity = tetrahedral_helicity(tetrahedron, deformed_tetrahedron())
    np.testing.assert_allclose((tet_helicity - deformed_helicity).kernel, 0.0, atol=1e-13)
    # Two edges link where they share no corner, with the sense of the tetrahedron's corners in the order they list
    # them, over six.
    edge_corners = tetrahedron.corners(3, 1)
    edge_pairs = np.concatenate(np.broadcast_arrays(edge_corners[:, None], edge_corners[None, :]), axis=-1)
    disjoint = (edge_corners[:, None, :, None] != edge_corners[None, :, None, :]).all(axis=(-1, -2))
    linking = np.where(disjoint, simplex_parity(edge_pairs), 0) / 6
    np.testing.assert_allclose(tet_helicity.kernel[..., 0, 0], linking, atol=1e-13)

    linear_potential = (mv.x & vertices) * 0.7 - (mv.y & vertices) * 1.1
    np.testing.assert_allclose(operators.d1(operators.d0(linear_potential)).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((operators.whitney1(operators.d0(linear_potential)) - operators.gradient(linear_potential)).kernel, 0.0, atol=1e-14)
    test_cochain = operators.d0(linear_potential) + as_scalar(np.sin(np.arange(len(mesh.edges)))).field()
    np.testing.assert_allclose((operators.curl(test_cochain) - operators.whitney2(operators.d1(test_cochain))).kernel, 0.0, atol=1e-12)

    edge_conductances = edge_conductance(operators, conductivity)
    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))
    quadratic_potential = (mv.x & vertices).squared() - 0.5 * (mv.y & vertices)
    np.testing.assert_allclose(
        (operators.d0.adjugate()(edge_conductances(operators.d0(quadratic_potential))) - conductance(quadratic_potential)).kernel,
        0.0, atol=1e-13,
    )
    np.testing.assert_allclose((conductance - conductance.adjugate()).cells.kernel, 0.0, atol=1e-13)

    obtuse_mesh, obtuse_vertices = obtuse_diamond()
    obtuse_ops = Operators(obtuse_mesh, obtuse_vertices)
    obtuse_edge_conductance = edge_conductance(obtuse_ops, Plane.dual())
    shared_cochain = as_scalar(np.isin(np.arange(len(obtuse_mesh.edges)), obtuse_mesh.interior_edges)).field()
    assert -(obtuse_edge_conductance(shared_cochain) & shared_cochain).sites.sum() > 0

    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~mesh.reading(free_vertices) * Scalar
    np.testing.assert_allclose(free.adjugate()(conductance(potential)).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
