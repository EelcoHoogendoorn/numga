"""Reference triangles, vertex stars, edge diamonds, and disk meshes for Whitney forms."""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                  # Vector+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 8
n_star_divisions = 2
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


def edge_diamond(n_subdivisions: int) -> tuple[Point, Point, Point, Point]:
    """Two triangles sharing a vertical interior edge, with their interior sample points."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    bottom, top = -vertical * 0.72, vertical * 0.72                          # [] Point-, [] Point-
    left_apex = -horizontal * 0.95 + vertical * 0.14                         # [] Point-
    right_apex = horizontal * 0.92 - vertical * 0.12                         # [] Point-
    left_corners = stack((bottom, top, left_apex))                           # [corners] Point-
    right_corners = stack((top, bottom, right_apex))                         # [corners] Point-
    return (left_corners, triangle_samples(left_corners, n_subdivisions),
            right_corners, triangle_samples(right_corners, n_subdivisions))


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
        Operators, barycentric_hats, conduction, edge_conductance, edge_whitney_forms,
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
    left_corners, left_points, right_corners, right_points = edge_diamond(n_triangle_subdivisions)
    left_shared = edge_whitney_forms(left_corners, left_points)[2]           # [samples] Plane+
    right_shared = -edge_whitney_forms(right_corners, right_points)[2]       # [samples] Plane+
    save_figure(
        render.draw_edge_1forms(corners, points, edge_forms,
                                left_corners, left_points, left_shared,
                                right_corners, right_points, right_shared),
        "whitney_edge_1form",
    )

    # 3. Exact midpoint quadrature and anisotropic conduction across a disk.
    mesh, vertices = disk(disk_radius, n_disk_divisions)
    operators = Operators(mesh, vertices)
    fibre = (mv.xy * (-fibre_angle / 2)).exp() >> mv.x.dual()                # [] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    directions = stack((fibre, quarter_turn >> fibre))                       # [axes] Point-
    conductivity = (directions * (Plane & directions) * principal_conductivities).sum(axis=-1)  # [cases] Point- <- Plane+

    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    abscissa = (mv.x & vertices).batch()                                     # [vertices] Scalar+
    electrodes = boundary[abscissa[boundary].norm() > electrode_threshold]
    pinned = SparseExtensor.selection(mv.context, electrodes, mesh.n_vertices)
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

    shared_edge = left_corners[1] - left_corners[0]
    seam_points = left_corners[0] * 0.35 + left_corners[1] * 0.65
    seam_left = edge_whitney_forms(left_corners, seam_points[None])[2]
    seam_right = -edge_whitney_forms(right_corners, seam_points[None])[2]
    np.testing.assert_allclose((seam_left & shared_edge).kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose((seam_right & shared_edge).kernel, 1.0, atol=1e-14)

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
    shared_cochain = as_scalar(np.arange(len(obtuse_mesh.edges)) == 0).field()
    assert -(obtuse_edge_conductance(shared_cochain) & shared_cochain).sites.sum() > 0

    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~SparseExtensor.selection(mv.context, free_vertices, mesh.n_vertices) * Scalar
    np.testing.assert_allclose(free.adjugate()(conductance(potential)).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
