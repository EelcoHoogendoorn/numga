"""A reference triangle and disk carry prescribed intrinsic edge lengths."""

from __future__ import annotations

import numpy as np

from numga import concatenate, stack
from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 24
n_disk_divisions = 9
disk_radius = 1.0
electrode_threshold = 0.78
edge_stretches = np.array([1.45, 0.65, 1.25])                                # [corners]
radial_rates = np.array([0.0, 3.2, -0.45])                                   # [cases]
hoop_rates = np.array([0.0, -0.78, 3.2])                                     # [cases]


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """The samples form a barycentric lattice across the triangle."""
    steps = np.arange(n_subdivisions + 1)
    first, second = np.meshgrid(steps, steps, indexing="ij")
    inside = first + second <= n_subdivisions
    weights = np.stack((n_subdivisions - first[inside] - second[inside],
                        first[inside], second[inside]), axis=-1) / n_subdivisions  # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-


def single_triangle(n_subdivisions: int) -> tuple[Point, Point]:
    """The reference triangle is scalene, with its sample points across it."""
    horizontal, vertical = mv.x.dual(), mv.y.dual()                          # [] Point-, [] Point-
    corners = stack((horizontal * 0 + vertical * 0,
                     horizontal * 1.2 + vertical * 0.12,
                     horizontal * 0.38 + vertical * 0.96))                   # [corners] Point-
    return corners, triangle_samples(corners, n_subdivisions)


def disk(radius: float, n_divisions: int) -> tuple[Mesh, Point]:
    """The disk is triangulated into nearly equilateral triangles, with its first vertex at the origin."""
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


def main() -> None:
    from examples.surfaces.regge_calculus.core import (
        intrinsic_conduction, mesh_geometry, regge_metric_basis, triangle_geometry,
    )
    from examples.surfaces.regge_calculus import render
    from examples.util.animation import save_figure

    # 1. Regge edge metric basis, reconstructed metric ellipse, and corner rotor turns.
    corners, points = single_triangle(n_triangle_subdivisions)
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    squared_lengths = sides.norm_squared() * edge_stretches                  # [corners] Scalar+
    basis, gradients, patch = regge_metric_basis(corners)
    metric, intrinsic_area, corner_rotors, corner_turns = triangle_geometry(corners, squared_lengths)
    save_figure(
        render.draw_metric_basis(corners, points, basis, metric),
        "regge_metric_basis",
    )

    # 2. Intrinsic Gaussian curvature defects and Gauss-Bonnet from edge lengths on a disk.
    mesh, vertices = disk(disk_radius, n_disk_divisions)
    edges = mesh.d[0] * vertices                                             # Point-[edges]
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices                 # Point-[edges]
    # The dot reading measures radial motion times radius; the join measures swept area,
    # hence circumferential motion times radius. Their squares choose the two length responses.
    edge_squared_lengths = (
        edges.norm_squared()
        + radial_rates * midpoints.dot(edges).squared()
        + hoop_rates * (midpoints & edges).norm_squared()
    )                                                                        # [cases] Scalar+[edges]
    centres = mesh.incidence[2, 0].mean(axis=0) * vertices                   # Point-[faces]
    metrics, intrinsic_areas, defects, curvatures, conductivities = mesh_geometry(
        mesh, vertices, edge_squared_lengths,
    )
    save_figure(
        render.draw_curvature_defects(mesh, vertices, centres, metrics, curvatures),
        "regge_curvature_defects",
    )

    # 3. Steady conduction across flat, radially stretched, and circumferentially stretched geometry.
    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    abscissa = (mv.x & vertices).batch()                                     # [vertices] Scalar+
    electrodes = boundary[abscissa[boundary].norm() > electrode_threshold]
    pinned = SparseExtensor.selection(mv.context, electrodes, mesh.n_vertices)
    prescribed = ~pinned * (pinned * (mv.x & vertices))                      # Scalar+[vertices]
    potential, current, streamfunction = intrinsic_conduction(
        mesh, vertices, conductivities, electrodes, prescribed,
    )
    save_figure(
        render.draw_intrinsic_flow(mesh, vertices, centres, potential, streamfunction, current),
        "regge_intrinsic_laplacian",
    )

    # checks
    side_readings = basis[:, None](sides[None, :]) & sides[None, :]
    np.testing.assert_allclose(side_readings.kernel, np.eye(3)[..., None], atol=1e-14)
    np.testing.assert_allclose(((metric(sides) & sides) - squared_lengths).kernel, 0.0, atol=1e-14)

    euclidean_metric, euclidean_area, _, _ = triangle_geometry(corners, sides.norm_squared())
    np.testing.assert_allclose((euclidean_metric - Point.dual_inverse()).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((euclidean_area - patch.norm()).kernel, 0.0, atol=1e-14)

    lengths = np.sqrt(squared_lengths.kernel[..., 0])
    semiperimeter = lengths.sum() / 2
    heron_area = np.sqrt(semiperimeter * np.prod(semiperimeter - lengths))
    np.testing.assert_allclose(intrinsic_area.kernel, heron_area, atol=1e-14)
    np.testing.assert_allclose(corner_rotors.scalar_norm_squared().kernel, 1.0, atol=1e-14)
    np.testing.assert_allclose((corner_turns.sum(axis=0) - np.pi * mv.xy).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose((corner_rotors[0] * corner_rotors[1] * corner_rotors[2]).kernel, [-1.0, 0.0], atol=1e-14)

    np.testing.assert_allclose((defects.sites.sum() - 2 * np.pi * mv.xy).kernel, 0.0, atol=1e-12)
    is_boundary = np.isin(np.arange(mesh.n_vertices), boundary)
    np.testing.assert_allclose(defects.batch()[0, ~is_boundary].kernel, 0.0, atol=1e-13)
    assert (curvatures.batch()[1, 0] & mv.xy.dual()) > 0
    assert (curvatures.batch()[2, 0] & mv.xy.dual()) < 0

    inverse_conductivities = intrinsic_areas * metrics.batch().inverse().field()
    np.testing.assert_allclose((conductivities - inverse_conductivities).kernel, 0.0, atol=1e-13)

    face_sides = mesh.boundary[2, 1] * edges
    face_patches = (face_sides[0] & face_sides[1]) / 2
    gradient = (spdiag(face_sides / (2 * face_patches)) * mesh.incidence[2, 0]).sum(axis=0) * Scalar
    conductance = gradient.adjugate()(spdiag(conductivities)(gradient))
    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~SparseExtensor.selection(mv.context, free_vertices, mesh.n_vertices) * Scalar
    np.testing.assert_allclose(free.adjugate()(conductance(potential)).kernel, 0.0, atol=1e-11)


if __name__ == "__main__":
    main()
