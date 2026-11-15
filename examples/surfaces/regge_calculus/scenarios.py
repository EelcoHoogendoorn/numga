"""A reference triangle and triangulated disk carry prescribed intrinsic edge lengths."""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag
from examples.util import shapes

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Point = ga.gatype.antivector()                                               # Antivector-

n_triangle_subdivisions = 24
n_disk_divisions = 9
disk_radius = 1.0
electrode_threshold = 0.78
edge_stretches = np.array([1.25, 0.65, 1.45])                                # [3]
radial_rates = np.array([0.0, 3.2, -0.45])                                   # [3]
hoop_rates = np.array([0.0, -0.78, 3.2])                                     # [3]


# --- plumbing -------------------------------------------------------------------------
def triangle_samples(corners: Point, n_subdivisions: int) -> Point:
    """Convex combinations of `corners` on a regular grid form interior sample points (`[...] Point-`)."""
    grid_steps = np.arange(n_subdivisions + 1)
    first, second = np.meshgrid(grid_steps, grid_steps, indexing="ij")
    in_triangle = first + second <= n_subdivisions
    weights = np.stack((n_subdivisions - first[in_triangle] - second[in_triangle],
                        first[in_triangle], second[in_triangle]), axis=-1) / n_subdivisions  # [..., 3]
    return (corners.batch() * weights).sum(axis=-1)                          # [...] Point-


def single_triangle(n_subdivisions: int) -> tuple[Point, Point]:
    """A scalene reference triangle (`Point-[V]`) pairs with its interior sample lattice (`[...] Point-`)."""
    coordinates = np.array([[0.0, 0.0], [1.2, 0.12], [0.38, 0.96]])          # [3, 2]
    corners = mv.vector(coordinates).dual().field()                          # Point-[V]
    return corners, triangle_samples(corners, n_subdivisions)


def main() -> None:
    from examples.surfaces.regge_calculus.core import (
        corner_rotors, intrinsic_areas, metric_basis, simplex_gradient,
        steady_potential, triangle, triangle_conductivities, vertex_defects,
    )
    from examples.surfaces.regge_calculus import render
    from examples.util.animation import save_figure

    # 1. Regge edge-metric basis, reconstructed metric ellipse, and corner rotor turns.
    corners, sample_points = single_triangle(n_triangle_subdivisions)
    triangle_basis = metric_basis(triangle, corners)                         # [3] (Plane+ <- Point-)[F]
    triangle_squared_lengths = (
        (triangle.d[0] * corners).norm_squared() * mv.scalar(edge_stretches[:, None]).field()
    )                                                                        # Scalar+[E]
    triangle_metric = (spdiag(triangle_basis) * triangle.incidence[2, 1]).sum(axis=0) * triangle_squared_lengths  # (Plane+ <- Point-)[F]
    triangle_area = intrinsic_areas(triangle, corners, triangle_metric)      # Scalar+[F]
    triangle_rotors = corner_rotors(triangle, corners, triangle_metric, triangle_area)  # [3] Rotor+[F]
    triangle_turns = triangle_rotors.log()                                   # [3] Bivector+[F]
    save_figure(
        render.metric_basis_figure(corners, sample_points, triangle_basis, triangle_metric),
        "regge_metric_basis",
    )

    # 2. Intrinsic Gaussian curvature defects and Gauss-Bonnet from edge lengths on a disk.
    topology, coordinates = shapes.hexagonal_disk(disk_radius, n_disk_divisions)
    V, E, F = topology.n_elements
    vertices = mv.vector(coordinates).dual().field()                         # Point-[V]
    edges = topology.d[0] * vertices                                         # Point-[E]
    midpoints = topology.incidence[1, 0].mean(axis=0) * vertices             # Point-[E]
    # The inner product measures radial motion times radius; the join measures swept area,
    # hence circumferential motion times radius. Their squares choose the two length responses.
    edge_squared_lengths = (
        edges.norm_squared()
        + radial_rates * (midpoints | edges).squared()
        + hoop_rates * (midpoints & edges).norm_squared()
    )                                                                        # [3] Scalar+[E]
    disk_basis = metric_basis(topology, vertices)                            # [3] (Plane+ <- Point-)[F]
    metrics = (spdiag(disk_basis) * topology.incidence[2, 1]).sum(axis=0) * edge_squared_lengths  # [3] (Plane+ <- Point-)[F]
    areas = intrinsic_areas(topology, vertices, metrics)                     # [3] Scalar+[F]
    turns = corner_rotors(topology, vertices, metrics, areas).log()          # [3, 3] Bivector+[F]
    defects = vertex_defects(topology, vertices, turns)                      # [3] Bivector+[V]
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas    # [3] Scalar+[V]
    curvatures = defects / dual_areas                                        # [3] Bivector+[V]
    centres = topology.incidence[2, 0].mean(axis=0) * vertices               # Point-[F]
    save_figure(
        render.curvature_defects_figure(topology, vertices, centres, metrics, curvatures),
        "regge_curvature_defects",
    )

    # 3. Steady conduction across flat, radially stretched, and circumferentially stretched geometry.
    conductivities = triangle_conductivities(topology, vertices, metrics, areas)  # [3] (Point- <- Plane+)[F]
    rim_indices = topology.exterior[0]
    rim = topology.reading(rim_indices, 0)                                   # Scalar+[V_rim <- V]
    electrodes = rim_indices[(mv.x & (rim * vertices)).abs() > electrode_threshold]
    electrode_reading = topology.reading(electrodes, 0)                      # Scalar+[V_electrodes <- V]
    boundary_potential = electrode_reading.adjugate() * (electrode_reading * (mv.x & vertices))  # Scalar+[V]
    free_indices = np.setdiff1d(np.arange(V), electrodes)
    gradient = simplex_gradient(topology, vertices)                          # (Plane+ <- Scalar+)[F <- V]
    conductance = gradient.adjugate()(conductivities(gradient))              # [3] (Antiscalar- <- Scalar+)[V <- V]
    potential = steady_potential(topology, conductance, free_indices, boundary_potential)  # [3] Scalar+[V]
    reference_sides = topology.boundary[2, 1] * edges                        # [3] Point-[F]
    reference_areas = ((reference_sides[0] & reference_sides[1]) / 2).antinorm()  # Scalar+[F]
    current = -conductivities(gradient(potential)) / reference_areas         # [3] Point-[F]
    save_figure(
        render.intrinsic_flow_figure(topology, vertices, potential, current),
        "regge_intrinsic_laplacian",
    )

    # checks
    triangle_sides = (triangle.boundary[2, 1] * (triangle.d[0] * corners)).batch()[:, 0]
    side_readings = triangle_basis.batch()[:, 0, None](triangle_sides[None, :]) & triangle_sides[None, :]
    assert ((side_readings - np.eye(3)).abs() < 1e-10).all()

    prescribed_side_lengths = (triangle.incidence[2, 1] * triangle_squared_lengths).batch()[:, 0]
    assert (((triangle_metric.batch()[0](triangle_sides) & triangle_sides) - prescribed_side_lengths).abs() < 1e-10).all()

    side_lengths = prescribed_side_lengths.square_root()
    semiperimeter = side_lengths.sum() / 2
    heron_area = (
        semiperimeter
        * (semiperimeter - side_lengths[0])
        * (semiperimeter - side_lengths[1])
        * (semiperimeter - side_lengths[2])
    ).square_root()
    assert ((triangle_area.batch()[0] - heron_area).abs() < 1e-10).all()
    assert ((triangle_rotors.batch()[:, 0].scalar_norm_squared() - 1).abs() < 1e-10).all()
    assert ((triangle_turns.batch()[:, 0].sum(axis=0) - np.pi * mv.xy).norm() < 1e-10).all()
    assert ((triangle_rotors.batch()[0, 0] * triangle_rotors.batch()[1, 0] * triangle_rotors.batch()[2, 0] + 1).norm() < 1e-10).all()

    assert ((defects.sites.sum() - 2 * np.pi * mv.xy).norm() < 1e-10).all()
    assert ((topology.reading(topology.interior[0], 0) * defects[0]).batch().norm() < 1e-10).all()
    assert (curvatures.batch()[1, 0] & mv.xy.dual()) > 0
    assert (curvatures.batch()[2, 0] & mv.xy.dual()) < 0

    basis_planes = stack((mv.x, mv.y))
    identity_residual = (
        metrics.batch()[..., None](conductivities.batch()[..., None](basis_planes))
        - areas.batch()[..., None] * basis_planes
    )
    assert (identity_residual.norm() < 1e-10).all()
    reciprocity = (
        (potential & conductance(boundary_potential)).sites.sum()
        - (boundary_potential & conductance(potential)).sites.sum()
    )
    assert (reciprocity.abs() < 1e-10).all()
    assert ((topology.reading(free_indices, 0) * conductance(potential)).batch().norm() < 1e-9).all()


if __name__ == "__main__":
    main()
