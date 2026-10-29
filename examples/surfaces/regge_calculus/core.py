"""Triangle edge lengths determine a surface's intrinsic geometry and the current it carries.

A metric is a correlation from displacements to planes: `metric(edge) & edge` reads an edge's squared length. Symmetric products of corner gradients reconstruct that correlation. Corner turns record curvature at the vertices, and the gradient's adjugate assembles the local current response into steady conduction on the mesh.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar, mv = mesh_module.Mesh, mesh_module.as_scalar, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Plane = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Rotor = ga.gatype.rotor()                                                    # Rotor+
Metric = ga.gatype((Plane, Point))                                           # Vector+ <- Antivector-
Conductivity = ga.gatype((Point, Plane))                                     # Antivector- <- Vector+


# --- math -----------------------------------------------------------------------------
def regge_metric_basis(corners: Point) -> tuple[Metric, Plane, Patch]:
    """Each edge's correlation reads one on that edge and zero on the other two; its corner gradients and signed patch define the reference triangle."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    gradients = sides / (2 * patch)                                          # [corners] Plane+
    # Along the two sides meeting at a corner, one endpoint gradient vanishes while on the opposite
    # side the two endpoint gradients pair to -1 and +1, making their symmetric product dual to the squared edge lengths.
    first, second = gradients[[1, 2, 0]], gradients[[2, 0, 1]]               # [corners] Plane+, [corners] Plane+
    basis = -(first * (second & Point) + second * (first & Point)) / 2       # [corners] Plane+ <- Point-
    return basis, gradients, patch


def triangle_geometry(corners: Point, squared_lengths: Scalar
                      ) -> tuple[Metric, Scalar, Rotor, Bivector]:
    """Prescribed squared edge lengths determine the triangle's correlation, area, unit corner rotors, and corner turns."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    basis, _, patch = regge_metric_basis(corners)                            # [corners] Plane+ <- Point-, [corners] Plane+, [] Patch-
    metric = (squared_lengths * basis).sum(axis=0)                           # [] Plane+ <- Point-
    # The wedge carries the correlation's area scaling; pairing it with the reference patch
    # and dividing by two gives the intrinsic area squared.
    intrinsic_area = (((metric(sides[0]) ^ metric(sides[1])) & patch) / 2).square_root()  # [] Scalar+
    # Combining the metric inner product of the two sides meeting at each corner with twice the intrinsic
    # area in the even subalgebra forms a rotor whose norm is the product of those two side lengths.
    dots = -(metric(sides[[1, 2, 0]]) & sides[[2, 0, 1]])                    # [corners] Scalar+
    corner_rotors = (dots + 2 * intrinsic_area * mv.xy).normalized()         # [corners] Rotor+
    corner_turns = corner_rotors.log()                                       # [corners] Bivector+
    return metric, intrinsic_area, corner_rotors, corner_turns


def mesh_geometry(mesh: Mesh, vertices: Point, edge_squared_lengths: Scalar
                  ) -> tuple[Metric, Scalar, Bivector, Bivector, Conductivity]:
    """The edge lengths define face correlations, areas, vertex angle defects, area-normalized defects, and local conductivities."""
    edges = mesh.d[0] * vertices                                             # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                      # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    corner_gradients = sides / (2 * patches)                                 # [corners] Plane+[faces]
    first, second = corner_gradients[[1, 2, 0]], corner_gradients[[2, 0, 1]]  # [corners] Plane+[faces], [corners] Plane+[faces]
    basis = -(first * (second & Point) + second * (first & Point)) / 2       # [corners] (Plane+ <- Point-)[faces]

    metrics = (spdiag(basis) * mesh.incidence[2, 1]).sum(axis=0) * edge_squared_lengths  # [...] (Plane+ <- Point-)[faces]
    intrinsic_areas = (((metrics(sides[0]) ^ metrics(sides[1])) & patches) / 2).square_root()  # [...] Scalar+[faces]
    dots = -(metrics[..., None](sides[[1, 2, 0]]) & sides[[2, 0, 1]])        # [..., corners] Scalar+[faces]
    corner_rotors = (dots + 2 * intrinsic_areas[..., None] * mv.xy).normalized()  # [..., corners] Rotor+[faces]
    corner_turns = corner_rotors.log()                                       # [..., corners] Bivector+[faces]

    # Each vertex holds a third of the intrinsic area of the triangles around it. The corner turns, carried from
    # the faces back to their vertices and summed, fall short of the flat turn by the defect.
    corner_mean = mesh.incidence[2, 0].mean(axis=0)                          # Scalar+[faces <- vertices]
    vertex_areas = ~corner_mean * intrinsic_areas                            # [...] Scalar+[vertices]
    boundary = np.unique(mesh.edges[mesh.boundary_edges])
    is_boundary = np.isin(np.arange(mesh.n_vertices), boundary)
    target_turns = as_scalar(np.where(is_boundary, np.pi, 2 * np.pi)).field() * mv.xy  # Bivector+[vertices]
    defects = target_turns - (~mesh.incidence[2, 0] * corner_turns).sum(axis=-1)  # [...] Bivector+[vertices]
    curvatures = defects / vertex_areas                                      # [...] Bivector+[vertices]

    # Each side dyad turns a gradient reading into flow along that side;
    # the corner cotangents weight the response of the intrinsic triangle.
    cotangents = dots / (2 * intrinsic_areas[..., None])                     # [..., corners] Scalar+[faces]
    conductivities = ((cotangents / 2) * (sides * (Plane & sides))).sum(axis=-1)  # [...] (Point- <- Plane+)[faces]
    return metrics, intrinsic_areas, defects, curvatures, conductivities


def intrinsic_conduction(mesh: Mesh, vertices: Point, conductivities: Conductivity,
                         electrodes: np.ndarray, prescribed: Scalar
                         ) -> tuple[Scalar, Point, Scalar]:
    """Prescribed electrode potentials determine a steady potential, reference-frame flux density, and streamfunction."""
    edges = mesh.d[0] * vertices                                             # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                      # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    areas = patches.norm()                                                   # Scalar+[faces]
    corner_gradients = sides / (2 * patches)                                 # [corners] Plane+[faces]
    gradient = (spdiag(corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

    # The adjugate carries each triangle's current response back to its vertex potentials.
    conductance = gradient.adjugate()(spdiag(conductivities)(gradient))      # [cases] (Amount- <- Scalar+)[vertices <- vertices]
    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~SparseExtensor.selection(mv.context, free_vertices, mesh.n_vertices) * Scalar  # (Scalar+ <- Scalar+)[vertices <- free]
    rhs = -free.adjugate()(conductance(prescribed))                          # [cases] Amount-[free]
    potential = prescribed + free(free.adjugate()(conductance(free)).solve(rhs))  # [cases] Scalar+[vertices]
    integrated_flux = -conductivities(gradient(potential))                   # [cases] Point-[faces]
    current = integrated_flux / areas                                        # [cases] Point-[faces]

    # Because the area-integrated flux is divergence-free on the reference mesh, turning it by a quarter turn
    # gives the area-weighted reference gradient of the streamfunction whose level curves follow the intrinsic flow.
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    laplacian = gradient.adjugate()(spdiag(areas * Plane.dual())(gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    unpinned = ~SparseExtensor.selection(mv.context, np.arange(1, mesh.n_vertices), mesh.n_vertices) * Scalar  # (Scalar+ <- Scalar+)[vertices <- unpinned]
    stream_rhs = unpinned.adjugate()(gradient.adjugate()(quarter_turn >> integrated_flux))  # [cases] Amount-[unpinned]
    streamfunction = unpinned(unpinned.adjugate()(laplacian(unpinned)).solve(stream_rhs))  # [cases] Scalar+[vertices]
    return potential, current, streamfunction
