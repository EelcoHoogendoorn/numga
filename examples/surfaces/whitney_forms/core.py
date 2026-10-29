"""Whitney forms interpolate vertex and edge data across triangle meshes.

The triangle's signed patch defines both corner gradients and edge forms. Their reconstruction maps let a material conductivity act on local fields, with the adjugate carrying its response back to the mesh.
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
Conductivity = ga.gatype((Point, Plane))


# --- math -----------------------------------------------------------------------------
def barycentric_hats(corners: Point, points: Point) -> tuple[Scalar, Plane]:
    """The barycentric hat values at `points` and the constant corner gradients."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Dividing each opposite edge by twice the signed area patch gives the gradient
    # that falls by one from its corner to the opposite edge.
    gradients = sides / (2 * patch)                                          # [corners] Plane+
    hats = 1 + (gradients[:, None] & (points[None, :] - corners[:, None]))   # [corners, samples] Scalar+
    return hats, gradients


def edge_whitney_forms(corners: Point, points: Point) -> Plane:
    """The three edge Whitney 1-forms evaluated at `points`, circulating around each edge's opposite corner."""
    sides = corners[[2, 0, 1]] - corners[[1, 2, 0]]                          # [corners] Point-
    patch = (sides[0] & sides[1]) / 2                                        # [] Patch-
    # Along the two edges meeting at a corner, the radial displacement from that corner is parallel
    # to the edge, while along the opposite edge its join with the edge equals twice the triangle patch.
    return (points[None, :] - corners[:, None]) / (2 * patch)                # [corners, samples] Plane+


class Operators:
    """Topological exterior derivatives and Whitney reconstruction maps on an oriented triangle mesh."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh = mesh
        self.d0 = mesh.d[0] * Scalar                                         # (Scalar+ <- Scalar+)[edges <- vertices]
        self.d1 = mesh.d[1] * Scalar                                         # (Scalar+ <- Scalar+)[faces <- edges]
        edges = mesh.d[0] * vertices                                         # Point-[edges]
        self.corners = mesh.incidence[2, 0] * vertices                       # [corners] Point-[faces]
        self.sides = mesh.boundary[2, 1] * edges                             # [corners] Point-[faces]
        self.patches = (self.sides[0] & self.sides[1]) / 2                   # Patch-[faces]
        self.areas = self.patches.norm()                                     # Scalar+[faces]
        self.centres = self.corners.sum(axis=0) / 3                          # Point-[faces]

        # Each corner hat's gradient is constant across its face; assembling by corner maps
        # vertex potentials to piecewise-constant face gradients without any metric.
        self.corner_gradients = self.sides / (2 * self.patches)              # [corners] Plane+[faces]
        self.gradient = (spdiag(self.corner_gradients) * mesh.incidence[2, 0]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- vertices]

        # At each face centroid the three barycentric hats equal 1/3, so the edge 1-form
        # reduces to one third of the difference of its endpoints' corner gradients.
        centroid_forms = (self.corner_gradients[[2, 0, 1]] - self.corner_gradients[[1, 2, 0]]) / 3  # [corners] Plane+[faces]
        self.whitney1 = (spdiag(centroid_forms) * mesh.boundary[2, 1]).sum(axis=0) * Scalar  # (Plane+ <- Scalar+)[faces <- edges]

        # The exterior derivative of each edge's Whitney 1-form is twice the wedge of its endpoints'
        # gradients, matching the constant Whitney 2-form of unit integral over the face patch.
        first, second = self.corner_gradients[[1, 2, 0]], self.corner_gradients[[2, 0, 1]]  # [corners] Plane+[faces], [corners] Plane+[faces]
        self.curl = (spdiag(2 * (first ^ second)) * mesh.boundary[2, 1]).sum(axis=0) * Scalar  # (Bivector+ <- Scalar+)[faces <- edges]
        self.whitney2 = spdiag(mv.xy / (mv.xy & self.patches) * Scalar)      # (Bivector+ <- Scalar+)[faces <- faces]

        # Evaluating the linear Whitney 1-forms at the three edge midpoints integrates
        # quadratic products over every triangle exactly.
        midpoints = mesh.incidence[2, 1] * (mesh.incidence[1, 0].mean(axis=0) * vertices)  # [midpoints] Point-[faces]
        midpoint_forms = (midpoints[:, None] - self.corners[None, :]) / (2 * self.patches)  # [midpoints, corners] Plane+[faces]
        self.midpoint_whitney1 = (spdiag(midpoint_forms) * mesh.boundary[2, 1]).sum(axis=1) * Scalar  # [midpoints] (Plane+ <- Scalar+)[faces <- edges]


def edge_conductance(operators: Operators, conductivity: Conductivity) -> SparseExtensor:
    """The 1-form conductance operator on edges, integrated exactly over each triangle from its three edge midpoints."""
    # Each midpoint carries one third of the triangle's area; pulling the material map back
    # through the midpoint evaluation maps and summing over the three midpoints is exact for quadratics.
    weights = spdiag(((operators.areas / 3) * conductivity)[..., None])      # [..., midpoints] (Point- <- Plane+)[faces <- faces]
    return operators.midpoint_whitney1.adjugate()(weights(operators.midpoint_whitney1)).sum(axis=-1)  # [...] (Amount- <- Scalar+)[edges <- edges]


def conduction(operators: Operators, conductivity: Conductivity,
               electrodes: np.ndarray, prescribed: Scalar
               ) -> tuple[Scalar, Point, Scalar]:
    """Steady potential, face current, and streamfunction for prescribed electrode potentials."""
    mesh = operators.mesh
    # Pulling the area-weighted conductivity back through the face gradient gives the vertex conductance.
    conductance = operators.gradient.adjugate()(spdiag(operators.areas * conductivity)(operators.gradient))  # [cases] (Amount- <- Scalar+)[vertices <- vertices]
    free_vertices = np.setdiff1d(np.arange(mesh.n_vertices), electrodes)
    free = ~mesh.reading(free_vertices) * Scalar                             # (Scalar+ <- Scalar+)[vertices <- free]
    rhs = -free.adjugate()(conductance(prescribed))                          # [cases] Amount-[free]
    potential = prescribed + free(free.adjugate()(conductance(free)).solve(rhs))  # [cases] Scalar+[vertices]
    current = -conductivity(operators.gradient(potential))                   # [cases] Point-[faces]

    # Turning the face current by a quarter turn gives, through the unit isotropic conductivity `Plane.dual()`,
    # the target gradient of the streamfunction whose level curves follow the current paths.
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                              # [] Rotor+
    laplacian = operators.gradient.adjugate()(spdiag(operators.areas * Plane.dual())(operators.gradient))  # (Amount- <- Scalar+)[vertices <- vertices]
    unpinned = ~mesh.reading(np.arange(1, mesh.n_vertices)) * Scalar         # (Scalar+ <- Scalar+)[vertices <- unpinned]
    stream_rhs = unpinned.adjugate()(operators.gradient.adjugate()(operators.areas * (quarter_turn >> current)))  # [cases] Amount-[unpinned]
    streamfunction = unpinned(unpinned.adjugate()(laplacian(unpinned)).solve(stream_rhs))  # [cases] Scalar+[vertices]
    return potential, current, streamfunction
