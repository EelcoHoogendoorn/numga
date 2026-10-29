"""Stokes flow on a triangle mesh and its circumcentric dual.

Edge fluxes balance on every triangle. Their Hodge dual measures circulation along
the dual edges, and its boundary sum gives vorticity on the dual cells around vertices.
Viscous resistance balances the pressure that enforces zero net flux. Incidences contain
only signed connectivity; lengths and areas enter through separate Hodge maps.

Integrated primal fluxes are scalar fields on edges, while the circumcentric dual cells
carry oriented area patches from the regressive product of vertex displacements. Summing
dual-edge circulation around each dual polygon and dividing by its area gives the vertex
vorticity bivector, whose edge differences drive the viscous response.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag

from examples import instantiate
from examples.fluids.dec_stokes.numerical import solve_stokes

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-
Amount = ga.gatype.antiscalar()                                              # Antiscalar-


# --- math -----------------------------------------------------------------------------
class Operators:
    """Topological exterior derivatives and circumcentric Hodge maps relative to the mesh pseudo-volume."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        # Integrated cochains carry one value per cell. These derivatives contain
        # only the incidence signs; their composition is zero before any geometry.
        self.d0 = mesh.d[0] * Scalar                                         # (Scalar+ <- Scalar+)[edges <- vertices]
        self.d1 = mesh.d[1] * Scalar                                         # (Scalar+ <- Scalar+)[faces <- edges]
        self.d0_vorticity = mesh.d[0] * Bivector                             # (Bivector+ <- Bivector+)[edges <- vertices]
        self.edges = mesh.d[0] * vertices                                    # Point-[edges]
        self.corners = mesh.incidence[2, 0] * vertices                       # [corners] Point-[faces]
        a, b, c = self.corners
        u, v = b - a, c - a
        self.face_patches = (u & v) / 2                                      # Patch-[faces]
        self.areas = self.face_patches.norm()                                # Scalar+[faces]
        self.face_centres = self.corners.sum(axis=0) / 3                     # Point-[faces]

        # The triangle's signed area patch turns the squared-length-weighted edge difference
        # into the offset equidistant from all three corners.
        weighted = v.norm_squared() * u - u.norm_squared() * v               # Point-[faces]
        self.circumcentres = a + weighted.commutator(self.face_patches.inverse().dual()) / 4  # Point-[faces]

        one_face = mv.scalar(np.ones((len(mesh.faces), 1))).field()          # Scalar+[faces]
        midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices             # Point-[edges]
        boundary = ~mesh.d[1] * one_face                                     # Scalar+[edges]
        # Interior dual edges join adjacent circumcentres. At the boundary they
        # finish at the primal edge's midpoint, closing the truncated dual cells.
        # The dual edge crosses to the primal edge's right, matching positive flux.
        self.dual_edges = boundary * midpoints - ~mesh.d[1] * self.circumcentres  # Point-[edges]

        # Oriented dual/primal edge ratios and dual cell patches from the quarter-kites
        # incident to each vertex.
        self.edge_ratios = (self.dual_edges & self.edges) / self.edges.norm_squared()  # Patch-[edges]
        self.dual_patches = ~mesh.incidence[1, 0].sum(axis=0) * ((self.dual_edges & self.edges) / 4)  # Patch-[vertices]
        self.dual_areas = self.dual_patches.norm()                           # Scalar+[vertices]

        # Each Hodge map scales by the oriented patch of the complementary dual cell
        # divided by that of its primal cell.
        self.star0 = spdiag(self.dual_patches * Scalar)                      # (Patch- <- Scalar+)[vertices <- vertices]
        self.star1 = spdiag(self.edge_ratios * Scalar)                       # (Patch- <- Scalar+)[edges <- edges]
        self.star2 = spdiag(self.face_patches.inverse() * Scalar)            # (Patch- <- Scalar+)[faces <- faces]
        self.inverse_star0 = spdiag(Patch.dual_inverse() / self.dual_areas)  # (Bivector+ <- Patch-)[vertices <- vertices]

        # The adjugate sums dual-edge circulation around each dual cell.
        # Dividing by that cell's area gives the vertex vorticity bivector directly.
        self.curl = self.inverse_star0(self.d0_vorticity.adjugate()(self.star1))  # (Bivector+ <- Scalar+)[vertices <- edges]


def stokes(operators: Operators, viscosity: float, prescribed_flux: Scalar,
           fixed_edges: np.ndarray, boundary_circulation: Patch
           ) -> tuple[Scalar, Scalar, Bivector, Point]:
    """Steady flow with prescribed boundary fluxes and tangential boundary circulation.

    Flux is positive through the right side of a directed edge. Pressure is determined
    up to a constant; the returned field has zero area-weighted mean.
    """
    mesh = operators.mesh
    # Vorticity differences across primal edges drive the paired edge response
    # through the adjugate of the dual-edge Hodge map.
    viscous = viscosity * operators.star1.adjugate()(operators.d0_vorticity(operators.curl))  # (Amount- <- Scalar+)[edges <- edges]
    boundary_vorticity = operators.inverse_star0(boundary_circulation)       # Bivector+[vertices]
    boundary_resistance = viscosity * operators.star1.adjugate()(operators.d0_vorticity(boundary_vorticity))  # Amount-[edges]

    free_edges = np.setdiff1d(np.arange(len(mesh.edges)), fixed_edges)
    embedding = ~SparseExtensor.selection(mv.context, free_edges, len(mesh.edges)) * Scalar  # (Scalar+ <- Scalar+)[edges <- free_edges]
    stiffness = embedding.adjugate()(viscous(embedding))                     # (Amount- <- Scalar+)[free_edges <- free_edges]
    divergence = operators.d1(embedding)                                     # (Scalar+ <- Scalar+)[faces <- free_edges]
    force = -embedding.adjugate()(viscous(prescribed_flux) + boundary_resistance)  # Amount-[free_edges]
    balance = -operators.d1(prescribed_flux)                                 # Scalar+[faces]
    free_flux, pressure = solve_stokes(stiffness, divergence, force, balance)  # Scalar+[free_edges], Scalar+[faces]
    flux = prescribed_flux + embedding(free_flux)                            # Scalar+[edges]
    pressure = pressure - (pressure * operators.areas).sites.sum() / operators.areas.sites.sum()  # Scalar+[faces]
    vorticity = operators.curl(flux) + boundary_vorticity                    # Bivector+[vertices]
    # The first moment of the outward flux gives the face's mean velocity.
    outward_flux = mesh.boundary[2, 1] * flux                                # [corners] Scalar+[faces]
    velocity = -((operators.corners - operators.face_centres) * outward_flux).sum(axis=0) / (2 * operators.areas)  # Point-[faces]
    return flux, pressure, vorticity, velocity
