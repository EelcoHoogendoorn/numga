"""Stokes flow on a triangle mesh and its circumcentric dual.

Edge fluxes balance on every triangle. Their Hodge dual measures circulation along
the dual edges, and its boundary sum gives vorticity on the dual cells around vertices.
Viscous resistance balances the pressure that enforces zero net flux. Incidences contain
only signed connectivity; lengths and areas enter through separate Hodge maps.

Integrated primal values are scalar fields and their duals are antiscalar fields,
paired by the regressive product. Their form degree belongs to the cells they occupy:
vertices, edges or faces, rather than to the ambient blade grade of each stored value.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import VGA2D as ga
from numga.sparse import SparseExtensor, spdiag

from examples import instantiate
from examples.fluids.dec_stokes.numerical import solve_stokes

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()
Bivector = ga.gatype.bivector()
Point = ga.gatype.antivector()
Amount = ga.gatype.antiscalar()


# --- math -----------------------------------------------------------------------------
class Operators:
    """Topological exterior derivatives and separate circumcentric Hodge maps."""

    def __init__(self, mesh: Mesh, vertices: Point):
        self.mesh, self.vertices = mesh, vertices
        # Integrated cochains carry one value per cell. These derivatives contain
        # only the incidence signs; their composition is zero before any geometry.
        self.d0 = mesh.d0 * Scalar                                           # Scalar[E] <- Scalar[V]
        self.d1 = mesh.d1 * Scalar                                           # Scalar[F] <- Scalar[E]
        self.edges = mesh.d0 * vertices                                      # Point[E]
        corners = vertices.batch()[mesh.faces.T].field()                     # [corners] Point[F]
        a, b, c = corners
        u, v = b - a, c - a
        self.area_blades = (u & v) / 2                                       # Antibivector[F]
        self.areas = self.area_blades.norm()                                  # Scalar[F]
        self.sense = (self.area_blades / self.areas).sites.mean()             # Antibivector
        self.face_centres = corners.sum(axis=0) / 3                          # Point[F]

        # The triangle's area turns the squared-length-weighted edge difference
        # into the offset equidistant from all three corners.
        weighted = v.norm_squared() * u - u.norm_squared() * v
        self.circumcentres = a + weighted.commutator(self.area_blades.dual()) / (4 * self.areas.squared())

        one_face = mv.scalar(np.ones((len(mesh.faces), 1))).field()
        tail, head = mesh.edge_ends
        midpoints = (tail * vertices + head * vertices) / 2                   # Point[E]
        boundary = ~mesh.d1 * one_face                                       # Scalar[E]
        # Interior dual edges join adjacent circumcentres. At the boundary they
        # finish at the primal edge's midpoint, closing the truncated dual cells.
        # The dual edge crosses to the primal edge's right, matching positive flux.
        self.dual_edges = boundary * midpoints - ~mesh.d1 * self.circumcentres  # Point[E]

        # Signed dual/primal edge-length ratios, and the dual areas made of
        # the quarter-kites incident to each vertex.
        self.h1 = (self.dual_edges & self.edges) / (self.sense * self.edges.norm_squared())
        self.dual_areas = (~tail + ~head) * (self.h1 * self.edges.norm_squared() / 4)

        # Each Hodge map changes orientation and scales by the measure of the
        # complementary dual cell divided by that of its primal cell.
        self.star0 = spdiag(self.dual_areas * Scalar.dual())                  # Amount[V] <- Scalar[V]
        self.star1 = spdiag(self.h1 * Scalar.dual())                          # Amount[E] <- Scalar[E]
        self.star2 = spdiag(Scalar.dual() / self.areas)                       # Amount[F] <- Scalar[F]
        self.inverse_star0 = spdiag(Amount.dual_inverse() / self.dual_areas)  # Scalar[V] <- Amount[V]

        # The adjugate sums dual-edge circulation around each dual cell.
        # Dividing by that cell's area gives its vorticity density.
        self.curl = self.inverse_star0(self.d0.adjugate()(self.star1))        # Scalar[V] <- Scalar[E]


def stokes(operators: Operators, viscosity: float, prescribed_flux: Scalar,
           fixed_edges: np.ndarray, boundary_circulation: Scalar
           ) -> tuple[Scalar, Scalar, Bivector, Point]:
    """Steady flow with prescribed boundary fluxes and tangential boundary circulation.

    Flux is positive through the right side of a directed edge. Pressure is determined
    up to a constant; the returned field has zero area-weighted mean.
    """
    mesh = operators.mesh
    # Vorticity differences drive viscous resistance. The Hodge map supplies
    # the paired edge response, whose work is the fluid's dissipation.
    viscous = viscosity * operators.star1(operators.d0(operators.curl))      # Amount[E] <- Scalar[E]
    boundary_vorticity = boundary_circulation / operators.dual_areas         # Scalar[V]
    boundary_resistance = viscosity * operators.star1(operators.d0(boundary_vorticity))

    free_edges = np.setdiff1d(np.arange(len(mesh.edges)), fixed_edges)
    embedding = ~SparseExtensor.selection(mv.context, free_edges, len(mesh.edges)) * Scalar
    stiffness = embedding.adjugate()(viscous(embedding))                    # Amount[free] <- Scalar[free]
    divergence = operators.d1(embedding)                                   # Scalar[F] <- Scalar[free]
    force = -embedding.adjugate()(viscous(prescribed_flux) + boundary_resistance)
    balance = -operators.d1(prescribed_flux)                                # Scalar[F]
    unknown, pressure = solve_stokes(stiffness, divergence, force, balance)
    flux = prescribed_flux + embedding(unknown)                             # Scalar[E]
    pressure = pressure - (pressure * operators.areas).sites.sum() / operators.areas.sites.sum()
    vorticity = operators.sense.dual_inverse() * (operators.curl(flux) + boundary_vorticity)  # Bivector[V]
    # The first moment of the outward flux gives the face's mean velocity.
    corners = operators.vertices.batch()[mesh.faces.T].field()
    outward = (flux.batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()
    velocity = -((corners - operators.face_centres) * outward).sum(axis=0) / (2 * operators.areas)
    return flux, pressure, vorticity, velocity
