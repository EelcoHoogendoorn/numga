r"""Stokes flow through a triangulated channel and its circumcentric dual.

This example computes the slow viscous flow of a fluid through a channel past an offset circular obstacle, where inertia is negligible and pressure balances viscous resistance alone. The channel is modelled as a triangulated strip with a circumcentric dual: fluid crosses the edges as integrated fluxes, every triangle passes on exactly as much fluid as it receives, and the solid walls bring the fluid to rest.

Edge fluxes are scalar fields on the edges, and the incidences between vertices, edges and triangles carry only signs, so `d1(d0)` vanishes before any geometry enters. Joining a dual edge with the primal edge it crosses, `dual_edges & edges`, gives the oriented patch of their kite, and the diagonal Hodge maps scale by these patches, carrying the length ratios and the complementary orientation at once. Vorticity is a bivector at each vertex, `curl = inverse_star0(d0_vorticity.adjugate()(star1))`, and the viscous force is the same chain read back, `star1.adjugate()(d0_vorticity(curl))`, symmetric by construction. [Jagad et al.](https://arxiv.org/abs/2010.15520) give a pressure–velocity formulation of incompressible flow in discrete exterior calculus.

In the matrix notation of discrete exterior calculus, the edge fluxes $u$ and face pressures $p$ solve the saddle-point system $\begin{pmatrix}\mu\,\star_1 d_0 \star_0^{-1} d_0^{T} \star_1 & -d_1^{T} \\ -d_1 & 0\end{pmatrix}\begin{pmatrix}u \\ p\end{pmatrix} = \begin{pmatrix}f \\ g\end{pmatrix}$, where each $\star_k$ is the unsigned diagonal of ratios $|\star\sigma^k|/|\sigma^k|$, and the vorticity $\omega = \star_0^{-1} d_0^{T} \star_1 u$ is one pseudoscalar per vertex whose sign follows a chosen orientation of the plane.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D as ga
from numga.sparse import spdiag

from examples.fluids.dec_stokes.numerical import stokes_solution
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
Bivector = ga.gatype.bivector()                                              # Bivector+
Point = ga.gatype.antivector()                                               # Antivector-
Patch = ga.gatype.antibivector()                                             # Antibivector-


# --- math -----------------------------------------------------------------------------
class Operators:
    """The topological exterior derivatives and circumcentric Hodge maps of a triangulated surface."""

    def __init__(self, topology: Topology, vertices: Point):
        self.topology, self.vertices = topology, vertices
        V, E, F = topology.n_elements
        # Integrated fields carry one reading per simplex. These derivatives contain
        # only the incidence signs; their composition is zero before any geometry.
        self.d0 = topology.d[0] * Scalar                                     # (Scalar+ <- Scalar+)[E <- V]
        self.d1 = topology.d[1] * Scalar                                     # (Scalar+ <- Scalar+)[F <- E]
        self.d0_vorticity = topology.d[0] * Bivector                         # (Bivector+ <- Bivector+)[E <- V]
        self.edges = topology.d[0] * vertices                                # Point-[E]
        self.corners = topology.incidence[2, 0] * vertices                   # [3] Point-[F]
        sides = topology.boundary[2, 1] * self.edges                         # [3] Point-[F]
        self.face_patches = (sides[0] & sides[1]) / 2                        # Patch-[F]
        self.areas = self.face_patches.norm()                                # Scalar+[F]
        self.face_centres = self.corners.sum(axis=0) / 3                     # Point-[F]

        # On the paraboloid of half-squared distances, the plane through the three corners has slope equal to the circumcentre.
        gradients = sides / (2 * self.face_patches)                          # [3] Vector+[F]
        self.circumcentres = (gradients * (self.corners.norm_squared() / 2)).sum(axis=0).dual()  # Point-[F]

        ones = mv.scalar(np.ones((F, 1))).field()                            # Scalar+[F]
        midpoints = topology.incidence[1, 0].mean(axis=0) * vertices         # Point-[E]
        boundary = topology.d[1].adjugate() * ones                           # Scalar+[E]
        # Interior dual edges join adjacent circumcentres. At the boundary they
        # finish at the primal edge's midpoint, closing the truncated dual cells.
        # The dual edge crosses to the primal edge's right, matching positive flux.
        self.dual_edges = boundary * midpoints - topology.d[1].adjugate() * self.circumcentres  # Point-[E]

        # Oriented dual/primal edge ratios, and dual cell patches from the quarter-kites
        # incident to each vertex.
        self.edge_ratios = (self.dual_edges & self.edges) / self.edges.norm_squared()  # Patch-[E]
        self.dual_patches = topology.incidence[1, 0].adjugate().sum(axis=0) * ((self.dual_edges & self.edges) / 4)  # Patch-[V]
        self.dual_areas = self.dual_patches.norm()                           # Scalar+[V]

        # Each Hodge map scales by the dual cell's patch, the dual-to-primal edge ratio,
        # or the inverse face patch.
        self.star0 = spdiag(self.dual_patches * Scalar)                      # (Patch- <- Scalar+)[V <- V]
        self.star1 = spdiag(self.edge_ratios * Scalar)                       # (Patch- <- Scalar+)[E <- E]
        self.star2 = spdiag(self.face_patches.inverse() * Scalar)            # (Patch- <- Scalar+)[F <- F]
        self.inverse_star0 = Patch.dual_inverse() / self.dual_areas          # (Bivector+ <- Patch-)[V]

        # The adjugate sums dual-edge circulation around each dual cell.
        # Dividing by that dual cell's area gives the vertex vorticity bivector directly.
        self.curl = self.inverse_star0(self.d0_vorticity.adjugate()(self.star1))  # (Bivector+ <- Scalar+)[V <- E]


def stokes_flow(operators: Operators, viscosity: float, prescribed_flux: Scalar,
                fixed_edges: np.ndarray, boundary_circulation: Patch
                ) -> tuple[Scalar, Scalar, Bivector, Point]:
    """The steady flux, pressure, vorticity and face velocity under prescribed boundary fluxes and boundary circulation.

    Flux is positive through the right side of a directed edge. Pressure is determined
    up to a constant, fixed here by a zero area-weighted mean.
    """
    topology = operators.topology
    V, E, F = topology.n_elements
    # Vorticity differences across primal edges drive the paired edge response
    # through the adjugate of the dual-edge Hodge map.
    viscous = viscosity * operators.star1.adjugate()(operators.d0_vorticity(operators.curl))  # (Antiscalar- <- Scalar+)[E <- E]
    boundary_vorticity = operators.inverse_star0(boundary_circulation)       # Bivector+[V]
    boundary_resistance = viscosity * operators.star1.adjugate()(operators.d0_vorticity(boundary_vorticity))  # Antiscalar-[E]

    free_edges = np.setdiff1d(np.arange(E), fixed_edges)
    embedding = topology.reading(free_edges, 1).adjugate() * Scalar                    # (Scalar+ <- Scalar+)[E <- E_free]
    stiffness = embedding.adjugate()(viscous(embedding))                     # (Antiscalar- <- Scalar+)[E_free <- E_free]
    divergence = operators.d1(embedding)                                     # (Scalar+ <- Scalar+)[F <- E_free]
    force = -embedding.adjugate()(viscous(prescribed_flux) + boundary_resistance)  # Antiscalar-[E_free]
    balance = -operators.d1(prescribed_flux)                                 # Scalar+[F]
    free_flux, pressure = stokes_solution(stiffness, divergence, force, balance)  # Scalar+[E_free], Scalar+[F]
    flux = prescribed_flux + embedding(free_flux)                            # Scalar+[E]
    pressure = pressure - (pressure * operators.areas).sites.sum() / operators.areas.sites.sum()  # Scalar+[F]
    vorticity = operators.curl(flux) + boundary_vorticity                    # Bivector+[V]
    # The first moment of the outward flux gives the face's mean velocity.
    outward_flux = topology.boundary[2, 1] * flux                            # [3] Scalar+[F]
    velocity = -((operators.corners - operators.face_centres) * outward_flux).sum(axis=0) / (2 * operators.areas)  # Point-[F]
    return flux, pressure, vorticity, velocity
