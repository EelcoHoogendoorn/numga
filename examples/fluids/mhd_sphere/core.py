"""Incompressible magnetohydrodynamics on a sphere couples a scalar stream function and a scalar magnetic flux function in spacetime.

At a single instant, points on the unit sphere are spatial directions and fluid worldlines join time with tangent velocity. An incompressible surface flow and a source-free tangential magnetic field are represented by vertex stream and flux functions whose contours form streamlines and magnetic field lines. Meeting a scalar gradient with each triangle's area patch turns slope planes into tangent area-weighted flows, and pairing another gradient with that flow integrates their surface Jacobian over each triangle. In a perfect conductor, magnetic flux advects along fluid worldlines while resistivity diffuses flux by the electric current density, and an antisymmetric bracket couples vorticity advection with Lorentz tension.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                   # Scalar+
Antiscalar = ga.gatype.antiscalar()                           # Antiscalar-
# Spatial planes read coordinates on the sphere and represent triangle gradients.
Plane = ga.gatype.vector().derive.within(Algebra("x+y+z+"))   # Vector+
# Spatial directions represent sphere vertices, triangle edges, and tangent velocities.
Point = Plane.derive.dual                                     # Antivector-
# Joining two triangle edges forms an oriented surface patch.
Patch = ga.gatype.antibivector()                              # Antibivector-


class Surface(NamedTuple):
    """Precomputed geometric and elliptic operators (`Patch-[F]` and sparse maps between `V` and `F`) act on vertex functions over a triangulated surface."""
    topology: Topology
    patches: Patch                                            # Patch-[F]
    gradient: SparseExtensor                                  # (Plane+ <- Scalar+)[F <- V]
    velocity: SparseExtensor                                  # (Point- <- Scalar+)[F <- V]
    vertex_projection: SparseExtensor                         # Scalar+[V <- F]
    stiffness: SparseExtensor                                 # (Antiscalar- <- Scalar+)[V <- V]
    mass: SparseExtensor                                      # (Antiscalar- <- Scalar+)[V <- V]
    pinned_stiffness: SparseExtensor                          # (Antiscalar- <- Scalar+)[V <- V]


# --- math -----------------------------------------------------------------------------
def surface(topology: Topology, vertices: Point) -> Surface:
    """Triangle patches, simplex gradients, tangent velocity maps, dual-cell projections, and Dirichlet stiffness operators assemble a `Surface` from `vertices` (`Point-[V]`)."""
    V, E, F = topology.n_elements
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                     # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                              # Patch-[F]
    areas = patches.antinorm()                                                       # Scalar+[F]
    # Each corner's simplex gradient divides the opposite edge by twice the oriented triangle patch.
    gradients = (sides | patches.inverse()) / 2                                      # [3] Plane+[F]
    gradient = (spdiag(gradients * Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[F <- V]
    velocity = (Plane ^ (patches / areas))(gradient)                                 # (Point- <- Scalar+)[F <- V]
    # Each vertex dual cell receives one third of each incident triangle's area and integral.
    corner_mean = topology.incidence[2, 0].mean(axis=0)                              # Scalar+[F <- V]
    dual_areas = corner_mean.adjugate() * areas                                      # Scalar+[V]
    vertex_projection = spdiag(1 / dual_areas) * corner_mean.adjugate()              # Scalar+[V <- F]
    momentum = areas * Point.dual()                                                  # (Plane+ <- Point-)[F]
    stiffness = velocity.adjugate()(momentum(velocity))                              # (Antiscalar- <- Scalar+)[V <- V]
    mass = spdiag(dual_areas * Scalar.dual())                                        # (Antiscalar- <- Scalar+)[V <- V]
    # Pinning vertex zero removes the stream function's free additive constant without altering velocities.
    reference = mv.scalar((np.arange(V) == 0)[:, None]).field()                      # Scalar+[V]
    pinned_stiffness = stiffness + stiffness.diagonal() * reference                  # (Antiscalar- <- Scalar+)[V <- V]
    return Surface(topology, patches, gradient, velocity, vertex_projection, stiffness, mass, pinned_stiffness)


def bracket(surface: Surface, stream: Scalar, function: Scalar) -> Scalar:
    """Triangle-integrated pairing of `function`'s gradient with `stream`'s tangent flow projects onto dual cells as the vertex advection rate (`Scalar+[V]`)."""
    flows = surface.gradient(stream) ^ surface.patches                               # Point-[F]
    return -(surface.vertex_projection * (surface.gradient(function) & flows))       # Scalar+[V]


def laplacian(surface: Surface, function: Scalar) -> Scalar:
    """Solving the lumped dual-cell mass against Dirichlet stiffness evaluates the positive-definite negated surface Laplacian (`Scalar+[V]`)."""
    return surface.mass.solve(surface.stiffness(function))                           # Scalar+[V]


def stream_function(surface: Surface, vorticity: Scalar) -> Scalar:
    """Solving the pinned Dirichlet system recovers the zero-pinned vertex stream function (`Scalar+[V]`) from `vorticity` (`Scalar+[V]`)."""
    return surface.pinned_stiffness.solve(surface.mass(vorticity))                   # Scalar+[V]


def rate(surface: Surface, state: Scalar, viscosity: float, resistivity: float) -> Scalar:
    """Vorticity advection, Lorentz tension, frozen-in flux advection, and diffusive damping determine the time derivative (`[2] Scalar+[V]`) of `state` (`[2] Scalar+[V]`)."""
    vorticity, flux = state                                                          # Scalar+[V], Scalar+[V]
    stream = stream_function(surface, vorticity)                                     # Scalar+[V]
    current = laplacian(surface, flux)                                               # Scalar+[V]
    # Fluid shear advects vorticity while bent magnetic field lines pull back through the flux-current bracket.
    vorticity_rate = (bracket(surface, stream, vorticity) - bracket(surface, flux, current)
                      - viscosity * laplacian(surface, vorticity))                   # Scalar+[V]
    # Vanishing pairing of the spacetime flux gradient along fluid worldlines sets ideal advection, slipped by current.
    flows = surface.gradient(stream) ^ surface.patches                               # Point-[F]
    advection = -(surface.gradient(flux) & flows)                                    # Scalar+[F]
    flux_rate = surface.vertex_projection * advection - resistivity * current        # Scalar+[V]
    return stack((vorticity_rate, flux_rate))                                        # [2] Scalar+[V]


def evolution(surface: Surface, initial: Scalar, viscosity: float, resistivity: float, dt: float,
              n_steps: int) -> Iterator[Scalar]:
    """Classical fourth-order Runge–Kutta integration yields `n_steps + 1` successive vorticity-flux states (`[2] Scalar+[V]`) starting from `initial`."""
    state = initial                                                                  # [2] Scalar+[V]
    yield state
    for _ in range(n_steps):
        first_rate = rate(surface, state, viscosity, resistivity)                    # [2] Scalar+[V]
        second_rate = rate(surface, state + (dt / 2) * first_rate, viscosity, resistivity)   # [2] Scalar+[V]
        third_rate = rate(surface, state + (dt / 2) * second_rate, viscosity, resistivity)   # [2] Scalar+[V]
        fourth_rate = rate(surface, state + dt * third_rate, viscosity, resistivity)         # [2] Scalar+[V]
        state = state + (dt / 6) * (first_rate + 2 * second_rate + 2 * third_rate + fourth_rate)  # [2] Scalar+[V]
        yield state
