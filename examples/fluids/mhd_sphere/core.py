"""Magnetohydrodynamics on a sphere: a thin conducting fluid whose flow winds up the magnetic field it carries.

Spacetime, `x+y+z+t-`, holds the sphere in one instant: its points are directions of space, and the fluid's
worldlines are time joined with its velocity. On a surface, an incompressible flow and a field without sources are each
one function on the vertices: the stream function, whose contours are streamlines, and the flux function, whose
contours are field lines. The flux function's spacetime gradient is the field: paired with an edge, `gradient(flux) &
edge`, it is the magnetic flux through the wall standing on that edge; paired with time, the electric field along the
surface's normal. A perfect conductor freezes the field into the fluid, the gradient pairing to nothing with every
worldline, and resistivity is what lets that pairing slip. The flow carries its vorticity and the field pulls back on
it through one bracket, the Jacobian of two functions on each triangle, with flow and field swapping roles.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# The planes of space, which read a point's coordinates; a function's gradient on a triangle is one.
Plane = ga.gatype.from_blades("x y z")              # Vector+
# The points of space, each the direction from the origin to it; the sphere's points, its edges and the fluid's
# velocities are points too.
Point = Plane.dual().output_gatype                  # Antivector-
# A triangle's patch, the join of two of its edges.
Patch = ga.gatype.antibivector()                    # Antibivector-
# A function integrated over each vertex's share of the surface is an amount.
Amount = ga.gatype.antiscalar()                     # Antiscalar-
GradientMap = ga.gatype((Plane, Scalar))
VelocityMap = ga.gatype((Point, Scalar))


class Surface(NamedTuple):
    """A triangulated surface in spacetime: its patches, the gradient and velocity maps from functions on its
    vertices to its faces, the share of each face its vertices get, its stiffness and mass, and the stiffness with its
    free constant pinned, which finds a stream function from its vorticity."""
    topology: Topology
    patches: Patch                                  # Patch-[faces]
    gradient: GradientMap                           # (Plane+ <- Scalar+)[faces <- vertices]
    velocity: VelocityMap                           # (Point- <- Scalar+)[faces <- vertices]
    lumped: SparseExtensor                          # Scalar+[vertices <- faces]
    stiffness: SparseExtensor                       # (Amount- <- Scalar+)[vertices <- vertices]
    mass: SparseExtensor                            # (Amount- <- Scalar+)[vertices <- vertices]
    pinned: SparseExtensor                          # (Amount- <- Scalar+)[vertices <- vertices]


# --- math -----------------------------------------------------------------------------
def surface(topology: Topology, vertices: Point) -> Surface:
    """The surface's operators. A corner's hat gradient is the edge facing it over the triangle's patch; a stream
    function's velocity is its gradient met with the triangle's unit patch, and the stiffness is the kinetic energy of
    that velocity, momentum paired with velocity, pulled back to the vertices."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)                 # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                          # Patch-[faces]
    areas = patches.squared().square_root()                                      # Scalar+[faces]
    corner_gradients = (sides | patches.inverse()) / 2                           # [corners] Plane+[faces]
    gradient = (spdiag(corner_gradients * Scalar) * topology.incidence[2, 0]).sum(axis=0)   # (Plane+ <- Scalar+)[faces <- vertices]
    velocity = spdiag(Plane ^ (patches / areas))(gradient)                     # (Point- <- Scalar+)[faces <- vertices]
    # Each corner's share of the faces around it, and the map handing each vertex its share of a face integral.
    corner_mean = topology.incidence[2, 0].mean(axis=0)                          # Scalar+[faces <- vertices]
    dual_areas = ~corner_mean * areas                                            # Scalar+[vertices]
    lumped = spdiag(1 / dual_areas) * ~corner_mean                               # Scalar+[vertices <- faces]
    momentum = spdiag(areas * Point.dual())                                  # (Plane+ <- Point-)[faces <- faces]
    stiffness = velocity.adjugate()(momentum(velocity))                          # (Amount- <- Scalar+)[vertices <- vertices]
    mass = spdiag(dual_areas * Scalar.dual())                                    # (Amount- <- Scalar+)[vertices <- vertices]
    # A stream function's constant is free; pinning the first vertex to zero fixes it and leaves every velocity as it is.
    reference = mv.scalar((np.arange(topology.n_vertices) == 0)[:, None]).field()   # Scalar+[vertices]
    pinned = stiffness + spdiag(stiffness.diagonal() * reference)                # (Amount- <- Scalar+)[vertices <- vertices]
    return Surface(topology, patches, gradient, velocity, lumped, stiffness, mass, pinned)


def bracket(surface: Surface, stream: Scalar, carried: Scalar) -> Scalar:
    """The rate at which the flow of a stream function carries a function, at the vertices: on each triangle the
    carried gradient paired with the stream function's area-weighted velocity, the Jacobian of the two, taken away as
    the flow brings in what lies upstream, and shared out to the corners."""
    flows = surface.gradient(stream) ^ surface.patches                          # Point-[faces]
    return -(surface.lumped * (surface.gradient(carried) & flows))               # Scalar+[vertices]


def laplacian(surface: Surface, function: Scalar) -> Scalar:
    """The Laplacian at the vertices, negated: the stiffness's amount over each vertex's share of the surface."""
    return surface.mass.solve(surface.stiffness(function))                      # Scalar+[vertices]


def stream_function(surface: Surface, vorticity: Scalar) -> Scalar:
    """The stream function whose negated Laplacian is the vorticity."""
    return surface.pinned.solve(surface.mass(vorticity))                        # Scalar+[vertices]


def rate(surface: Surface, state: Scalar, viscosity: float, resistivity: float) -> Scalar:
    """The rate of change of the vorticity and flux functions `[2] Scalar+[vertices]`."""
    vorticity, flux = state                                                      # Scalar+[vertices], Scalar+[vertices]
    stream = stream_function(surface, vorticity)                                 # Scalar+[vertices]
    current = laplacian(surface, flux)                                           # Scalar+[vertices]
    # The flow carries its vorticity, and the field's tension pulls on the flow as the field's own flow would carry
    # the current, the other way round.
    vorticity_rate = (bracket(surface, stream, vorticity) - bracket(surface, flux, current)
                      - viscosity * laplacian(surface, vorticity))              # Scalar+[vertices]
    # Frozen in, the flux function's spacetime gradient pairs to nothing with each worldline, time joined with the
    # velocity: its rate along time is minus its change along the velocity, here over each triangle's area.
    # Resistivity lets it slip by the current.
    flows = surface.gradient(stream) ^ surface.patches                          # Point-[faces]
    frozen = -(surface.gradient(flux) & flows)                                   # Scalar+[faces]
    flux_rate = surface.lumped * frozen - resistivity * current                  # Scalar+[vertices]
    return stack((vorticity_rate, flux_rate))                                    # [2] Scalar+[vertices]


def evolve(surface: Surface, initial: Scalar, viscosity: float, resistivity: float, dt: float,
           n_steps: int) -> Iterator[Scalar]:
    """Successive vorticity and flux functions `[2] Scalar+[vertices]` follow classical fourth-order Runge–Kutta
    steps."""
    state = initial                                                              # [2] Scalar+[vertices]
    yield state
    for _ in range(n_steps):
        first = rate(surface, state, viscosity, resistivity)                     # [2] Scalar+[vertices]
        second = rate(surface, state + (dt / 2) * first, viscosity, resistivity)  # [2] Scalar+[vertices]
        third = rate(surface, state + (dt / 2) * second, viscosity, resistivity)  # [2] Scalar+[vertices]
        fourth = rate(surface, state + dt * third, viscosity, resistivity)       # [2] Scalar+[vertices]
        state = state + (dt / 6) * (first + 2 * second + 2 * third + fourth)     # [2] Scalar+[vertices]
        yield state
