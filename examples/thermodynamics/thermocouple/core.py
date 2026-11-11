"""A thermocouple: a wire of two metals joined end to end, in which heat at the junction gives a voltage between the
wire's ends, the Seebeck effect, and a current through the junction cools or warms it, the Peltier effect; on a mesh
of triangles, in planar spacetime joined with a direction for entropy, settling over time.

An electron holds a charge and an entropy, so a fall of the electrochemical potential pushes it by its charge and a
fall of temperature by its entropy, and wherever it drifts it takes both along. The electrons of the two metals hold
different entropies. With the junction warmer than the ends and no current passing, each metal builds up the voltage
that holds its electrons still against the fall of temperature, a different one in either metal, and the difference
stands between the ends. With a current passing, the electrons bring one entropy to the junction and take another
away, and the junction cools or warms by the difference, according to the current's sense. The lattice's vibrations
carry entropy and no charge, and conduct heat along the wire in either case.

The algebra is planar spacetime with a projective direction, `x+ y+ t- w0`, joined with one degenerate direction
for entropy, `s0`. A unit of charge is a direction along time, `-mv.t.dual()`, and a unit of entropy `mv.s.dual()`; a
carrier's parcel holds its share of both. The potentials are one vector, the electrochemical potential along t and
the temperature along s, and their drive is their fall, minus their gradient wedged with them, a bivector. A carrier's
velocity joined with its parcel is the flux it makes, and that map from velocities to fluxes is the advection,
`Spatial.regressive(parcel)`; its adjugate takes a drive to the force on the carrier, so that `drive &
advection(velocity) == force & velocity`. A mobility takes the force to a velocity, and the three together are the
response, the map from a drive to the flux it drives, `advection(mobility(advection.adjugate()))`, summed over the
kinds of carrier. The mobility is symmetric, so the response is its own adjugate, and `drive & response(drive)`, the
dissipation, is never negative.

Over the wire the potentials vary from place to place. The simplex gradient takes the potentials at the vertices to
their gradient wedged with them on each edge, and the wire's conductance, what each vertex's cell loses of charge and
entropy as the potentials rise at each vertex, is the response between the gradient and its adjugate,
`gradient.adjugate()(response(gradient))`. It is its own adjugate too, which ties the wire's two effects: a unit of
heat a second put in at the junction raises the voltage between the open ends by as much as a unit of current lowers
the junction's temperature, at every moment. A cell stores entropy as its temperature rises and no charge, so the
temperature takes time to spread and the electrochemical potential follows it at once. The model is the linear
response, the entropy the drives produce left out. Potentials are measured from their values in
equilibrium, lengths in the wire's length and charges in an electron's charge.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from math import factorial
from operator import and_

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, its degenerate w weighing a place, and the entropy a parcel holds, a direction of its own
# with no metric.
ga = Algebra("x+y+t-w0") * Algebra("s0")
# The plane of space, and what a parcel holds, charge along time and entropy.
space = Algebra("x+y+")
amounts = Algebra("t-s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# What a parcel holds, charge along time and entropy; and a direction through space, a carrier's velocity.
Parcel = Plane.derive.within(amounts).derive.dual                             # Antivector-
Spatial = Plane.derive.within(space).derive.dual                              # Antivector-
# A force on a carrier, which reads its velocity; the potentials, the electrochemical potential along t and the
# temperature along s; their drive; a flux; and the response, the map from a drive to the flux it drives.
Force = Plane.derive.within(space)                                            # Vector+
Potential = Plane.derive.within(amounts)                                      # Vector+
Drive = ga.gatype.bivector().derive.touching(space, amounts)                  # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
Response = ga.gatype((Flux, Drive))                                           # Antibivector- <- Bivector+
# The plane's two directions, and the origin, a place of weight one.
across = mv.x.dual()                                                          # [] Point-
up = mv.y.dual()                                                              # [] Point-
origin = mv.w.dual()                                                          # [] Point-
# A unit of charge, a direction along time, and a unit of entropy.
time = -mv.t.dual()                                                           # [] Parcel-
entropy = mv.s.dual()                                                         # [] Parcel-


# --- math -----------------------------------------------------------------------------
def simplex_gradients(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Each simplex's size `Scalar+[simplices]`, a triangle's area, and each corner's simplex gradient `[corners]
    Plane+[simplices]`, at these vertices, points of weight one: the slope across space of the function that is one at
    the corner and nothing on the facet across."""
    corners = topology.incidence[-1, 0] * vertices                            # [corners] Point-[simplices]
    # The facet across from each corner, joined with time and entropy, along which the potentials do not vary off the
    # wire, over that join closed by the corner.
    facing = reduce(and_, topology.off[-1, 0] * vertices, time & entropy)     # [corners] Antibivector-[simplices]
    gradients = facing / (facing & corners)                                   # [corners] Plane+[simplices]
    # A simplex's corners joined, along time and entropy: its size, times the factorial of its dimension.
    joined = reduce(and_, corners) & time & entropy                           # Scalar-[simplices]
    # A gradient's slope across space, its value at the origin, read along w, dropped.
    return joined.norm() / factorial(topology.n_dim), gradients.cast(Force)


def response(charges: np.ndarray, entropies: np.ndarray, mobilities: np.ndarray) -> Response:
    """A material's response `[...] Flux- <- Drive+`, the flux a drive drives: each kind of carrier, holding this
    charge and this entropy `[..., kinds]`, pushed by the drive, drifting at its mobility `[..., kinds]`, the velocity
    of a unit of it per unit force times its density, and taking its parcel along."""
    parcels = charges * time + entropies * entropy                           # [..., kinds] Parcel-
    # A carrier's velocity joined with its parcel is the flux it makes; the adjugate of that map takes a drive to the
    # force on the carrier.
    advection = Spatial.regressive(parcels)                                   # [..., kinds] Flux- <- Spatial-
    mobility = mv.scalar(mobilities[..., None]) * Force.dual()                # [..., kinds] Spatial- <- Force+
    return advection(mobility(advection.adjugate())).sum(axis=-1)


def conductance(topology: Topology, sizes: Scalar, gradients: Plane, responses: Response) -> SparseExtensor:
    """What each vertex's cell loses, charge and entropy, as the potentials rise at each vertex, `(Parcel- <-
    Potential+)[vertices <- vertices]`, each simplex responding by its own response `(Flux- <- Drive+)[simplices]`."""
    corners = topology.incidence[-1, 0]                                       # [corners] Scalar+[simplices <- vertices]
    # The potentials' gradient wedged with them on each simplex, from their values at its corners.
    gradient = (spdiag(gradients ^ Potential) * corners).sum(axis=0)          # (Drive+ <- Potential+)[simplices <- vertices]
    return gradient.adjugate()((sizes * responses)(gradient))


def storage(topology: Topology, sizes: Scalar, capacities: np.ndarray) -> SparseExtensor:
    """The entropy each vertex's cell takes up as its temperature rises, `(Parcel- <- Potential+)[vertices <-
    vertices]`: its share of the simplices around it, a part of each for each of its corners, times its capacity for
    each unit of size `[vertices]`; a cell takes up no charge."""
    n_corners = topology.n_dim + 1
    cells = (~topology.incidence[-1, 0] * (sizes / n_corners)).sum(axis=0)    # Scalar+[vertices]
    # A temperature read off the potentials, as that much entropy.
    warmed = entropy * (Potential & entropy)                                  # [] Parcel- <- Potential+
    return spdiag(cells * mv.scalar(capacities[:, None]).field() * warmed)


def settling(topology: Topology, losing: SparseExtensor, storing: SparseExtensor, held: np.ndarray, sources: Parcel,
             time_step: float, n_frames: int) -> Iterator[Potential]:
    """The wire's potentials `Potential+[vertices]` at each of `n_frames` moments, a time step apart, from
    equilibrium: each cell storing what is put into it `Parcel-[vertices]` a unit of time, less what it loses under
    this conductance `(Parcel- <- Potential+)[vertices <- vertices]`, by this storage `(Parcel- <- Potential+)[vertices
    <- vertices]`, the vertices `held` kept at their potentials in equilibrium. A cell stores no charge, so the
    electrochemical potential follows the temperature at once; each step is solved at its end."""
    free = topology.reading(np.setdiff1d(np.arange(topology.n_vertices), held), 0)   # Scalar+[free <- vertices]
    stepping = free * (losing + storing / time_step) * ~free                  # (Parcel- <- Potential+)[free <- free]
    # What is put in, and the potentials, in their full types: charge and entropy, and both potentials.
    nothing = 0 * ((mv.t + mv.s) & sources)                                   # Scalar+[vertices]
    sources = sources + nothing * (time + entropy)                            # Parcel-[vertices]
    potentials = nothing * (mv.t + mv.s)                                      # Potential+[vertices]
    for _ in range(n_frames):
        yield potentials
        potentials = ~free * stepping.solve(free * (sources + storing(potentials) / time_step))
