"""Salt fingers in a tank of warm salty water over cold fresh water, on a mesh of triangles, in planar spacetime joined
with three directions, for the volume, the entropy and the salt a parcel of water holds.

Warm water is light and salty water heavy, and with enough heat on top the tank is stably stratified, lighter water
over heavier. It still turns over, in narrow fingers: a salty parcel pushed down loses its warmth to its cooler
neighbours sideways far faster than it loses its salt, comes out heavier than they are and keeps sinking, and a fresh
parcel pushed up warms and keeps rising. The density ratio, the heat's share of the density difference over the salt's,
is above one where the tank is stable, and fingers grow as long as it stays below the ratio of the two diffusivities.

The algebra is planar spacetime, x+y+t-, joined with three degenerate directions, for volume, entropy and salt:
x+y+t-v0s0n0. A parcel is a point weighed by its volume, `mv.v.dual()`, offset along time by its mass, along the
entropy's direction by its entropy and along the salt's by its salt; a unit volume holds a little less mass the more
entropy it holds and a little more the more salt. Gravity's drive, minus the gradient of its potential wedged with
time, pulls on the mass alone, and the difference between parcels' masses is their buoyancy. The temperature's drive,
its gradient wedged with entropy, conducts entropy, and the salt's chemical potential's, its gradient wedged with salt,
conducts salt, each at its own rate: by the Gibbs-Duhem relation a parcel's volume already feels everything its
entropy, salt and mass would, so neither pushes anything. The flow carries whole parcels, and the entropy and the salt
it carries out of a cell are what the planes `mv.s` and `mv.n` read off that one flux. The water's flow is a stream
function on the vertices; its mass resists the flow's change and its viscosity the flow's turning, and the pressure,
which only keeps the flow from filling or emptying any cell, does no work on it and drops out.

Lengths are in units of the tank's depth, times in units of the time heat takes to conduct across it, and temperature
and salinity in units of their differences between the top and the bottom; the density is one, and the heat a unit of
entropy holds per volume is one degree.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, and the volume, the entropy and the salt a parcel holds, each a direction of its own with no metric.
spacetime = Algebra("x+y+t-")
ga = spacetime * Algebra("v0s0n0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
axis = mv.x.dual()                                                            # [] Point-
up = mv.y.dual()                                                              # [] Point-
time = -mv.t.dual()                                                           # [] Point-
# A unit of volume, the weight of a place, a unit of entropy and a unit of salt.
volume = mv.v.dual()                                                          # [] Point-
entropy = mv.s.dual()                                                         # [] Point-
salt = mv.n.dual()                                                            # [] Point-


# --- plumbing -------------------------------------------------------------------------
def tank(n_along: int, n_across: int, period: float) -> tuple[Topology, np.ndarray, np.ndarray]:
    """A tank of triangles of unit depth closed on itself along its length, `period` long: a chain up swept around a
    loop along. Its topology; its vertices' places, along and up `[vertices, 2]`; and how much further along each edge
    reaches than its ends' places, a period back where it crosses from the last column to the first, `[edges]`."""
    topology = Topology.chains([n_across]).product(Topology.loops([n_along]))
    places = np.stack(np.meshgrid(np.arange(n_along) / n_along * period, np.linspace(0, 1, n_across), indexing="ij"),
                      axis=-1)
    tails, heads = topology.elements[1].T
    steps = heads // n_across - tails // n_across
    return topology, places.reshape(-1, 2), period * np.where(np.abs(steps) > 1, -np.sign(steps), 0)


def placed(topology: Topology, places: np.ndarray, shifts: np.ndarray) -> tuple[Point, Point]:
    """A tank's vertices, places of unit volume `Point-[vertices]`, and its edges `Point-[edges]`."""
    vertices = (volume + axis * places[..., 0] + up * places[..., 1]).field()
    return vertices, topology.d[0] * vertices + (axis * shifts).field()


def assembled(topology: Topology, local: Scalar) -> SparseExtensor:
    """A form on each triangle's corners `[corners, corners] Scalar+[faces]` gathered onto the vertices it joins,
    `Scalar+[vertices <- vertices]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, edges: Point) -> tuple[Scalar, Point, Plane, Scalar]:
    """Each triangle's area `Scalar+[faces]`, the side facing each of its corners `[corners] Point-[faces]`, each
    corner's hat gradient `[corners] Plane+[faces]`, the plane through the facing side, along time, volume, entropy and
    salt, over twice the area, so that it reads one from the side to the corner, and each vertex's share of the area
    around it `Scalar+[vertices]`."""
    sides = topology.boundary[2, 1] * edges[..., None]                       # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time, at unit volume, entropy and salt.
    double_areas = ((((sides[..., 0] & sides[..., 1]) & time) & volume) & entropy) & salt   # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    gradients = (time & volume & entropy & salt & sides) / double_areas[..., None]   # [corners] Plane+[faces]
    return areas, sides, gradients, (~corners * (areas / 3)).sum(axis=-1)


def fingering(topology: Topology, areas: Scalar, sides: Point, gradients: Plane, cells: Scalar, free: np.ndarray,
              density: float, viscosity: float, expansion: float, contraction: np.ndarray, salt_diffusivity: float,
              gravity: float, temperature: Scalar, salinity: Scalar, dt: float, n_steps: int,
              stride: int) -> Iterator[tuple[Scalar, Scalar, Scalar]]:
    """The tank's stream function, temperature and salinity `[...] Scalar+[vertices]` every `stride` steps of `dt`,
    from rest and this temperature and salinity, for each contraction of the water by its salt `[...]`, the stream
    function nothing and the temperature and salinity held at the walls, free at the vertices `free`: each step
    gravity's drive does work on the flows the stream function can make, the water's mass and viscosity resist them,
    and the flow carries the entropy and the salt along while each spreads at its own rate."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    inside = topology.reading(free)                                           # Scalar+[free <- vertices]
    # Each corner's share summed back onto its vertex, and the free vertices' values put back among all of them.
    gathering = ~corners                                                      # [corners] Scalar+[vertices <- faces]
    embedding = ~inside                                                       # Scalar+[vertices <- free]
    # The stream function's kinetic form, the conduction of heat, and the viscous form, the squared vorticity.
    stiffness = assembled(topology, (gradients[..., :, None] | gradients[..., None, :]) * areas[..., None, None])
    viscous = viscosity * stiffness * spdiag(1 / cells) * stiffness           # Scalar+[vertices <- vertices]
    # Each step's implicit maps on the free vertices: the flow's inertia with its viscosity, the cells' heat with its
    # conduction, and their salt with its slower spreading.
    flowing = (inside * (density / dt * stiffness + viscous) * embedding) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    warming = (inside * (spdiag(cells / dt) + stiffness) * embedding) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    salting = (inside * (spdiag(cells / dt) + salt_diffusivity * stiffness)
               * embedding) * Scalar                                          # (Scalar+ <- Scalar+)[free <- free]
    # A stream function rising by one at a corner carries the water along the facing side, half the side's worth over
    # the triangle; gravity's drive pulls on the mass it carries.
    flows = sides / 2                                                         # [corners] Point-[faces]
    pull = -(gravity * mv.yt)                                                 # [] Drive+
    # A triangle's share of what its corners hold; the temperature and salinity the walls hold, and the heat and salt
    # they conduct into the free vertices.
    averaging = corners.mean(axis=0)                                          # Scalar+[faces <- vertices]
    held_temperature = temperature - embedding * (inside * temperature)       # [...] Scalar+[vertices]
    held_salinity = salinity - embedding * (inside * salinity)                # [...] Scalar+[vertices]
    wall_heat = inside * (stiffness * held_temperature)                       # [...] Scalar+[free]
    wall_salt = inside * (salt_diffusivity * (stiffness * held_salinity))     # [...] Scalar+[free]
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    for step in range(n_steps + 1):
        if step % stride == 0:
            yield stream, temperature, salinity
        # Each triangle's parcels per volume: unit volume, holding less mass the warmer and more the saltier they are,
        # and their entropy and salt.
        warmth = averaging * temperature                                      # [...] Scalar+[faces]
        saltiness = averaging * salinity                                      # [...] Scalar+[faces]
        parcels = (volume + (density - expansion * warmth + contraction * saltiness) * time + warmth * entropy
                   + saltiness * salt)                                        # [...] Point-[faces]
        load = (gathering * (pull & (flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        stream = embedding * flowing.solve(inside * (density / dt * (stiffness * stream) + load))
        # The flow carries the parcels along: what their flux carries out of each corner's cell leaves it, its entropy
        # and its salt alike.
        velocities = ((corners * stream[..., None]) * sides).sum(axis=-1) / (2 * areas)   # [...] Point-[faces]
        flux = velocities & parcels                                           # [...] Flux-[faces]
        carried = (gathering * (areas * (gradients ^ flux[..., None]))).sum(axis=-1)   # [...] Point-[vertices]
        temperature = held_temperature + embedding * warming.solve(
            inside * (cells / dt * temperature - (mv.s & carried)) - wall_heat)
        salinity = held_salinity + embedding * salting.solve(
            inside * (cells / dt * salinity - (mv.n & carried)) - wall_salt)
