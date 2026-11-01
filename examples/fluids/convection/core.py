"""Rayleigh-Bénard convection in a layer of liquid heated from below, on a mesh of triangles, in planar spacetime
joined with two directions, for the volume and the entropy a parcel of liquid holds.

A layer of liquid between a warm floor and a cool ceiling carries heat up by conduction while it is still. Warm liquid
is lighter, so the still layer is top-heavy; once buoyancy beats viscosity and conduction, which the Rayleigh number
measures, the layer turns over in rolls that carry the heat up themselves.

The algebra is planar spacetime, x+y+t-, joined with two degenerate directions, one for volume and one for entropy:
x+y+t-v0s0. A parcel is a point weighed by its volume, `mv.v.dual()`, offset along time by its mass and along the
entropy's direction by its entropy; a unit volume holds a little less mass the more entropy it holds. The mechanical
drive is minus the gradient of the potentials wedged with what they act on, the pressure with volume and gravity's
potential with time, and its commutator with a parcel is the force on it: the pressure's push on its volume and
gravity's pull on its mass, whose difference over a warm parcel is its buoyancy. The temperature's drive, its gradient
wedged with entropy, conducts entropy instead: by the Gibbs-Duhem relation a parcel's volume already feels everything
its entropy and mass would, so the temperature pushes nothing. The liquid's flow is a stream function on the vertices;
its mass resists the flow's change and its viscosity the flow's turning, and the pressure, which only keeps the flow
from filling or emptying any cell, does no work on it and drops out.

Lengths are in units of the layer's depth, times in units of the time heat takes to conduct across it, and
temperatures in units of the difference between floor and ceiling; the density is one, and the heat a unit of entropy
holds per volume is one degree.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, and the volume and the entropy a parcel holds, each a direction of its own with no metric.
spacetime = Algebra("x+y+t-")
ga = spacetime * Algebra("v0s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Drive = ga.gatype.from_blades("xt yt xv yv xs ys")                            # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
axis = mv.x.dual()                                                            # [] Point-
up = mv.y.dual()                                                              # [] Point-
time = -mv.t.dual()                                                           # [] Point-
# A unit of volume, the weight of a place, and a unit of entropy.
volume = mv.v.dual()                                                          # [] Point-
entropy = mv.s.dual()                                                         # [] Point-


# --- plumbing -------------------------------------------------------------------------
def layer(n_along: int, n_across: int, period: float) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A layer of triangles of unit depth closed on itself along its length, `period` long: a chain up swept around a
    loop along. Its topology; its vertices' places, along and up `[vertices, 2]`; how much further along each edge
    reaches than its ends' places, a period back where it crosses from the last column to the first, `[edges]`; and the
    vertices of the floor and of the ceiling `[along]`."""
    topology = Topology.chains([n_across]).product(Topology.loops([n_along]))
    places = np.stack(np.meshgrid(np.arange(n_along) / n_along * period, np.linspace(0, 1, n_across), indexing="ij"),
                      axis=-1)
    tails, heads = topology.elements[1].T
    steps = heads // n_across - tails // n_across
    floor = np.arange(n_along) * n_across
    return (topology, places.reshape(-1, 2), period * np.where(np.abs(steps) > 1, -np.sign(steps), 0), floor,
            floor + n_across - 1)


def placed(topology: Topology, places: np.ndarray, shifts: np.ndarray) -> tuple[Point, Point]:
    """A layer's vertices, places of unit volume `Point-[vertices]`, and its edges `Point-[edges]`."""
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
    corner's hat gradient `[corners] Plane+[faces]`, the plane through the facing side, along time, volume and entropy,
    over twice the area, so that it reads one from the side to the corner, and each vertex's share of the area around it
    `Scalar+[vertices]`."""
    sides = topology.boundary[2, 1] * edges[..., None]                       # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time, at unit volume and entropy.
    double_areas = (((sides[..., 0] & sides[..., 1]) & time) & volume) & entropy   # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    gradients = (time & volume & entropy & sides) / double_areas[..., None]   # [corners] Plane+[faces]
    return areas, sides, gradients, (~corners * (areas / 3)).sum(axis=-1)


def convecting(topology: Topology, areas: Scalar, sides: Point, gradients: Plane, cells: Scalar, free: np.ndarray,
               density: float, viscosity: float, expansion: float, gravity: np.ndarray, temperature: Scalar, dt: float,
               n_steps: int, stride: int) -> Iterator[tuple[Scalar, Scalar]]:
    """The layer's stream function and temperature `[...] Scalar+[vertices]` every `stride` steps of `dt`, from rest
    and this temperature, under each gravity `[...]`, the stream function nothing and the temperature held at the walls,
    free at the vertices `free`: each step gravity's drive does work on the flows the stream function can make, the
    liquid's mass and viscosity resist them, and the flow carries the entropy along while conduction spreads it."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    inside = topology.reading(free)                                           # Scalar+[free <- vertices]
    # The stream function's kinetic form, the conduction of heat, and the viscous form, the squared vorticity.
    stiffness = assembled(topology, (gradients[..., :, None] | gradients[..., None, :]) * areas[..., None, None])
    viscous = viscosity * stiffness * spdiag(1 / cells) * stiffness           # Scalar+[vertices <- vertices]
    # Each step's implicit maps on the free vertices: the flow's inertia with its viscosity, and the cells' heat with
    # its conduction.
    flowing = (inside * (density / dt * stiffness + viscous) * ~inside) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    warming = (inside * (spdiag(cells / dt) + stiffness) * ~inside) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    # A stream function rising by one at a corner carries the liquid along the facing side, half the side's worth over
    # the triangle; gravity's drive pulls on the mass it carries.
    flows = sides / 2                                                         # [corners] Point-[faces]
    pull = -(gravity * mv.yt)                                                 # [...] Drive+
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    for step in range(n_steps + 1):
        if step % stride == 0:
            yield stream, temperature
        # Each triangle's parcels per volume: unit volume, holding less mass the warmer they are, and their entropy.
        warmth = corners.mean(axis=0) * temperature                           # [...] Scalar+[faces]
        parcels = volume + (density - expansion * warmth) * time + warmth * entropy   # [...] Point-[faces]
        load = (~corners * (pull[..., None] & (flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        stream = ~inside * flowing.solve(inside * (density / dt * (stiffness * stream) + load))
        # The flow carries the entropy along; what the flux carries out of each corner's cell leaves it.
        velocities = ((corners * stream[..., None]) * sides).sum(axis=-1) / (2 * areas)   # [...] Point-[faces]
        flux = velocities.regressive(warmth * entropy)                        # [...] Flux-[faces]
        carried = mv.s & (~corners * (areas * (gradients ^ flux[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        held = temperature - ~inside * (inside * temperature)                 # [...] Scalar+[vertices]
        temperature = held + ~inside * warming.solve(inside * (cells / dt * temperature - carried - stiffness * held))
