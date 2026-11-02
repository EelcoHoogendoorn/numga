"""Salt fingers in a tank of warm salty water over cold fresh water, on a mesh of triangles, in the plane joined with
four directions, for the volume, the mass, the entropy and the salt a parcel of water holds.

Warm water is light and salty water heavy, and with enough heat on top the tank is stably stratified, lighter water
over heavier. It still turns over, in narrow fingers: a salty parcel pushed down loses its warmth to its cooler
neighbours sideways far faster than it loses its salt, comes out heavier than they are and keeps sinking, and a fresh
parcel pushed up warms and keeps rising. The density ratio, the heat's share of the density difference over the salt's,
is above one where the tank is stable, and fingers grow as long as it stays below the ratio of the two diffusivities.

The algebra is the plane, x+y+, joined with four degenerate directions, for volume, mass, entropy and salt:
x+y+v0m0s0n0. A parcel is a point weighed by its volume, `mv.v.dual()`, offset along the mass's direction by its mass,
along the entropy's by its entropy and along the salt's by its salt; a unit volume holds a little less mass the more
entropy it holds and a little more the more salt. A drive is minus the gradient of a potential wedged with what it acts
on, and pairs with the flux of that quantity: gravity's drive, the geopotential's gradient wedged with mass, pulls on
the mass alone, and the difference between parcels' masses is their buoyancy; the temperature's drive, wedged with
entropy, and the salt's, wedged with salt, are each mapped to their amount's flux, at its own rate. The flow carries
whole parcels, and the entropy and the salt it carries out of a cell are what the planes `mv.s` and `mv.n` read off
that one flux. The water's flow is a stream function on the vertices; its mass resists the flow's change and its
viscosity the flow's turning, and the pressure, which only keeps the flow from filling or emptying any cell, does no
work on it and drops out.

Lengths are in units of the tank's depth, times in units of the time heat takes to conduct across it, and temperature
and salinity in units of their differences between the top and the bottom; the density is one, and the heat a unit of
entropy holds per volume is one degree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# The plane, and the volume, the mass, the entropy and the salt a parcel holds, each a direction of its own with no
# metric.
ga = Algebra("x+y+") * Algebra("v0m0s0n0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# The planes reading a place's run along the tank and up it.
Upright = ga.gatype.from_blades("x y")                                        # Vector+
Drive = ga.gatype.from_blades("xv yv xm ym xs ys xn yn")                      # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# The tank's two directions, along its length and up its depth, and the planes reading a direction's run along each.
length = mv.x.dual()                                                          # [] Point-
height = mv.y.dual()                                                          # [] Point-
directions = stack((length, height))                                          # [2] Point-
readings = stack((mv.x, mv.y))                                                # [2] Plane+
# A unit of volume, the weight of a place, a unit of mass, a unit of entropy and a unit of salt.
volume = mv.v.dual()                                                          # [] Point-
mass = mv.m.dual()                                                            # [] Point-
entropy = mv.s.dual()                                                         # [] Point-
salt = mv.n.dual()                                                            # [] Point-


# --- plumbing -------------------------------------------------------------------------
def tank(n_along: int, n_across: int, period: float) -> tuple[Topology, Topology, SparseExtensor, SparseExtensor,
                                                               np.ndarray]:
    """A tank of triangles of unit depth closed on itself along its length, a chain up swept around a loop along, and
    the same tank opened at its seam, swept along a chain a column longer whose last column is the first again, a
    period further along: their topologies, whose triangles come in the same order; the maps reading a field on the
    open tank at the closed one's vertices and on the closed tank at the open one's; and the open tank's places along
    and up `[vertices, 2]`."""
    closed = Topology.chains([n_across]).product(Topology.loops([n_along]))
    opened = Topology.chains([n_across]).product(Topology.chains([n_along + 1]))
    places = np.stack(np.meshgrid(np.arange(n_along + 1) / n_along * period, np.linspace(0, 1, n_across),
                                  indexing="ij"), axis=-1)
    closing = opened.reading(np.arange(closed.n_vertices))                   # numbers[closed <- opened]
    opening = closed.reading(np.arange(opened.n_vertices) % closed.n_vertices)   # numbers[opened <- closed]
    return closed, opened, closing, opening, places.reshape(-1, 2)


def holding(form: SparseExtensor, interior: SparseExtensor) -> Callable[[Scalar, Scalar], Scalar]:
    """The step of a map on a mesh's vertices solved with its walls held: given the right-hand side and the walls'
    values `[...] Scalar+[vertices]`, the values the map takes to it at the free vertices, read by `interior`, and the
    walls' values elsewhere. The map on the free vertices is built once, so that every step only substitutes."""
    free = (interior * form * ~interior) * Scalar                             # (Scalar+ <- Scalar+)[free <- free]
    return lambda rhs, walls: walls + ~interior * free.solve(interior * (rhs - form * walls))


# --- math -----------------------------------------------------------------------------
class Forms(NamedTuple):
    """The water's maps on a mesh's triangles: each corner's flow, the velocity of a stream function and the outflow of
    a flux, each vertex's share of the area, and the forms of a step."""
    corners: SparseExtensor                                                   # [corners] Scalar+[faces <- vertices]
    corner_mean: SparseExtensor                                               # Scalar+[faces <- vertices]
    flows: Point                                                              # [corners] Point-[faces]
    stream_velocity: SparseExtensor                                           # (Point- <- Scalar+)[faces <- vertices]
    cell_outflow: SparseExtensor                                              # (Point- <- Flux-)[vertices <- faces]
    cells: Scalar                                                             # Scalar+[vertices]
    kinetic: SparseExtensor                                                   # Scalar+[vertices <- vertices]
    viscous: SparseExtensor                                                   # Scalar+[vertices <- vertices]
    conduction: SparseExtensor                                                # Scalar+[vertices <- vertices]
    diffusion: SparseExtensor                                                 # Scalar+[vertices <- vertices]


def forms(closed: Topology, opened: Topology, vertices: Point, density: float, viscosity: float,
          salt_diffusivity: float) -> Forms:
    """The water's maps, built once from a tank's triangles, their geometry read on the open tank and gathered onto the
    closed one's vertices: each corner's flow, a stream function's velocity, the outflow of a flux, the stream
    function's kinetic and viscous forms, and the conduction of heat and the spreading of salt, each the outflow of the
    flux its drive drives."""
    sides = opened.boundary[2, 1] * (opened.d[0] * vertices)                  # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, at unit volume, mass, entropy and salt.
    double_areas = ((((sides[0] & sides[1]) & volume) & mass) & entropy) & salt   # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    # A corner's hat gradient: the plane through the facing side, along volume, mass, entropy and salt, over twice
    # the area, so that it reads one from the side to the corner.
    gradients = (volume & mass & entropy & salt & sides) / double_areas       # [corners] Plane+[faces]
    corners = closed.incidence[2, 0]                                          # [corners] Scalar+[faces <- vertices]
    # Each vertex's share of the area around it, and each triangle's mean over its corners.
    cells = (~corners * (areas / 3)).sum(axis=0)                              # Scalar+[vertices]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[faces <- vertices]
    # A stream function rising by one at a corner carries the water along the facing side, half the side's worth
    # over the triangle: each corner's flow, and its velocity.
    flows = sides / 2                                                         # [corners] Point-[faces]
    velocities = flows / areas                                                # [corners] Point-[faces]
    # A stream function's velocity on each triangle, its corners' velocities weighed by its values there, and what
    # a flux carries out of each corner's cell: its hat gradient wedged with the flux, gathered onto the vertex.
    stream_velocity = (spdiag(velocities * Scalar) * corners).sum(axis=0)     # (Point- <- Scalar+)[faces <- vertices]
    cell_outflow = (~corners * spdiag(areas * (gradients ^ Flux))).sum(axis=0)   # (Point- <- Flux-)[vertices <- faces]
    # The kinetic form: the water's inertia reads a velocity's run along and up into its momentum, paired with the
    # velocity of each other corner over the triangle, and gathered onto the vertices.
    inertia = density * (readings * (readings & Point)).sum(axis=0)           # [] Plane+ <- Point-
    local = areas * (inertia(velocities[:, None]) & velocities[None, :])      # [corners, corners] Scalar+[faces]
    kinetic = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    # The vorticity at each vertex, the kinetic form's amount over its share of the area and the density, and the
    # viscous form, its square over the tank.
    vorticity = spdiag(1 / (density * cells)) * kinetic                       # Scalar+[vertices <- vertices]
    viscous = viscosity * ~vorticity * spdiag(cells) * vorticity              # Scalar+[vertices <- vertices]
    # The temperature's drive at each corner's hat, minus its gradient wedged with the entropy it pushes, and the
    # salt's, wedged with salt; the conductivity and the salt's diffusivity, each drive read across the tank's two
    # directions driving its amount along them.
    heat_drives, salt_drives = -(gradients ^ mv.s), -(gradients ^ mv.n)       # [corners] Drive+[faces], [corners] Drive+[faces]
    conductivity = ((directions & entropy) * ((directions & entropy) & Drive)).sum(axis=0)   # [] Flux- <- Drive+
    diffusivity = salt_diffusivity * ((directions & salt) * ((directions & salt) & Drive)).sum(axis=0)   # [] Flux- <- Drive+
    # What a flux carries out of each corner's cell: its hat gradient wedged with the flux, read for its entropy or
    # its salt, gathered onto the vertices.
    local = areas * (mv.s & (gradients[:, None] ^ conductivity(heat_drives[None, :])))   # [corners, corners] Scalar+[faces]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    local = areas * (mv.n & (gradients[:, None] ^ diffusivity(salt_drives[None, :])))   # [corners, corners] Scalar+[faces]
    diffusion = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    return Forms(corners, corner_mean, flows, stream_velocity, cell_outflow, cells, kinetic, viscous, conduction, diffusion)


def fingering(water: Forms, interior: SparseExtensor, density: float, expansion: float, contraction: np.ndarray,
              gravity: float, temperature: Scalar, salinity: Scalar, dt: float, n_steps: int,
              n_stride: int) -> Iterator[tuple[Scalar, Scalar, Scalar]]:
    """The tank's stream function, temperature and salinity `[...] Scalar+[vertices]` every `n_stride` steps of `dt`,
    from rest and this temperature and salinity, for each contraction of the water by its salt `[...]`, the stream
    function nothing and the temperature and salinity held at the walls, all but the vertices `interior` reads: each
    step gravity's drive does work on the flows the stream function can make, the water's mass and viscosity resist
    them, and the flow carries the entropy and the salt along while each one's drive spreads it at its own rate."""
    # Each step's implicit maps, solved with the walls held: the flow's inertia with its viscosity, the cells' heat
    # with its conduction, and their salt with its slower spreading.
    flow_step = holding(water.kinetic / dt + water.viscous, interior)
    heat_step = holding(spdiag(water.cells / dt) + water.conduction, interior)
    salt_step = holding(spdiag(water.cells / dt) + water.diffusion, interior)
    # The temperature and the salinity the walls hold.
    wall_temperature = temperature - ~interior * (interior * temperature)     # [...] Scalar+[vertices]
    wall_salinity = salinity - ~interior * (interior * salinity)              # [...] Scalar+[vertices]
    pull = -(gravity * mv.ym)                                                 # [] Drive+
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            yield stream, temperature, salinity
        # Each triangle's parcels per volume: unit volume, holding less mass the warmer and more the saltier they are,
        # and their entropy and salt.
        warmth = water.corner_mean * temperature                              # [...] Scalar+[faces]
        saltiness = water.corner_mean * salinity                              # [...] Scalar+[faces]
        parcels = (volume + (density - expansion * warmth + contraction * saltiness) * mass + warmth * entropy
                   + saltiness * salt)                                        # [...] Point-[faces]
        # Gravity's drive does work on each corner's flow by pulling on the mass it carries.
        load = (~water.corners * (pull & (water.flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        stream = flow_step(water.kinetic / dt * stream + load, 0 * stream)
        # The flow carries the parcels along: what their flux carries out of each corner's cell leaves it, its entropy
        # and its salt alike, and each one's drive spreads the rest, implicitly.
        velocity = water.stream_velocity(stream)                              # [...] Point-[faces]
        flux = velocity & parcels                                             # [...] Flux-[faces]
        outflow = water.cell_outflow(flux)                                    # [...] Point-[vertices]
        temperature = heat_step(water.cells / dt * temperature - (mv.s & outflow), wall_temperature)
        salinity = salt_step(water.cells / dt * salinity - (mv.n & outflow), wall_salinity)
