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
entropy it holds and a little more the more salt. A drive is minus the gradient of a potential wedged with what it acts
on, and pairs with the flux of that quantity: gravity's drive, wedged with time, pulls on the mass alone, and the
difference between parcels' masses is their buoyancy; the temperature's drive, wedged with entropy, and the salt's,
wedged with salt, are each mapped to their amount's flux, at its own rate. The flow carries whole parcels, and the
entropy and the salt it carries out of a cell are what the planes `mv.s` and `mv.n` read off that one flux. The water's
flow is a stream function on the vertices; its mass resists the flow's change and its viscosity the flow's turning, and
the pressure, which only keeps the flow from filling or emptying any cell, does no work on it and drops out.

Lengths are in units of the tank's depth, times in units of the time heat takes to conduct across it, and temperature
and salinity in units of their differences between the top and the bottom; the density is one, and the heat a unit of
entropy holds per volume is one degree.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, and the volume, the entropy and the salt a parcel holds, each a direction of its own with no metric.
spacetime = Algebra("x+y+t-")
ga = spacetime * Algebra("v0s0n0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# The planes reading a place's run along the tank and up it.
Upright = ga.gatype.from_blades("x y")                                        # Vector+
Drive = ga.gatype.from_blades("xt yt xv yv xs ys xn yn")                      # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# The tank's two directions, along its length and up its depth.
length = mv.x.dual()                                                          # [] Point-
height = mv.y.dual()                                                          # [] Point-
time = -mv.t.dual()                                                           # [] Point-
# A unit of volume, the weight of a place, a unit of entropy and a unit of salt.
volume = mv.v.dual()                                                          # [] Point-
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


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Point, Plane]:
    """Each triangle's area `Scalar+[faces]`, the side facing each of its corners `[corners] Point-[faces]`, and each
    corner's hat gradient `[corners] Plane+[faces]`, the plane through the facing side, along time, volume, entropy and
    salt, over twice the area, so that it reads one from the side to the corner."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time, at unit volume, entropy and salt.
    double_areas = ((((sides[0] & sides[1]) & time) & volume) & entropy) & salt   # Scalar-[faces]
    gradients = (time & volume & entropy & salt & sides) / double_areas       # [corners] Plane+[faces]
    return double_areas.norm() / 2, sides, gradients


def fingering(topology: Topology, areas: Scalar, sides: Point, gradients: Plane, free: np.ndarray, density: float,
              viscosity: float, expansion: float, contraction: np.ndarray, salt_diffusivity: float, gravity: float,
              temperature: Scalar, salinity: Scalar, dt: float, n_steps: int,
              n_stride: int) -> Iterator[tuple[Scalar, Scalar, Scalar]]:
    """The tank's stream function, temperature and salinity `[...] Scalar+[vertices]` every `n_stride` steps of `dt`,
    from rest and this temperature and salinity, for each contraction of the water by its salt `[...]`, the stream
    function nothing and the temperature and salinity held at the walls, free at the vertices `free`: each step
    gravity's drive does work on the flows the stream function can make, the water's mass and viscosity resist them,
    and the flow carries the entropy and the salt along while each one's drive spreads it at its own rate."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    interior = topology.reading(free)                                         # Scalar+[free <- vertices]
    # Each vertex's share of the area around it, and each triangle's mean over its corners.
    cells = (~corners * (areas / 3)).sum(axis=0)                              # Scalar+[vertices]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[faces <- vertices]
    # A stream function rising by one at a corner carries the water along the facing side, half the side's worth over
    # the triangle: each corner's flow, and its velocity.
    flows = sides / 2                                                         # [corners] Point-[faces]
    velocities = flows / areas                                                # [corners] Point-[faces]
    # The kinetic form: the water's inertia reads a velocity's run along and up into its momentum, paired with the
    # velocity of each other corner over the triangle, and gathered onto the vertices.
    inertia = density * (stack((mv.x, mv.y)) * (stack((mv.x, mv.y)) & Point)).sum(axis=0)   # [] Plane+ <- Point-
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
    heat_sweeps = stack((length, height)) & entropy                           # [2] Flux-
    salt_sweeps = stack((length, height)) & salt                              # [2] Flux-
    conductivity = (heat_sweeps * (heat_sweeps & Drive)).sum(axis=0)          # [] Flux- <- Drive+
    diffusivity = salt_diffusivity * (salt_sweeps * (salt_sweeps & Drive)).sum(axis=0)   # [] Flux- <- Drive+
    # What a flux carries out of each corner's cell: its hat gradient wedged with the flux, read for its entropy or its
    # salt, gathered onto the vertices.
    local = areas * (mv.s & (gradients[:, None] ^ conductivity(heat_drives[None, :])))   # [corners, corners] Scalar+[faces]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    local = areas * (mv.n & (gradients[:, None] ^ diffusivity(salt_drives[None, :])))   # [corners, corners] Scalar+[faces]
    diffusion = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    # Each step's implicit maps on the free vertices: the flow's inertia with its viscosity, the cells' heat with its
    # conduction, and their salt with its slower spreading.
    flow_step = (interior * (kinetic / dt + viscous) * ~interior) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    heat_step = (interior * (spdiag(cells / dt) + conduction) * ~interior) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    salt_step = (interior * (spdiag(cells / dt) + diffusion) * ~interior) * Scalar   # (Scalar+ <- Scalar+)[free <- free]
    pull = -(gravity * mv.yt)                                                 # [] Drive+
    # The temperature and salinity the walls hold, and the heat and salt they conduct into the free vertices.
    wall_temperature = temperature - ~interior * (interior * temperature)     # [...] Scalar+[vertices]
    wall_salinity = salinity - ~interior * (interior * salinity)              # [...] Scalar+[vertices]
    wall_heat = interior * (conduction * wall_temperature)                    # [...] Scalar+[free]
    wall_salt = interior * (diffusion * wall_salinity)                        # [...] Scalar+[free]
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            yield stream, temperature, salinity
        # Each triangle's parcels per volume: unit volume, holding less mass the warmer and more the saltier they are,
        # and their entropy and salt.
        warmth = corner_mean * temperature                                    # [...] Scalar+[faces]
        saltiness = corner_mean * salinity                                    # [...] Scalar+[faces]
        parcels = (volume + (density - expansion * warmth + contraction * saltiness) * time + warmth * entropy
                   + saltiness * salt)                                        # [...] Point-[faces]
        load = (~corners * (pull & (flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        stream = ~interior * flow_step.solve(interior * (kinetic / dt * stream + load))
        # The flow carries the parcels along: what their flux carries out of each corner's cell leaves it, its entropy
        # and its salt alike, and each one's drive spreads the rest, implicitly.
        velocity = ((corners * stream[..., None]) * velocities).sum(axis=-1)   # [...] Point-[faces]
        flux = velocity & parcels                                             # [...] Flux-[faces]
        outflow = (~corners * (areas * (gradients ^ flux[..., None]))).sum(axis=-1)   # [...] Point-[vertices]
        temperature = wall_temperature + ~interior * heat_step.solve(
            interior * (cells / dt * temperature - (mv.s & outflow)) - wall_heat)
        salinity = wall_salinity + ~interior * salt_step.solve(
            interior * (cells / dt * salinity - (mv.n & outflow)) - wall_salt)
