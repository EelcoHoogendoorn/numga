"""Rayleigh-Bénard convection in a layer of liquid heated from below, on a mesh of triangles, in planar spacetime
joined with two directions, for the volume and the entropy a parcel of liquid holds.

A layer of liquid between a warm floor and a cool ceiling carries heat up by conduction while it is still. Warm liquid
is lighter, so the still layer is top-heavy; once buoyancy beats viscosity and conduction, which the Rayleigh number
measures, the layer turns over in rolls that carry the heat up themselves.

The algebra is planar spacetime, x+y+t-, joined with two degenerate directions, one for volume and one for entropy:
x+y+t-v0s0. A parcel is a point weighed by its volume, `mv.v.dual()`, offset along time by its mass and along the
entropy's direction by its entropy; a unit volume holds a little less mass the more entropy it holds. A drive is minus
the gradient of a potential wedged with what it acts on, and pairs with the flux of that quantity: gravity's drive,
wedged with time, pulls on the mass a flow carries, and the temperature's drive, wedged with entropy, conducts entropy,
a conductivity mapping it to the entropy's flux. The liquid's flow is a stream function on the vertices; its mass
resists the flow's change and its viscosity the flow's turning, and the pressure, which only keeps the flow from
filling or emptying any cell, does no work on it and drops out.

Lengths are in units of the layer's depth, times in units of the time heat takes to conduct across it, and
temperatures in units of the difference between floor and ceiling; the density is one, and the heat a unit of entropy
holds per volume is one degree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, and the volume and the entropy a parcel holds, each a direction of its own with no metric.
spacetime = Algebra("x+y+t-")
ga = spacetime * Algebra("v0s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# The planes reading a place's run along the layer and up it.
Upright = ga.gatype.from_blades("x y")                                        # Vector+
Drive = ga.gatype.from_blades("xt yt xv yv xs ys")                            # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# The layer's two directions, along its length and up its depth, and the planes reading a direction's run along each.
length = mv.x.dual()                                                          # [] Point-
height = mv.y.dual()                                                          # [] Point-
directions = stack((length, height))                                          # [2] Point-
readings = stack((mv.x, mv.y))                                                # [2] Plane+
time = -mv.t.dual()                                                           # [] Point-
# A unit of volume, the weight of a place, and a unit of entropy.
volume = mv.v.dual()                                                          # [] Point-
entropy = mv.s.dual()                                                         # [] Point-


# --- plumbing -------------------------------------------------------------------------
def layer(n_along: int, n_across: int, period: float) -> tuple[Topology, Topology, SparseExtensor, SparseExtensor,
                                                                np.ndarray]:
    """A layer of triangles of unit depth closed on itself along its length, a chain up swept around a loop along, and
    the same layer opened at its seam, swept along a chain a column longer whose last column is the first again, a
    period further along: their topologies, whose triangles come in the same order; the maps reading a field on the
    open layer at the closed one's vertices and on the closed layer at the open one's; and the open layer's places
    along and up `[vertices, 2]`."""
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
    """The liquid's maps on a mesh's triangles: each corner's flow, the velocity of a stream function and the outflow of
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


def forms(closed: Topology, opened: Topology, vertices: Point, density: float, viscosity: float) -> Forms:
    """The liquid's maps, built once from a layer's triangles, their geometry read on the open layer and gathered onto
    the closed one's vertices: each corner's flow, a stream function's velocity, the outflow of a flux, the stream
    function's kinetic and viscous forms, and the conduction of heat, the outflow of the flux the temperature's drive
    drives."""
    sides = opened.boundary[2, 1] * (opened.d[0] * vertices)                  # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time, at unit volume and entropy.
    double_areas = (((sides[0] & sides[1]) & time) & volume) & entropy        # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    # A corner's hat gradient: the plane through the facing side, along time, volume and entropy, over twice the
    # area, so that it reads one from the side to the corner.
    gradients = (time & volume & entropy & sides) / double_areas              # [corners] Plane+[faces]
    corners = closed.incidence[2, 0]                                          # [corners] Scalar+[faces <- vertices]
    # Each vertex's share of the area around it, and each triangle's mean over its corners.
    cells = (~corners * (areas / 3)).sum(axis=0)                              # Scalar+[vertices]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[faces <- vertices]
    # A stream function rising by one at a corner carries the liquid along the facing side, half the side's worth
    # over the triangle: each corner's flow, and its velocity.
    flows = sides / 2                                                         # [corners] Point-[faces]
    velocities = flows / areas                                                # [corners] Point-[faces]
    # A stream function's velocity on each triangle, its corners' velocities weighed by its values there, and what
    # a flux carries out of each corner's cell: its hat gradient wedged with the flux, gathered onto the vertex.
    stream_velocity = (spdiag(velocities * Scalar) * corners).sum(axis=0)     # (Point- <- Scalar+)[faces <- vertices]
    cell_outflow = (~corners * spdiag(areas * (gradients ^ Flux))).sum(axis=0)   # (Point- <- Flux-)[vertices <- faces]
    # The kinetic form: the liquid's inertia reads a velocity's run along and up into its momentum, paired with the
    # velocity of each other corner over the triangle, and gathered onto the vertices.
    inertia = density * (readings * (readings & Point)).sum(axis=0)           # [] Plane+ <- Point-
    local = areas * (inertia(velocities[:, None]) & velocities[None, :])      # [corners, corners] Scalar+[faces]
    kinetic = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    # The vorticity at each vertex, the kinetic form's amount over its share of the area and the density, and the
    # viscous form, its square over the layer.
    vorticity = spdiag(1 / (density * cells)) * kinetic                       # Scalar+[vertices <- vertices]
    viscous = viscosity * ~vorticity * spdiag(cells) * vorticity              # Scalar+[vertices <- vertices]
    # The temperature's drive at each corner's hat, minus its gradient wedged with the entropy it pushes, and the
    # conductivity, the drive read across each of the layer's directions driving the entropy along it.
    drives = -(gradients ^ mv.s)                                              # [corners] Drive+[faces]
    conductivity = ((directions & entropy) * ((directions & entropy) & Drive)).sum(axis=0)   # [] Flux- <- Drive+
    # What a flux carries out of each corner's cell: its hat gradient wedged with the flux, read for its entropy,
    # gathered onto the vertices.
    local = areas * (mv.s & (gradients[:, None] ^ conductivity(drives[None, :])))   # [corners, corners] Scalar+[faces]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    return Forms(corners, corner_mean, flows, stream_velocity, cell_outflow, cells, kinetic, viscous, conduction)


def convecting(liquid: Forms, interior: SparseExtensor, density: float, expansion: float, gravity: np.ndarray,
               temperature: Scalar, dt: float, n_steps: int, n_stride: int) -> Iterator[tuple[Scalar, Scalar]]:
    """The layer's stream function and temperature `[...] Scalar+[vertices]` every `n_stride` steps of `dt`, from rest
    and this temperature, under each gravity `[...]`, the stream function nothing and the temperature held at the walls,
    all but the vertices `interior` reads: each step gravity's drive does work on the flows the stream function can
    make, the liquid's mass and viscosity resist them, and the flow carries the entropy along while the temperature's
    drive conducts it."""
    # Each step's implicit maps, solved with the walls held: the flow's inertia with its viscosity, and the cells' heat
    # with its conduction.
    flow_step = holding(liquid.kinetic / dt + liquid.viscous, interior)
    heat_step = holding(spdiag(liquid.cells / dt) + liquid.conduction, interior)
    # The temperature the walls hold.
    wall_temperature = temperature - ~interior * (interior * temperature)     # [...] Scalar+[vertices]
    pull = -(gravity * mv.yt)                                                 # [...] Drive+
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            yield stream, temperature
        # Each triangle's parcels per volume: unit volume, holding less mass the warmer they are, and their entropy.
        warmth = liquid.corner_mean * temperature                             # [...] Scalar+[faces]
        parcels = volume + (density - expansion * warmth) * time + warmth * entropy   # [...] Point-[faces]
        # Gravity's drive does work on each corner's flow by pulling on the mass it carries.
        load = (~liquid.corners * (pull[..., None] & (liquid.flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        stream = flow_step(liquid.kinetic / dt * stream + load, 0 * stream)
        # The flow carries the entropy along; what its flux carries out of each corner's cell leaves it, and the
        # temperature's drive conducts the rest, implicitly.
        velocity = liquid.stream_velocity(stream)                             # [...] Point-[faces]
        flux = velocity & (warmth * entropy)                                  # [...] Flux-[faces]
        outflow = liquid.cell_outflow(flux)                                   # [...] Point-[vertices]
        temperature = heat_step(liquid.cells / dt * temperature - (mv.s & outflow), wall_temperature)
