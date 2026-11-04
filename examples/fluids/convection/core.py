"""Rayleigh-Bénard convection in a layer of liquid heated from below, on a mesh of triangles, in the plane joined with
three directions, for the volume, the mass and the entropy a parcel of liquid holds, with the layer's energy budget.

A layer of liquid between a warm floor and a cool ceiling carries heat up by conduction while it is still. Warm liquid
is lighter, so the still layer is top-heavy; once buoyancy beats viscosity and conduction, which the Rayleigh number
measures, the layer turns over in rolls that carry the heat up themselves.

The algebra is the plane, x+y+, joined with three degenerate directions, for volume, mass and entropy: x+y+v0m0s0. A
parcel is a point weighed by its volume, `mv.v.dual()`, offset along the mass's direction by its mass and along the
entropy's by its entropy; a unit volume holds a little less mass the more entropy it holds. A drive is minus the
gradient of a potential wedged with what it acts on, and pairs with the flux of that quantity: gravity's drive, the
geopotential's gradient wedged with mass, pulls on the mass a flow carries, and the temperature's drive, wedged with
entropy, conducts entropy, a conductivity mapping it to the entropy's flux. The liquid's flow is a stream function on
the vertices; its mass resists the flow's change and its viscosity the flow's turning, and the pressure, which only
keeps the flow from filling or emptying any cell, does no work on it and drops out.

Gravity's drive reads a flux of parcels, `flows & parcels`, as the work it does on the flow, and the parcels' places
joined with them, `places & parcels`, as minus their potential energy: one pairing, so that the work the flow gains is
the potential energy its carrying releases whenever the force and the carrying read the same flow and the same
temperature. Each step reads both at its midpoint, the temperature there foreseen from the last step, and the flow's
inertia changes the kinetic energy by just the work done there; viscosity and conduction are pullbacks of positive
materials. The budget, the kinetic and potential energy with what viscosity dissipates and what the heat moves, closes
each step to round-off, and the entropy produced never falls.

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

# The plane, and the volume, the mass and the entropy a parcel holds, each a direction of its own with no metric.
space = Algebra("x+y+")
amounts = Algebra("v0m0s0")
ga = space * amounts
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# The planes reading a place's run along the layer and up it.
Upright = Plane.derive.within(space)                                          # Vector+
Drive = ga.gatype.bivector().derive.touching(space, amounts)                  # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# The layer's two directions, along its length and up its depth, and the planes reading a direction's run along each.
length = mv.x.dual()                                                          # [] Point-
height = mv.y.dual()                                                          # [] Point-
directions = stack((length, height))                                          # [2] Point-
readings = stack((mv.x, mv.y))                                                # [2] Plane+
# A unit of volume, the weight of a place, a unit of mass and a unit of entropy.
volume = mv.v.dual()                                                          # [] Point-
mass = mv.m.dual()                                                            # [] Point-
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
    closing = opened.reading(np.arange(closed.n_vertices), 0)                # numbers[closed <- opened]
    opening = closed.reading(np.arange(opened.n_vertices) % closed.n_vertices, 0)   # numbers[opened <- closed]
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
    # Twice a triangle's area: two of its sides joined, at unit volume, mass and entropy.
    double_areas = (((sides[0] & sides[1]) & volume) & mass) & entropy        # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    # A corner's hat gradient: the plane through the facing side, along volume, mass and entropy, over twice the
    # area, so that it reads one from the side to the corner.
    gradients = (volume & mass & entropy & sides) / double_areas              # [corners] Plane+[faces]
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


class Budget(NamedTuple):
    """The layer's energy budget at a state, `[...] Scalar+` each: the flow's kinetic energy and the layer's potential
    energy, and, over the steps that led to the state, the energy viscosity dissipated, the potential energy the heat
    moved besides the flow's carrying, and the entropy produced."""
    kinetic: Scalar
    potential: Scalar
    dissipated: Scalar
    conducted: Scalar
    produced: Scalar


def convecting(liquid: Forms, interior: SparseExtensor, places: Point, density: float, expansion: float,
               gravity: np.ndarray, reference_temperature: float, heat_capacity: float, temperature: Scalar, dt: float,
               n_steps: int, n_stride: int) -> Iterator[tuple[Scalar, Scalar, Budget]]:
    """The layer's stream function and temperature `[...] Scalar+[vertices]`, with its budget, every `n_stride` steps
    of `dt`, from rest and this temperature, its vertices at these places `Point-[vertices]`, under each gravity
    `[...]`, at this absolute temperature and with this heat a unit volume holds per degree, in the units of the
    flow's energy, the stream function nothing and the temperature held at the walls, all but
    the vertices `interior` reads: each step gravity's drive does work on the flows the stream function can make, the
    liquid's mass and viscosity resist them, and the flow carries the entropy along while the temperature's drive
    conducts it, all at the step's midpoint."""
    # Each half step's implicit maps, solved with the walls held: the flow's inertia with its viscosity, and the cells'
    # heat with its conduction.
    flow_step = holding(liquid.kinetic / (dt / 2) + liquid.viscous, interior)
    heat_step = holding(spdiag(liquid.cells / (dt / 2)) + liquid.conduction, interior)
    # A field's values at the free vertices, nothing at the walls, and the temperature the walls hold.
    held = ~interior * interior                                               # Scalar+[vertices <- vertices]
    wall_temperature = temperature - held * temperature                       # [...] Scalar+[vertices]
    pull = -(gravity * mv.ym)                                                 # [...] Drive+
    # Gravity's reading of a degree of warmth at each vertex: the entropy it adds and the mass it takes away, joined
    # with the vertex's place.
    lift = pull & (places & (entropy - expansion * mass))                     # [...] Scalar+[vertices]
    # The liquid at rest; the midpoint temperature of the step before the first, the start's; and nothing yet
    # dissipated, moved or produced.
    stream = 0 * temperature                                                  # [...] Scalar+[vertices]
    middle_temperature = temperature                                          # [...] Scalar+[vertices]
    dissipated = conducted = produced = 0 * temperature.sites.sum()           # [...] Scalar+
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            # The flow's kinetic energy, and the layer's potential energy, minus gravity's drive read on each vertex's
            # place joined with its parcel per volume.
            kinetic_energy = (stream * (liquid.kinetic * stream)).sites.sum() / 2   # [...] Scalar+
            vertex_parcels = (volume + (density - expansion * temperature) * mass
                              + temperature * entropy)                        # [...] Point-[vertices]
            potential_energy = -(liquid.cells * (pull & (places & vertex_parcels))).sites.sum()   # [...] Scalar+
            yield stream, temperature, Budget(kinetic_energy, potential_energy, dissipated, conducted, produced)
        # The step's midpoint temperature, foreseen as far past this state as the last step's midpoint lay before it.
        middle_temperature = 2 * temperature - middle_temperature
        # Each triangle's parcels per volume at the midpoint: unit volume, holding less mass the warmer they are, and
        # their entropy.
        warmth = liquid.corner_mean * middle_temperature                      # [...] Scalar+[faces]
        parcels = volume + (density - expansion * warmth) * mass + warmth * entropy   # [...] Point-[faces]
        # Gravity's drive does work on each corner's flow by pulling on the mass it carries, and the flow's inertia
        # and viscosity take it to the step's midpoint.
        load = (~liquid.corners * (pull[..., None] & (liquid.flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[vertices]
        middle_stream = flow_step(liquid.kinetic / (dt / 2) * stream + load, 0 * stream)
        # The midpoint's flow carries the same parcels' entropy along; what its flux carries out of each corner's cell
        # leaves it, and the temperature's drive conducts the rest.
        velocity = liquid.stream_velocity(middle_stream)                      # [...] Point-[faces]
        flux = velocity & (warmth * entropy)                                  # [...] Flux-[faces]
        carried = mv.s & liquid.cell_outflow(flux)                            # [...] Scalar+[vertices]
        middle_temperature = heat_step(liquid.cells / (dt / 2) * temperature - carried, wall_temperature)
        # The step's budget at its midpoint: the energy viscosity dissipates; the potential energy the heat moves
        # besides the flow's carrying, read off each cell's outflow of entropy, nothing at the walls, which hold their
        # temperature, less what the flow carries out; and the entropy viscosity and conduction produce at this
        # absolute temperature, conduction carrying heat by the liquid's capacity for it.
        dissipation = dt * (middle_stream * (liquid.viscous * middle_stream)).sites.sum()   # [...] Scalar+
        exchange = held * (liquid.conduction * middle_temperature + carried) - carried   # [...] Scalar+[vertices]
        dissipated = dissipated + dissipation
        conducted = conducted + dt * (lift * exchange).sites.sum()
        produced = produced + dissipation / reference_temperature + dt * heat_capacity * (
            middle_temperature * (liquid.conduction * middle_temperature)).sites.sum() / reference_temperature ** 2
        # The step ends as far past the midpoint as it began before it.
        stream, temperature = 2 * middle_stream - stream, 2 * middle_temperature - temperature
