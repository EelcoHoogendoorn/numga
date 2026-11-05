"""Electrokinetic transport through a charged channel, on a mesh of triangles, in planar spacetime joined with a
direction for volume.

A charged wall draws a cloud of counter-ions into the liquid beside it, the double layer. A pressure drop along the
channel pushes the liquid and carries the cloud with it, a streaming current; a field along it pushes the cloud, which
drags the liquid along, electro-osmosis; and each ion drifts through the liquid under the field, conduction. Where the
walls change along the channel, the liquid finds its own way between them, and the potential it adds keeps the current
the same through every cross-section.

The algebra is planar spacetime, x+y+t-, joined with one degenerate direction for the volume a flow carries:
x+y+t-v0. A parcel of the solution is a point whose weight is its volume, `mv.v.dual()`, offset along time by the
charge it holds; an ion, charge without volume, is a point at infinity along time. A drive is a field of bivectors:
minus the gradient of the potentials, each wedged with the direction of what it acts on, the pressure with volume and
the electric potential with time, `-(gradient ^ (pressure * mv.v + potential * mv.t))`. A drive's commutator with a
parcel is the force on it, `drive.commutator(parcel)`, and a drive reads a flux by the regressive product, `drive &
flux`, the power it spends. The liquid's flow is a stream function on the vertices, whose rise along an edge is the
volume crossing it: a stream function rising by one at a corner of a triangle carries the liquid along the side facing
that corner. Viscosity resists the flow by its vorticity; the drives do work on it by reading what it carries; and the
potential the liquid adds is the one that every potential's drive reads the channel's flux as nothing, so that no
charge piles up anywhere. The ions stay in their local equilibrium, the Boltzmann distribution of the double layer.

Lengths are in Debye lengths, potentials in thermal volts and charges per area in units that make the permittivity
one; the viscosity is one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology
from examples.fluids.electrokinetics.numerical import solve_coupled

# Planar spacetime, and the volume a flow carries, a direction of its own with no metric.
spacetime = Algebra("x+y+t-")
ga = spacetime * Algebra("v0")
# The plane of the channel.
space = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# A drive: the field, the electric potential's gradient wedged with time, and the pressure's, wedged with volume.
Drive = ga.gatype.bivector().derive.touching(space, Algebra("t-v0"))          # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
time = -mv.t.dual()                                                           # [] Point-
# A unit of volume, the weight of a point.
volume = mv.v.dual()                                                          # [] Point-
# The patch the plane of the flow spans, and the liquid's local turning in it read back as the moment that resists it,
# per unit of viscosity.
patch = axis & across                                                         # [] Patch-
Rotation = ga.gatype.bivector().derive.within(space)                          # Bivector+
resisting = patch * (Rotation & patch)                                        # [] Patch- <- Bivector+
# A velocity read as the plane that takes its circulation along a path: its run along and across, read back.
circulating = (stack((mv.x, mv.y)) * (stack((mv.x, mv.y)) & Point)).sum(axis=0)   # [] Plane+ <- Point-
# A gradient read along and across as the flux it drives down itself at a unit diffusivity: Fick's law.
spreading = (stack((axis, across)) * (Plane & stack((axis, across)))).sum(axis=0)  # [] Point- <- Plane+


# --- plumbing -------------------------------------------------------------------------
def strip(n_along: int, n_across: int, periods: np.ndarray, half_widths: np.ndarray
          ) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strips of triangles closed on themselves along their length, `periods` long `[...]`: a chain across swept
    around a loop along, the walls at the half widths of the columns `[..., along]`. Their topology; their vertices'
    places, along and across `[..., vertices, 2]`; how much further along each edge reaches than its ends' places, a
    period back where it crosses from the last column to the first, `[..., edges]`; and the vertices of the lower and of
    the upper wall `[along]`."""
    topology = Topology.chains([n_across]).product(Topology.loops([n_along]))
    places = np.multiply.outer(periods, np.arange(n_along) / n_along)        # [..., along]
    heights = np.multiply.outer(half_widths, np.linspace(-1, 1, n_across))   # [..., along, across]
    coordinates = np.stack(np.broadcast_arrays(places[..., None], heights), axis=-1)
    tails, heads = topology.elements[1].T
    steps = heads // n_across - tails // n_across
    wraps = np.where(np.abs(steps) > 1, -np.sign(steps), 0)
    lower = np.arange(n_along) * n_across
    return (topology, coordinates.reshape(heights.shape[:-2] + (-1, 2)), np.multiply.outer(periods, wraps), lower,
            lower + n_across - 1)


def wall_charge(n_across: int, periods: np.ndarray, half_widths: np.ndarray, surface_charges: np.ndarray
                ) -> np.ndarray:
    """The charge the walls put on each vertex `[..., vertices]`: each wall vertex holds its column's charge per area
    `[..., along]` over half of each wall edge beside it, the strips `periods` long `[...]` and their walls the columns'
    half widths `[..., along]` from the middle."""
    n_along = half_widths.shape[-1]
    # Each wall edge's length, from a column to the next around the strip.
    lengths = np.hypot(periods[..., None] / n_along, np.roll(half_widths, -1, axis=-1) - half_widths)   # [..., along]
    shares = surface_charges * (lengths + np.roll(lengths, 1, axis=-1)) / 2                  # [..., along]
    on_walls = np.zeros(shares.shape + (n_across,))
    on_walls[..., 0] = on_walls[..., -1] = shares
    return on_walls.reshape(shares.shape[:-1] + (-1,))


def placed(topology: Topology, coordinates: np.ndarray, shifts: np.ndarray) -> tuple[Point, Point]:
    """A strip's vertices, places of unit volume `[...] Point-[vertices]`, and its edges `[...] Point-[edges]`."""
    vertices = (volume + axis * coordinates[..., 0] + across * coordinates[..., 1]).field()
    return vertices, topology.d[0] * vertices + (axis * shifts).field()


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, edges: Point) -> tuple[Scalar, Point, Plane]:
    """Twice each triangle's area `[...] Scalar-[faces]`, the side facing each of its corners `[..., corners]
    Point-[faces]`, and each corner's hat gradient `[..., corners] Plane+[faces]`: the plane through the facing side,
    along time and at unit volume, over twice the area, so that it reads one from the side to the corner."""
    sides = topology.boundary[2, 1] * edges[..., None]                       # [..., corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time at unit volume.
    double_areas = ((sides[..., 0] & sides[..., 1]) & time) & volume         # [...] Scalar-[faces]
    return double_areas, sides, (time & volume & sides) / double_areas[..., None]


def assembled(topology: Topology, local: Scalar) -> SparseExtensor:
    """A form on each triangle's corners `[..., corners, corners] Scalar+[faces]` gathered onto the vertices it joins,
    `Scalar+[..., vertices <- vertices]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=-1).sum(axis=-1)


def double_layer(outflow: SparseExtensor, cells: Scalar, wall_charge: Scalar, valences: np.ndarray,
                 bulk: np.ndarray, n_steps: int) -> Scalar:
    """The potential over the channel `[...] Scalar+[vertices]`, where the field's outflow from each vertex's cell,
    `outflow`, balances the cell's charge, that of its ions in their Boltzmann distribution and, on the walls, the
    walls' own: Gauss's law, solved by a damped Newton's method from the bulk's potential."""
    potential = 0 * wall_charge                                               # [...] Scalar+[vertices]
    for _ in range(n_steps):
        concentrations = bulk * (-valences * potential[..., None]).exp()      # [..., species] Scalar+[vertices]
        residual = outflow * potential - cells * (valences * concentrations).sum(axis=-1) - wall_charge
        # The charge a cell loses as its potential rises, by its ions' squared valences.
        susceptibility = (valences ** 2 * concentrations).sum(axis=-1)        # [...] Scalar+[vertices]
        step = ((outflow + cells * susceptibility) * Scalar).solve(residual)             # [...] Scalar+[vertices]
        # Newton's step, held below a thermal volt so that no Boltzmann factor changes by much more than e at once.
        potential = potential - step.tanh()
    return potential


def transport(topology: Topology, areas: Scalar, sides: Point, gradients: Plane, circulation: SparseExtensor,
              cells: Scalar, potential: Scalar, valences: np.ndarray, bulk: np.ndarray, viscosity: float,
              mobilities: np.ndarray, periods: np.ndarray, lower: np.ndarray, upper: np.ndarray, drives: Drive
              ) -> tuple[Scalar, Scalar, Flux, Point, Drive, Scalar]:
    """For each drive applied along the channel `[...]`, broadcast against the channels: the stream function
    `[...] Scalar+[vertices]`, the potential the liquid adds `[...] Scalar+[vertices]`, what the channel carries along
    per length `[...] Flux-`, and in each triangle the liquid's velocity `[...] Point-[faces]`, the whole drive
    `[...] Drive+[faces]` and the conductivity `[...] Scalar+[faces]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    # The ions in each triangle, the mean of its corners'.
    concentrations = corners.mean(axis=0) * (bulk * (-valences * potential[..., None]).exp())   # [..., species] Scalar+[faces]
    parcels = volume + (time * valences * concentrations).sum(axis=-1)       # [...] Point-[faces]
    # An ion of valence z conducts as z squared times the flux of a unit charge drifting under a drive.
    conductivity = (mobilities * valences ** 2 * concentrations).sum(axis=-1)   # [...] Scalar+[faces]
    # A stream function rising by one at a corner carries the liquid along the facing side, half the side's worth
    # over the triangle; a potential rising by one there drives the field down its hat's gradient, through time.
    flows = sides / 2                                                         # [..., corners] Point-[faces]
    pushes = -(gradients ^ mv.t)                                              # [..., corners] Bivector+[faces]
    conducts = pushes.commutator(time).regressive(time)                       # [..., corners] Flux-[faces]
    # Viscosity resists a flow by its vorticity, its circulation around each vertex's cell over the cell's area.
    viscous = viscosity * circulation * spdiag(1 / cells) * circulation       # Scalar+[..., vertices <- vertices]
    # A potential's drive reads what a flow carries, and what the ions conduct under another potential's drive.
    carried = flows & parcels[..., None]                                      # [..., corners] Flux-[faces]
    coupling = assembled(topology, carried[..., :, None] & pushes[..., None, :])
    conduction = assembled(topology, (pushes[..., :, None] & conducts[..., None, :]) * (conductivity * areas)[..., None, None])
    flow_load = (~corners * (drives[..., None] & carried)).sum(axis=-1)      # [...] Scalar+[vertices]
    applied_current = conductivity * drives.commutator(time).regressive(time)   # [...] Flux-[faces]
    current_load = (~corners * ((pushes & applied_current[..., None]) * areas[..., None])).sum(axis=-1)
    stream, added = solve_coupled(viscous, coupling, conduction, flow_load, current_load, topology.interior[0], upper,
                                  int(lower[0]))
    # The channel's flux per length: the parcels with the flow, the ions with their drift under the whole drive.
    velocities = ((corners * stream[..., None]) * sides).sum(axis=-1) / (2 * areas)   # [...] Point-[faces]
    field = drives + ((corners * added[..., None]) * pushes).sum(axis=-1)    # [...] Drive+[faces]
    flux = velocities.regressive(parcels) + conductivity * field.commutator(time).regressive(time)   # [...] Flux-[faces]
    return stream, added, (flux * areas).sites.sum() / periods, velocities, field, conductivity


def entropy_production(topology: Topology, areas: Scalar, circulation: SparseExtensor, cells: Scalar, viscosity: float,
                       stream: Scalar, field: Drive, conductivity: Scalar) -> tuple[Scalar, Scalar]:
    """The heat the liquid makes per volume at each vertex, the temperature times the entropy it produces, under the
    flow `[...] Scalar+[vertices]` and the whole drive in each triangle `[...] Drive+[faces]`: by its turning, the
    rate read by the viscous moment that resists it, and by its ions' drift, the drive read by the current it conducts;
    `[...] Scalar+[vertices]` each."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    # The liquid's turning: its circulation around each vertex's cell over the cell's area, a rate in the plane of the
    # flow.
    rotation = mv.xy * ((circulation * stream) / cells)                       # [...] Bivector+[vertices]
    viscous = rotation & (viscosity * resisting(rotation))                    # [...] Scalar+[vertices]
    joule = field & (conductivity * field.commutator(time).regressive(time))  # [...] Scalar+[faces]
    return viscous, (~corners.sum(axis=0) * (joule * areas / 3)) / cells


def equilibrium_pushes(topology: Topology, areas: Scalar, gradients: Plane, cells: Scalar, potential: Scalar,
                       valences: np.ndarray, bulk: np.ndarray) -> tuple[Point, Point]:
    """The double layer's own pushes on the liquid at each vertex, its triangles' weighed by their areas, `[...]
    Point-[vertices]` each: its osmotic pressure's drive, the pressure's gradient wedged with volume, on the parcels'
    volume, and its own field's drive, the potential's gradient wedged with time, on their charge."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    concentrations = bulk * (-valences * potential[..., None]).exp()          # [..., species] Scalar+[vertices]
    # The osmotic pressure: the ions' excess over the bulk's, each pushing as hard as a molecule of gas.
    osmotic = (concentrations - bulk).sum(axis=-1)                            # [...] Scalar+[vertices]
    parcels = volume + (time * valences * (corners.mean(axis=0) * concentrations)).sum(axis=-1)   # [...] Point-[faces]
    pushes = []
    for pressure, label in ((osmotic, mv.v), (potential, mv.t)):
        drive = -(((corners * pressure[..., None]) * gradients).sum(axis=-1) ^ label)   # [...] Bivector+[faces]
        pushes.append((~corners.sum(axis=0) * (drive.commutator(parcels) * (areas / 3))) / cells)
    return pushes[0], pushes[1]


def dyeing(topology: Topology, areas: Scalar, gradients: Plane, velocities: Point, field: Drive, charges: np.ndarray,
           mobility: float, diffusivity: float, dye: Scalar, dt: float, n_frames: int, stride: int) -> Iterator[Scalar]:
    """The dye over the channel `[...] Scalar+[vertices]`, this dye and then `n_frames` frames every `stride` steps of
    `dt`, in each run `[...]`: carried by the liquid at its velocity `[...] Point-[faces]` and drifting under the whole
    drive `[...] Drive+[faces]` by the charge it holds `[...]`, at this mobility, while it spreads at this diffusivity;
    each step halfway implicit, so that the dye moves at its own pace."""
    # A dye molecule is a parcel of charge without volume, a point at infinity along time: the drive pushes it by the
    # charge it holds, and it drifts at its mobility through the liquid that carries it.
    drifts = velocities + mobility * field.commutator(charges * time)        # [...] Point-[faces]
    # How much of a triangle's dye each corner's cell gains per unit time, the drift read by the corner's hat gradient,
    # from each corner's share of the triangle's dye.
    towards = areas[..., None] * (gradients & drifts[..., None])             # [..., corners] Scalar+[faces]
    carrying = assembled(topology, towards[..., :, None] * np.full(3, 1 / 3))   # Scalar+[..., vertices <- vertices]
    # The overlaps of the corners' hats over each triangle, whose sum over a vertex's triangles is its cell.
    overlaps = assembled(topology, areas[..., None, None] * (np.ones((3, 3)) + np.eye(3)) / 12)
    # How much each cell loses to its neighbours as the dye spreads: each hat's gradient reading the flux another's
    # drives down itself.
    diffusion = assembled(topology, areas[..., None, None] * (gradients[..., :, None] & spreading(gradients[..., None, :])))
    changing = diffusivity * diffusion - carrying                            # Scalar+[..., vertices <- vertices]
    # Each step halfway implicit: half a step forward from the dye before it, and half a step back from the dye after it.
    forward = overlaps / dt - changing / 2                                   # Scalar+[..., vertices <- vertices]
    backward = (overlaps / dt + changing / 2) * Scalar                       # (Scalar+ <- Scalar+)[..., vertices <- vertices]
    yield dye
    for _ in range(n_frames):
        for _ in range(stride):
            dye = backward.solve(forward * dye)
        yield dye
