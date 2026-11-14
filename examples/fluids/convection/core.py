"""A layer of liquid heated from below turns over in Rayleigh-Bénard convection on a triangulated strip, in the plane
joined with three degenerate directions for volume, mass and entropy.

A layer of liquid between a warm floor and a cool ceiling carries heat up by conduction while it stays still. Warm
liquid is lighter, so the still layer is top-heavy; once buoyancy beats viscosity and conduction, as the Rayleigh number
measures, the layer turns over in rolls that carry heat up with the flow.

The algebra is the plane `x+ y+` joined with three degenerate directions for the quantities (`x+ y+ v0 m0 s0`), namely
volume, mass and entropy. A carrier's contents are modelled as its parcel, `volume + (density - expansion * warmth) *
mass + warmth * entropy`: a unit of volume `v.dual()` has less mass when it is warmer and has more entropy. A drive
`-gradient ^ potentials` is the fall of the potentials, minus their gradient wedged with them, and pairs with the flux
of each quantity. Gravity's drive, minus the geopotential's gradient wedged with mass, pulls on the advected mass, while
the temperature's drive, wedged with entropy, drives the conductive entropy flux through a conductivity. The liquid's
flow is modelled by a stream function on the vertices; its inertia resists changes in the flow and its viscosity resists
turning, while pressure keeps each cell incompressible and drops out of the work.

Gravity's drive pairs with a flux of parcels, `flows & parcels`, to give the work done on the flow, and pairs with the
parcels joined to their places, `places & parcels`, to give minus their potential energy. Because one pairing reads
both, the kinetic work gained by the flow equals the potential energy released by advection whenever both evaluate at
the same midpoint flow and temperature. Viscosity and conduction are pullbacks of positive materials, so the energy
budget closes to round-off and the entropy produced never falls.

Lengths are in units of the layer's depth, times in units of the thermal diffusion time across the layer, and
temperatures in units of the floor-to-ceiling difference; the reference density is one, and a unit of entropy per unit
volume corresponds to one degree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import Sparse, SparseExtensor, spdiag

from numga.topology import Topology

# The plane x+ y+ joined with three degenerate directions for the quantities: volume, mass and entropy.
space = Algebra("x+y+")
quantities = Algebra("v0m0s0")
ga = space * quantities
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
# Planes in the spatial sector that read horizontal and vertical offsets.
Upright = ga.gatype.vector().derive.within(space)                             # Vector+
Drive = ga.gatype.bivector().derive.touching(space, quantities)               # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# Horizontal and vertical directions, and the spatial planes that read them.
length = mv.x.dual()                                                          # [] Antivector-
height = mv.y.dual()                                                          # [] Antivector-
directions = stack((length, height))                                          # [2] Antivector-
readings = stack((mv.x, mv.y))                                                # [2] Vector+
# Units of the three quantities: volume, mass and entropy.
volume = mv.v.dual()                                                          # [] Antivector-
mass = mv.m.dual()                                                            # [] Antivector-
entropy = mv.s.dual()                                                         # [] Antivector-


# --- plumbing -------------------------------------------------------------------------
def layer(n_along: int, n_across: int, period: float) -> tuple[Topology, Topology, Sparse, Sparse,
                                                                np.ndarray]:
    """A periodic strip of triangles of unit depth closed along its length, together with the same strip opened at its
    seam one period further along: their topologies with matching triangle order, the vertex-reading maps between the
    closed and opened strips, and the opened strip's vertex coordinates."""
    closed = Topology.chains([n_across]).product(Topology.loops([n_along]))
    opened = Topology.chains([n_across]).product(Topology.chains([n_along + 1]))
    places = np.stack(np.meshgrid(np.arange(n_along + 1) / n_along * period, np.linspace(0, 1, n_across),
                                  indexing="ij"), axis=-1)
    closing = opened.reading(np.arange(closed.n_vertices), 0)                 # Scalar+[V <- V_opened]
    opening = closed.reading(np.arange(opened.n_vertices) % closed.n_vertices, 0)   # Scalar+[V_opened <- V]
    return closed, opened, closing, opening, places.reshape(-1, 2)


def holding(form: SparseExtensor, interior: Sparse) -> Callable[[Scalar, Scalar], Scalar]:
    """A vertex operator solved with fixed wall values: for a right-hand side and wall values,
    it solves on the free vertices and leaves the wall values unchanged."""
    free = (interior * form * interior.adjugate()) * Scalar                             # (Scalar+ <- Scalar+)[V_free <- V_free]
    return lambda rhs, walls: walls + interior.adjugate() * free.solve(interior * (rhs - form * walls))


# --- math -----------------------------------------------------------------------------
class Forms(NamedTuple):
    """Discrete operators on the triangulated layer: face-vertex incidences, corner flows, stream-function velocity,
    cell flux divergence, vertex cell areas, and the kinetic, viscous and conductive forms."""
    corners: Sparse                                                           # [3] Scalar+[F <- V]
    corner_mean: Sparse                                                       # Scalar+[F <- V]
    flows: Point                                                              # [3] Antivector-[F]
    stream_velocity: SparseExtensor                                           # (Antivector- <- Scalar+)[F <- V]
    cell_outflow: SparseExtensor                                              # (Antivector- <- Antibivector-)[V <- F]
    cells: Scalar                                                             # Scalar+[V]
    kinetic: SparseExtensor                                                   # Scalar+[V <- V]
    viscous: SparseExtensor                                                   # Scalar+[V <- V]
    conduction: SparseExtensor                                                # Scalar+[V <- V]


def forms(closed: Topology, opened: Topology, vertices: Point, density: float, viscosity: float) -> Forms:
    """Operators assembled once from the opened strip's triangle geometry onto the closed strip's vertices: corner
    flows, stream-function velocity, cell outflow, and the kinetic, viscous and conductive forms."""
    sides = opened.boundary[2, 1] * (opened.d[0] * vertices)                  # [3] Antivector-[F]
    # Twice each triangle's signed area: two sides joined with unit volume, mass and entropy.
    double_areas = (((sides[0] & sides[1]) & volume) & mass) & entropy        # Scalar-[F]
    areas = double_areas.norm() / 2                                           # Scalar+[F]
    # Corner gradients: the plane through the facing side and the quantity directions, divided by twice the area.
    gradients = (volume & mass & entropy & sides) / double_areas              # [3] Vector+[F]
    corners = closed.incidence[2, 0]                                          # [3] Scalar+[F <- V]
    # Each vertex's dual cell area, a third of each incident triangle, and the corner-averaging map onto faces.
    cells = (corners.adjugate() * (areas / 3)).sum(axis=0)                              # Scalar+[V]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[F <- V]
    # A unit stream function at a corner drives flow along the facing side, giving each corner's integrated flow and velocity.
    flows = sides / 2                                                         # [3] Antivector-[F]
    velocities = flows / areas                                                # [3] Antivector-[F]
    # Stream-function velocity on each face, and the outflow from each vertex cell gathered by wedging the corner gradients with a flux.
    stream_velocity = (spdiag(velocities * Scalar) * corners).sum(axis=0)     # (Antivector- <- Scalar+)[F <- V]
    cell_outflow = (corners.adjugate() * spdiag(areas * (gradients ^ Flux))).sum(axis=0)   # (Antivector- <- Antibivector-)[V <- F]
    # Kinetic form: inertia pairs corner velocities across each triangle and gathers onto the vertices.
    inertia = density * (readings * (readings & Point)).sum(axis=0)           # [] Vector+ <- Antivector-
    local = areas * (inertia(velocities[:, None]) & velocities[None, :])      # [3, 3] Scalar+[F]
    kinetic = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[V <- V]
    # Mass-lumped vorticity on vertex cells, and the viscous form from its cell-weighted square.
    vorticity = spdiag(1 / (density * cells)) * kinetic                       # Scalar+[V <- V]
    viscous = viscosity * ~vorticity * spdiag(cells) * vorticity              # Scalar+[V <- V]
    # Thermal drives at the corners, minus each simplex gradient wedged with entropy, and isotropic entropy conductivity.
    drives = -gradients ^ mv.s                                                # [3] Bivector+[F]
    conductivity = ((directions & entropy) * ((directions & entropy) & Drive)).sum(axis=0)   # [] Antibivector- <- Bivector+
    # Conductive entropy outflow: corner gradients wedged with the conductive flux and gathered onto the vertices.
    local = areas * (mv.s & (gradients[:, None] ^ conductivity(drives[None, :])))   # [3, 3] Scalar+[F]
    conduction = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[V <- V]
    return Forms(corners, corner_mean, flows, stream_velocity, cell_outflow, cells, kinetic, viscous, conduction)


class Budget(NamedTuple):
    """Energy and entropy budget at a yielded state, each a scalar: kinetic and potential energy, cumulative
    viscous dissipation, cumulative potential energy transferred by conduction and wall heat exchange, and cumulative
    entropy production."""
    kinetic: Scalar
    potential: Scalar
    dissipated: Scalar
    conducted: Scalar
    produced: Scalar


def convecting(liquid: Forms, interior: Sparse, places: Point, density: float, expansion: float,
               gravity: np.ndarray, reference_temperature: float, heat_capacity: float, temperature: Scalar, dt: float,
               n_steps: int, n_stride: int) -> Iterator[tuple[Scalar, Scalar, Budget]]:
    """Stream function and temperature with the energy budget at regular intervals, starting
    from rest with spatial vertex offsets, zero stream function on the walls, and fixed wall
    temperatures."""
    # Implicit midpoint solvers for vorticity-stream momentum and heat conduction with held walls.
    flow_step = holding(liquid.kinetic / (dt / 2) + liquid.viscous, interior)
    heat_step = holding(liquid.cells / (dt / 2) + liquid.conduction, interior)
    # Mask zeroing wall vertices, and the fixed wall temperature profile.
    held = interior.adjugate() * interior                                               # Scalar+[V <- V]
    wall_temperature = temperature - held * temperature                       # [...] Scalar+[V]
    pull = -gravity * mv.ym                                                   # [...] Bivector+
    # Gravitational potential energy change per unit warmth at each vertex, from thermal expansion of its mass.
    lift = pull & (places & (entropy - expansion * mass))                     # [...] Scalar+[V]
    # Initial rest state, initial midpoint temperature, and zeroed cumulative budget totals.
    stream = 0 * temperature                                                  # [...] Scalar+[V]
    middle_temperature = temperature                                          # [...] Scalar+[V]
    dissipated = conducted = produced = 0 * temperature.sites.sum()           # [...] Scalar+
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            # Kinetic energy of the flow, and gravitational potential energy of the vertex parcels.
            kinetic_energy = (stream * (liquid.kinetic * stream)).sites.sum() / 2   # [...] Scalar+
            vertex_parcels = (volume + (density - expansion * temperature) * mass
                              + temperature * entropy)                        # [...] Antivector-[V]
            potential_energy = -(liquid.cells * (pull & (places & vertex_parcels))).sites.sum()   # [...] Scalar+
            yield stream, temperature, Budget(kinetic_energy, potential_energy, dissipated, conducted, produced)
        # Extrapolate the midpoint temperature from the previous step's midpoint.
        middle_temperature = 2 * temperature - middle_temperature
        # Midpoint face parcels: unit volume, thermally expanded mass, and entropy.
        warmth = liquid.corner_mean * middle_temperature                      # [...] Scalar+[F]
        parcels = volume + (density - expansion * warmth) * mass + warmth * entropy   # [...] Antivector-[F]
        # Buoyant work on each corner's flow drives the stream function to the step's midpoint.
        load = (liquid.corners.adjugate() * (pull[..., None] & (liquid.flows & parcels[..., None]))).sum(axis=-1)   # [...] Scalar+[V]
        middle_stream = flow_step(liquid.kinetic / (dt / 2) * stream + load, 0 * stream)   # [...] Scalar+[V]
        # Midpoint flow advects face entropy; cell outflow and thermal conduction advance temperature to the midpoint.
        velocity = liquid.stream_velocity(middle_stream)                      # [...] Antivector-[F]
        flux = velocity & (warmth * entropy)                                  # [...] Antibivector-[F]
        advected = mv.s & liquid.cell_outflow(flux)                           # [...] Scalar+[V]
        middle_temperature = heat_step(liquid.cells / (dt / 2) * temperature - advected, wall_temperature)   # [...] Scalar+[V]
        # Midpoint budget increments: viscous dissipation, conductive potential-energy transfer, and entropy production.
        dissipation = dt * (middle_stream * (liquid.viscous * middle_stream)).sites.sum()   # [...] Scalar+
        exchange = held * (liquid.conduction * middle_temperature + advected) - advected   # [...] Scalar+[V]
        dissipated = dissipated + dissipation                                 # [...] Scalar+
        conducted = conducted + dt * (lift * exchange).sites.sum()            # [...] Scalar+
        produced = produced + dissipation / reference_temperature + dt * heat_capacity * (
            middle_temperature * (liquid.conduction * middle_temperature)).sites.sum() / reference_temperature ** 2
        # Complete the full step symmetrically past the midpoint.
        stream, temperature = 2 * middle_stream - stream, 2 * middle_temperature - temperature
