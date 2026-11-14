"""Electrokinetic transport through triangulated charged channels in planar projective spacetime joined with volume.

A charged channel wall draws counter-ions into a screening cloud called the electric double layer. An axial pressure drop pushes the liquid and convects that charged cloud as a streaming current, while an axial electric field pulls the cloud and drags the liquid as electro-osmosis alongside ionic conduction. Where channel width or wall charge varies along the strip, the induced electric potential keeps total current uniform across every station while the liquid forms recirculation cells.

Planar projective spacetime places vertices at unit projective weight along `origin` (`w.dual()`), while a degenerate quantity direction `v` records transported fluid volume without mixing length and volume metrics. A unit-volume liquid parcel carries `volume` (`v.dual()`) together with ionic charge along `time`, whereas an ion or dye molecule is a material parcel carrying charge without volume. Thermodynamic drives take minus each spatial potential gradient wedged with its conjugate direction, `-(gradient ^ (pressure * v + potential * t))`, acting on parcels via `drive.commutator(parcel)` and spending power `drive & flux` on fluxes. Vertex stream values advance volume across triangle sides without accumulation, vorticity around dual cells resists the flow, and the induced potential enforces zero net current divergence against every corner's simplex gradient.

Lengths are measured in bulk Debye lengths, electric potentials in thermal volts, surface charges in permittivity times thermal volts per Debye length, and dynamic viscosity in unit reference viscosity.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology
from examples.fluids.electrokinetics.numerical import coupled_solution

# Planar projective spacetime for channel geometry and electrodynamics, joined with volume as a transported quantity.
spacetime = Algebra("x+y+t-w0")
quantities = Algebra("v0")
ga = spacetime * quantities
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Antiscalar = ga.gatype.antiscalar()                                           # Antiscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Upright = Plane.derive.within("x y")                                          # Vector+
Drive = ga.gatype.bivector().derive.touching("x y", "t v")                    # Bivector+
Rotation = ga.gatype.bivector().derive.within("x y")                          # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
origin = mv.w.dual()                                                          # [] Point-
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
time = -mv.t.dual()                                                           # [] Point-
volume = mv.v.dual()                                                          # [] Point-
unread = origin & time & volume                                               # [] Antitrivector-
# Spatial turning patch and viscous duality map from vorticity rate to resisting moment per unit viscosity.
patch = axis & across                                                         # [] Flux-
viscous_moment = patch * (Rotation & patch)                                   # [] Flux- <- Rotation+
# Euclidean velocity-to-circulation plane map and unit-diffusivity Fick gradient-to-flux map.
circulation_plane = (stack((mv.x, mv.y)) * (stack((mv.x, mv.y)) & Point)).sum(axis=0)  # [] Plane+ <- Point-
fick_flux = (stack((axis, across)) * (Plane & stack((axis, across)))).sum(axis=0)      # [] Point- <- Plane+


# --- plumbing -------------------------------------------------------------------------
def strip(
    n_along: int,
    n_across: int,
    periods: np.ndarray,
    half_widths: np.ndarray,
) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweeping a transverse chain of `n_across` vertices around a periodic loop of `n_along` stations forms a closed triangulated strip with vertex coordinates (`[..., V, 2]`), periodic seam shifts (`[..., E]`), and lower and upper wall vertex indices (`[n_along]`)."""
    topology = Topology.chains([n_across]).product(Topology.loops([n_along]))
    stations = np.multiply.outer(periods, np.arange(n_along) / n_along)       # [..., n_along]
    heights = np.multiply.outer(half_widths, np.linspace(-1, 1, n_across))    # [..., n_along, n_across]
    coordinates = np.stack(np.broadcast_arrays(stations[..., None], heights), axis=-1)
    tails, heads = topology.elements[1].T
    column_steps = heads // n_across - tails // n_across
    seam_wraps = np.where(np.abs(column_steps) > 1, -np.sign(column_steps), 0)
    lower_wall = np.arange(n_along) * n_across
    return (
        topology,
        coordinates.reshape(heights.shape[:-2] + (-1, 2)),
        np.multiply.outer(periods, seam_wraps),
        lower_wall,
        lower_wall + n_across - 1,
    )


def wall_charge(
    n_across: int,
    periods: np.ndarray,
    half_widths: np.ndarray,
    surface_charges: np.ndarray,
) -> np.ndarray:
    """Integrating `surface_charges` (`[..., n_along]`) over half of each adjacent wall segment assigns wall charge (`[..., V]`) to the boundary vertices and zero to interior vertices."""
    n_along = half_widths.shape[-1]
    # Segment length along the curved wall from each station to the next around the periodic loop.
    segment_lengths = np.hypot(periods[..., None] / n_along, np.roll(half_widths, -1, axis=-1) - half_widths)  # [..., n_along]
    vertex_shares = surface_charges * (segment_lengths + np.roll(segment_lengths, 1, axis=-1)) / 2             # [..., n_along]
    charges = np.zeros(vertex_shares.shape + (n_across,))
    charges[..., 0] = charges[..., -1] = vertex_shares
    return charges.reshape(vertex_shares.shape[:-1] + (-1,))


def strip_placement(topology: Topology, coordinates: np.ndarray, edge_shifts: np.ndarray) -> tuple[Point, Point]:
    """Channel `coordinates` (`[..., V, 2]`) and periodic `edge_shifts` (`[..., E]`) place vertices (`[...] Point-[V]`) at `origin` and form spatial edge displacements (`[...] Point-[E]`)."""
    vertices = (origin + axis * coordinates[..., 0] + across * coordinates[..., 1]).field()  # [...] Point-[V]
    edges = topology.d[0] * vertices + (axis * edge_shifts).field()                          # [...] Point-[E]
    return vertices, edges


# --- math -----------------------------------------------------------------------------
def triangle_geometry(topology: Topology, edges: Point) -> tuple[Scalar, Point, Upright]:
    """Oriented `edges` (`[...] Point-[E]`) determine each triangle's area (`[...] Scalar+[F]`), facing sides (`[..., 3] Point-[F]`), and spatial simplex gradients (`[..., 3] Upright+[F]`)."""
    sides = topology.boundary[2, 1] * edges[..., None]                        # [..., 3] Point-[F]
    # Joining two triangle sides past origin, time and volume gives twice the signed spatial area.
    double_areas = sides[..., 0] & sides[..., 1] & unread                     # [...] Antiscalar-[F]
    gradients = (unread & sides) / double_areas[..., None]                    # [..., 3] Upright+[F]
    return double_areas.norm() / 2, sides, gradients


def vertex_form(topology: Topology, corner_form: Scalar) -> SparseExtensor:
    """Pullback by triangle-corner incidence assembles a local `[..., 3, 3]` form (`Scalar+[F]`) into a vertex operator (`[...] Scalar+[V <- V]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    return (corners[:, None].adjugate() * spdiag(corner_form) * corners[None, :]).sum(axis=-1).sum(axis=-1)


def channel_operators(
    topology: Topology,
    edges: Point,
) -> tuple[Scalar, Point, Upright, SparseExtensor, SparseExtensor, Scalar]:
    """Triangle geometry on `edges` (`[...] Point-[E]`) determines areas (`[...] Scalar+[F]`), sides (`[..., 3] Point-[F]`), simplex gradients (`[..., 3] Upright+[F]`), dielectric outflow and stream circulation operators (`[...] Scalar+[V <- V]`), and lumped dual-cell areas (`[...] Scalar+[V]`)."""
    areas, sides, gradients = triangle_geometry(topology, edges)
    # Unit corner potentials drive electric fields -(gradients ^ t) whose vacuum excitations pair into cell outflow.
    corner_drives = -(gradients ^ mv.t)                                       # [..., 3] Drive+[F]
    excitations = corner_drives.commutator(time) & time                       # [..., 3] Flux-[F]
    outflow = vertex_form(topology, (corner_drives[..., :, None] & excitations[..., None, :]) * areas[..., None, None])
    # Unit corner stream values drive flow along facing sides whose Euclidean circulation pairs around dual cells.
    corner_velocities = sides / (2 * areas[..., None])                        # [..., 3] Point-[F]
    circulation = vertex_form(
        topology,
        (circulation_plane(corner_velocities[..., :, None]) & corner_velocities[..., None, :]) * areas[..., None, None],
    )
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    dual_areas = corners.sum(axis=0).adjugate() * (areas / 3)                 # [...] Scalar+[V]
    return areas, sides, gradients, outflow, circulation, dual_areas


def double_layer(
    outflow: SparseExtensor,
    dual_areas: Scalar,
    wall_charges: Scalar,
    valences: np.ndarray,
    bulk: np.ndarray,
    n_steps: int,
) -> Scalar:
    """Damped Newton iteration solves the Poisson-Boltzmann balance between dielectric `outflow` and ionic plus `wall_charges` for the equilibrium potential (`[...] Scalar+[V]`)."""
    potential = 0 * wall_charges                                              # [...] Scalar+[V]
    for _ in range(n_steps):
        concentrations = bulk * (-valences * potential[..., None]).exp()      # [...] Scalar+[V]
        residual = outflow * potential - dual_areas * (valences * concentrations).sum(axis=-1) - wall_charges
        # Differential charge capacitance per unit volume from Boltzmann ion screening.
        susceptibility = (valences ** 2 * concentrations).sum(axis=-1)        # [...] Scalar+[V]
        step = ((outflow + dual_areas * susceptibility) * Scalar).solve(residual)  # [...] Scalar+[V]
        # Hyperbolic tangent damping caps each update below one thermal volt so exponentials stay bounded.
        potential = potential - step.tanh()
    return potential


def transport(
    topology: Topology,
    areas: Scalar,
    sides: Point,
    gradients: Upright,
    circulation: SparseExtensor,
    dual_areas: Scalar,
    potential: Scalar,
    valences: np.ndarray,
    bulk: np.ndarray,
    viscosity: float,
    mobilities: np.ndarray,
    periods: np.ndarray,
    lower_wall: np.ndarray,
    upper_wall: np.ndarray,
    drives: Drive,
) -> tuple[Scalar, Scalar, Flux, Point, Drive, Scalar]:
    """Under applied `drives` (`[...] Drive+`), coupled Stokes vorticity balance and charge conservation determine the stream function and induced potential (`[...] Scalar+[V]`), period-averaged channel flux (`[...] Flux-`), triangle velocity (`[...] Point-[F]`), total drive (`[...] Drive+[F]`), and ionic conductivity (`[...] Scalar+[F]`)."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # Triangle ion concentrations interpolate from corner Boltzmann values to form charged fluid parcels.
    concentrations = corners.mean(axis=0) * (bulk * (-valences * potential[..., None]).exp())  # [...] Scalar+[F]
    parcels = volume + (time * valences * concentrations).sum(axis=-1)        # [...] Point-[F]
    conductivity = (mobilities * valences ** 2 * concentrations).sum(axis=-1) # [...] Scalar+[F]
    flows = sides / 2                                                         # [..., 3] Point-[F]
    corner_drives = -(gradients ^ mv.t)                                       # [..., 3] Drive+[F]
    corner_currents = corner_drives.commutator(time) & time                   # [..., 3] Flux-[F]
    # Viscous resistance penalizes dual-cell vorticity, while electro-hydrodynamic coupling pairs corner drives with advected parcels.
    viscous = viscosity * circulation * spdiag(1 / dual_areas) * circulation  # [...] Scalar+[V <- V]
    advected_fluxes = flows & parcels[..., None]                              # [..., 3] Flux-[F]
    coupling = vertex_form(topology, corner_drives[..., None, :] & advected_fluxes[..., :, None])
    conduction = vertex_form(
        topology, (corner_drives[..., :, None] & corner_currents[..., None, :]) * (conductivity * areas)[..., None, None])
    flow_load = (corners.adjugate() * (drives[..., None] & advected_fluxes)).sum(axis=-1)       # [...] Scalar+[V]
    applied_current = conductivity * (drives.commutator(time) & time)         # [...] Flux-[F]
    current_load = (corners.adjugate() * ((corner_drives & applied_current[..., None]) * areas[..., None])).sum(axis=-1)
    stream, induced_potential = coupled_solution(
        viscous, coupling, conduction, flow_load, current_load, topology.interior[0], upper_wall, int(lower_wall[0]))
    # Total channel flux sums advected fluid parcels and conductive ion drift under the combined applied and induced drives.
    velocities = ((corners * stream[..., None]) * sides).sum(axis=-1) / (2 * areas)             # [...] Point-[F]
    field = drives + ((corners * induced_potential[..., None]) * corner_drives).sum(axis=-1)    # [...] Drive+[F]
    flux = (velocities & parcels) + conductivity * (field.commutator(time) & time)              # [...] Flux-[F]
    return stream, induced_potential, (flux * areas).sites.sum() / periods, velocities, field, conductivity


def entropy_production(
    topology: Topology,
    areas: Scalar,
    circulation: SparseExtensor,
    dual_areas: Scalar,
    viscosity: float,
    stream: Scalar,
    field: Drive,
    conductivity: Scalar,
) -> tuple[Scalar, Scalar]:
    """Viscous vorticity dissipation and ionic Joule heating give the vertex entropy-production rates times temperature (`[...] Scalar+[V]`) under `stream` and `field`."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    rotation = mv.xy * ((circulation * stream) / dual_areas)                  # [...] Rotation+[V]
    viscous_heat = rotation & (viscosity * viscous_moment(rotation))          # [...] Scalar+[V]
    joule_density = field & (conductivity * (field.commutator(time) & time))  # [...] Scalar+[F]
    joule_heat = (corners.sum(axis=0).adjugate() * (joule_density * areas / 3)) / dual_areas  # [...] Scalar+[V]
    return viscous_heat, joule_heat


def equilibrium_forces(
    topology: Topology,
    areas: Scalar,
    gradients: Upright,
    dual_areas: Scalar,
    potential: Scalar,
    valences: np.ndarray,
    bulk: np.ndarray,
) -> tuple[Point, Point]:
    """Double-layer osmotic pressure and electrostatic potential gradients exert opposing body forces (`[...] Point-[V]`) on charged fluid parcels across the channel."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    concentrations = bulk * (-valences * potential[..., None]).exp()          # [...] Scalar+[V]
    osmotic_pressure = (concentrations - bulk).sum(axis=-1)                   # [...] Scalar+[V]
    parcels = volume + (time * valences * (corners.mean(axis=0) * concentrations)).sum(axis=-1) # [...] Point-[F]
    forces = []
    for scalar_potential, direction in ((osmotic_pressure, mv.v), (potential, mv.t)):
        drive = -(((corners * scalar_potential[..., None]) * gradients).sum(axis=-1) ^ direction)  # [...] Drive+[F]
        forces.append((corners.sum(axis=0).adjugate() * (drive.commutator(parcels) * (areas / 3))) / dual_areas)
    return forces[0], forces[1]


def dye_transport(
    topology: Topology,
    areas: Scalar,
    gradients: Upright,
    velocities: Point,
    field: Drive,
    charges: np.ndarray,
    mobility: float,
    diffusivity: float,
    dye: Scalar,
    dt: float,
    n_frames: int,
    stride: int,
) -> Iterator[Scalar]:
    """Yields the initial `dye` concentration (`[...] Scalar+[V]`) followed by `n_frames` Crank-Nicolson states every `stride` steps of `dt` under fluid `velocities`, electrophoretic drift in `field`, and Fickian diffusion."""
    # Charged dye molecules carry charge without volume, so only the electric part of the drive adds electrophoretic drift.
    drifts = velocities + mobility * field.commutator(charges * time)         # [...] Point-[F]
    corner_inflows = areas[..., None] * (gradients & drifts[..., None])       # [..., 3] Scalar+[F]
    advection = vertex_form(topology, corner_inflows[..., :, None] * np.full(3, 1 / 3))         # [...] Scalar+[V <- V]
    overlap_form = vertex_form(topology, areas[..., None, None] * (np.ones((3, 3)) + np.eye(3)) / 12)
    diffusion = vertex_form(
        topology, areas[..., None, None] * (gradients[..., :, None] & fick_flux(gradients[..., None, :])))
    transport_rate = diffusivity * diffusion - advection                      # [...] Scalar+[V <- V]
    forward_step = overlap_form / dt - transport_rate / 2                     # [...] Scalar+[V <- V]
    backward_step = (overlap_form / dt + transport_rate / 2) * Scalar         # [...] (Scalar+ <- Scalar+)[V <- V]
    yield dye
    for _ in range(n_frames):
        for _ in range(stride):
            dye = backward_step.solve(forward_step * dye)
        yield dye
