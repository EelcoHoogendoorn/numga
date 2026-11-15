"""Salt fingers in a triangulated tank of warm salty water over cold fresh water, in the projective plane joined with four degenerate directions for volume, mass, entropy and salt.

Warm water is light and salty water heavy, and with enough warmth on top the tank is stably stratified, lighter water over heavier. Even so, the interface breaks into narrow fingers: a salty parcel pushed down loses its warmth sideways to its cooler neighbours far faster than it loses its salt, becomes heavier than they are and keeps sinking, while a fresh parcel pushed up warms and keeps rising. Where the tank is stably stratified, the density ratio of thermal buoyancy to haline buoyancy exceeds one, and fingers grow whenever that ratio stays below the diffusivity ratio of heat to salt.

The projective plane places vertices at unit projective weight, while four degenerate directions record the volume, mass, entropy and salt a parcel carries. A unit-volume parcel carries slightly less mass when warmer and slightly more when saltier. Each thermodynamic drive is minus the spatial gradient of a potential wedged with the quantity it acts on, pairing directly with the flux of that quantity: gravity's drive wedges the geopotential gradient with mass and pulls on mass alone, whereas the thermal and haline drives map to entropy and salt fluxes at their own conductivities. Advection carries whole parcels `velocity & parcels`, and pairing the dual-cell outflow with `s` and `n` reads out the advected entropy and salt from one flux. A vertex stream function represents the incompressible flow, resisted by fluid inertia and viscous vorticity dissipation while pressure drops out.

Lengths are measured in tank depths, times in thermal diffusion times across the depth, and temperature and salinity in their top-to-bottom differences, at unit reference density and unit volumetric heat capacity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

from numga import Algebra, NumpyContext, stack
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

# The projective plane for spatial placement, joined with degenerate directions for volume, mass, entropy and salt.
plane = Algebra("x+y+")
space = Algebra("x+y+w0")
quantities = Algebra("v0m0s0n0")
ga = space * quantities
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Antiscalar = ga.gatype.antiscalar()                                           # Antiscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# Spatial coordinate planes and drives coupling spatial gradients to material quantities.
Upright = Plane.derive.within(plane)                                          # Vector+
Drive = ga.gatype.bivector().derive.touching(plane, quantities)               # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# Horizontal and vertical spatial directions and their conjugate coordinate planes.
length = mv.x.dual()                                                          # [] Point-
height = mv.y.dual()                                                          # [] Point-
directions = stack((length, height))                                          # [2] Point-
readings = stack((mv.x, mv.y))                                                # [2] Plane+
# Reference origin at unit projective weight, unit material quantities, and their joint complement for spatial areas.
origin = mv.w.dual()                                                          # [] Point-
volume = mv.v.dual()                                                          # [] Point-
mass = mv.m.dual()                                                            # [] Point-
entropy = mv.s.dual()                                                         # [] Point-
salt = mv.n.dual()                                                            # [] Point-
unread = volume & mass & entropy & salt                                       # [] Trivector-


# --- plumbing -------------------------------------------------------------------------
def wall_solver(form: SparseExtensor, interior: Sparse) -> Callable[[Scalar, Scalar], Scalar]:
    """Linear solver for a vertex operator with Dirichlet values held on the boundary vertices."""
    free = (interior * form * interior.adjugate()) * Scalar                   # (Scalar+ <- Scalar+)[V_free <- V_free]
    return lambda rhs, walls: walls + interior.adjugate() * free.solve(interior * (rhs - form * walls))


# --- math -----------------------------------------------------------------------------
class TankOperators(NamedTuple):
    """Discrete geometric and transport operators assembled on the closed tank topology."""
    corners: Sparse                                                           # [3] Scalar+[F <- V]
    corner_mean: Sparse                                                       # Scalar+[F <- V]
    flows: Point                                                              # [3] Point-[F]
    stream_velocity: SparseExtensor                                           # (Point- <- Scalar+)[F <- V]
    cell_outflow: SparseExtensor                                              # (Point- <- Flux-)[V <- F]
    dual_areas: Scalar                                                        # Scalar+[V]
    kinetic: SparseExtensor                                                   # Scalar+[V <- V]
    viscous: SparseExtensor                                                   # Scalar+[V <- V]
    conduction: SparseExtensor                                                # Scalar+[V <- V]
    diffusion: SparseExtensor                                                 # Scalar+[V <- V]


def tank_operators(
    closed: Topology,
    opened: Topology,
    vertices: Point,
    density: float,
    viscosity: float,
    salt_diffusivity: float,
) -> TankOperators:
    """Flow, advection, inertia, viscosity, heat-conduction and salt-diffusion operators assembled from the opened geometry onto the closed tank."""
    opened_corners = opened.incidence[2, 0] * vertices                        # [3] Point-[F]
    facing = opened.off[2, 0] * vertices                                      # [2, 3] Point-[F]
    sides = facing[0] & facing[1]                                             # [3] Flux-[F]
    # Twice each triangle's signed area in the projective plane, pairing past the four material directions.
    double_areas = opened_corners[0] & opened_corners[1] & opened_corners[2] & unread  # Antiscalar-[F]
    areas = double_areas.norm() / 2                                           # Scalar+[F]
    # Pure spatial simplex gradients: each plane vanishes along the facing side and rises to one at its corner.
    gradients = ((sides & unread) / (sides & unread & opened_corners)).cast(Upright)  # [3] Plane+[F]
    corners = closed.incidence[2, 0]                                          # [3] Scalar+[F <- V]
    dual_areas = (corners.adjugate() * (areas / 3)).sum(axis=0)               # Scalar+[V]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[F <- V]
    # Unit stream function at a corner drives flow along the oriented edge facing it, carrying half that edge displacement across the triangle.
    edges = opened.boundary[2, 1] * (opened.d[0] * vertices)                  # [3] Point-[F]
    flows = edges / 2                                                         # [3] Point-[F]
    velocities = flows / areas                                                # [3] Point-[F]
    # Triangle velocity from vertex stream values, and dual-cell outflow integrated against each corner's simplex gradient.
    stream_velocity = (spdiag(velocities * Scalar) * corners).sum(axis=0)     # (Point- <- Scalar+)[F <- V]
    cell_outflow = (corners.adjugate() * spdiag(areas * (gradients ^ Flux))).sum(axis=0)  # (Point- <- Flux-)[V <- F]
    # Kinetic energy form of the stream function, pulled back from triangle velocities to vertices.
    inertia = density * (readings * (readings & Point)).sum(axis=0)           # [] Plane+ <- Point-
    local = areas * (inertia(velocities[:, None]) & velocities[None, :])      # [3, 3] Scalar+[F]
    kinetic = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
    # Lumped vertex vorticity and its squared viscous dissipation form across the tank.
    vorticity = spdiag(1 / (density * dual_areas)) * kinetic                  # Scalar+[V <- V]
    viscous = (viscosity / density) * kinetic * vorticity                     # Scalar+[V <- V]
    # Thermal and haline drives on each corner's simplex gradient, mapped to entropy and salt fluxes along the tank directions.
    heat_drives, salt_drives = -(gradients ^ mv.s), -(gradients ^ mv.n)       # [3] Drive+[F], [3] Drive+[F]
    conductivity = ((directions & entropy) * ((directions & entropy) & Drive)).sum(axis=0)  # [] Flux- <- Drive+
    diffusivity = salt_diffusivity * ((directions & salt) * ((directions & salt) & Drive)).sum(axis=0)  # [] Flux- <- Drive+
    local = areas * (mv.s & (gradients[:, None] ^ conductivity(heat_drives[None, :])))  # [3, 3] Scalar+[F]
    conduction = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
    local = areas * (mv.n & (gradients[:, None] ^ diffusivity(salt_drives[None, :])))  # [3, 3] Scalar+[F]
    diffusion = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)  # Scalar+[V <- V]
    return TankOperators(corners, corner_mean, flows, stream_velocity, cell_outflow, dual_areas, kinetic, viscous, conduction, diffusion)


def fingering(
    operators: TankOperators,
    interior: Sparse,
    density: float,
    expansion: float,
    contraction: float,
    gravity: float,
    temperature: Scalar,
    salinity: Scalar,
    dt: float,
    n_steps: int,
    n_stride: int,
) -> Iterator[tuple[Scalar, Scalar, Scalar]]:
    """Yield the stream function, temperature and salinity fields every `n_stride` steps as salt fingers grow across the interface."""
    flow_step = wall_solver(operators.kinetic / dt + operators.viscous, interior)
    heat_step = wall_solver(operators.dual_areas / dt + operators.conduction, interior)
    salt_step = wall_solver(operators.dual_areas / dt + operators.diffusion, interior)
    wall_temperature = temperature - interior.adjugate() * (interior * temperature)  # [...] Scalar+[V]
    wall_salinity = salinity - interior.adjugate() * (interior * salinity)    # [...] Scalar+[V]
    pull = -(gravity * mv.ym)                                                 # [] Drive+
    stream = 0 * temperature                                                  # [...] Scalar+[V]
    for step in range(n_steps + 1):
        if step % n_stride == 0:
            yield stream, temperature, salinity
        if step == n_steps:
            break
        # Boussinesq parcel composition per unit volume: thermal expansion lowers mass while salt contraction raises it.
        warmth = operators.corner_mean * temperature                          # [...] Scalar+[F]
        saltiness = operators.corner_mean * salinity                          # [...] Scalar+[F]
        parcels = (
            volume
            + (density - expansion * warmth + contraction * saltiness) * mass
            + warmth * entropy
            + saltiness * salt
        )                                                                     # [...] Point-[F]
        # Buoyancy power of gravity's drive on each corner's parcel-carrying flow.
        load = (operators.corners.adjugate() * (pull & (operators.flows & parcels[..., None]))).sum(axis=-1)  # [...] Scalar+[V]
        stream = flow_step(operators.kinetic / dt * stream + load, 0 * stream)
        # Whole-parcel advection read out in entropy and salt, followed by implicit thermal conduction and salt diffusion.
        velocity = operators.stream_velocity(stream)                          # [...] Point-[F]
        flux = velocity & parcels                                             # [...] Flux-[F]
        outflow = operators.cell_outflow(flux)                                # [...] Point-[V]
        temperature = heat_step(operators.dual_areas / dt * temperature - (mv.s & outflow), wall_temperature)
        salinity = salt_step(operators.dual_areas / dt * salinity - (mv.n & outflow), wall_salinity)
