"""A rubber band hung with a weight lifts the weight when warmed by a lamp, the Gough-Joule effect, on a triangulated
strip in the plane joined with a direction for entropy.

A rubber band's tension comes from its entropy: stretching the band lines up its polymer chains, so a stretched band has
less entropy than a slack one at the same temperature, and it pulls back in proportion to its absolute temperature. When
a lamp warms a band that holds up a weight, the band pulls harder, shortens and lifts the weight; as it cools, it
lengthens and lets the weight down again, acting as a heat engine that turns some of the lamp's heat into gravitational
potential energy. Stretched quickly, the band warms; let go, it cools. The model uses elastic moduli proportional to the
absolute temperature and a constant heat capacity, so the band's internal energy depends on its temperature alone.

The algebra joins the plane `x+ y+` with one degenerate quantity direction `s0` for entropy: `x+ y+ s0`. Each vertex is
modelled as one point of that algebra, combining its spatial offset with its entropy, `places + entropies * entropy`. On
each triangle of the unloaded band, a corner's simplex gradient joins the facing side with entropy and divides by twice
the triangle's signed area. Evaluating an isotropic elastic work form on each triangle's corner gradients gives its
room-temperature stiffness. Each triangle's energy is a function of its corners' points, through their spatial shifts
and their entropies, and its derivative at a vertex is a single plane of potentials, `temperatures * s - forces`,
combining minus the elastic force in the plane with the temperature along the entropy direction. Stretched at fixed
entropy, a triangle warms directly from its equation of state, while pulling the rubber's conductivity back through the
simplex gradients gives a symmetric, positive conduction map that moves heat without creating or destroying energy. Each
step splits into a reversible mechanical move at fixed entropy and an irreversible thermal exchange with the lamp, the
room and neighbouring vertices.

Lengths are in units of the band's width, temperatures in units of the room's absolute temperature, and forces in units
of the band's room-temperature shear modulus times its width; gravitational acceleration is one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.extensor import Extensor
from numga.sparse import Sparse, SparseExtensor, spdiag

from numga.topology import Topology

# The plane `x+ y+` joined with one degenerate quantity direction `s0` for entropy.
space = Algebra("x+y+")
quantities = Algebra("s0")
ga = space * quantities
mv = NumpyContext(ga, units=True).multivector

Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# Spatial planes reading a place across and up the band, their dual spatial offsets, and thermal drives.
Upright = Plane.derive.within(space)                                          # Vector+
Spatial = Upright.derive.dual                                                 # Antivector-
Elastic = ga.gatype((Scalar, Upright, Upright, Spatial, Spatial))
Drive = ga.gatype.bivector().derive.touching(space, quantities)               # Bivector+
# Unit spatial directions across and up the plane, and a unit of entropy.
across = mv.x.dual()                                                          # [] Spatial-
up = mv.y.dual()                                                              # [] Spatial-
directions = stack((across, up))                                              # [2] Spatial-
entropy = mv.s.dual()                                                         # [] Point-
# Isotropic unit thermal conductivity: each spatial direction joined with entropy pairs with a drive to carry entropy.
sweeps = directions & entropy                                                 # [2] Antibivector-
spreading = (sweeps * (sweeps & Drive)).sum(axis=0)                           # [] Antibivector- <- Bivector+
# Isotropic plane-elastic tractions: dilation and rotation-free shear on displacement gradients.
dilation_traction = Upright * (Upright & Spatial)                             # [] Upright+ <- (Upright+, Upright+, Spatial-)
direct_shear = (Upright | Upright) * Spatial.dual_inverse()                   # [] Upright+ <- (Upright+, Upright+, Spatial-)
shear_traction = 2 * direct_shear - (Upright | (Upright ^ Spatial.dual_inverse()))  # [] Upright+ <- (Upright+, Upright+, Spatial-)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, places: Spatial) -> tuple[Scalar, Upright]:
    """Triangle areas (`Scalar+[F], length^2`) and corner simplex gradients (`[3] Upright+[F], 1/length`) for the
    unloaded vertices; each corner gradient reads one from the facing side to its corner and zero along entropy."""
    sides = topology.boundary[2, 1] * (topology.d[0] * places)                # [3] Spatial-[F], length
    # Joining two sides with a unit of entropy gives twice the triangle's signed area.
    double_areas = (sides[0] & sides[1]) & entropy                            # Scalar-[F], length^2
    gradients = (entropy & sides) / double_areas                              # [3] Upright+[F], 1/length
    return double_areas.norm() / 2, gradients


def constitutive(lame_lambda: Scalar, lame_mu: Scalar) -> Elastic:
    """Isotropic plane-elastic work form on pairs of displacement gradients for the given Lamé moduli."""
    traction = lame_lambda * dilation_traction + lame_mu * shear_traction     # [] Upright+ <- (Upright+, Upright+, Spatial-)
    return traction & Spatial                                                 # [] Elastic


class Band(NamedTuple):
    """Geometry, operators and parameters of the unloaded band at rest and its surrounding room."""
    corners: Sparse                                                           # [3] Scalar+[F <- V]
    corner_mean: Sparse                                                       # Scalar+[F <- V]
    rest: Spatial                                                             # Spatial-[V], length
    freeing: SparseExtensor                                                   # Scalar+[V <- V]
    masses: Scalar                                                            # Scalar+[V], extent*mass
    dual_areas: Scalar                                                        # Scalar+[V], length^2
    areas: Scalar                                                             # Scalar+[F], length^2
    elastic: Extensor                                                         # [3, 3] (Upright+ <- Spatial-)[F]
    conduction: SparseExtensor                                                # Scalar+[V <- V], extent*length^2*mass/(temperature*time^3)
    heat_capacity: Scalar                                                     # [] Scalar+, extent*mass/(temperature*time^2)
    cooling: Scalar                                                           # [] Scalar+, extent*mass/(temperature*time^3)
    room_temperature: Scalar                                                  # [] Scalar+, temperature
    pull: Upright                                                             # [] Upright+, length/time^2
    drag: Scalar                                                              # [] Scalar+, 1/time
    hand: Upright                                                             # Upright+[V], extent*length*mass/time^2


def band(topology: Topology, rest: Spatial, held: np.ndarray, material: Elastic, stiffening: Scalar, densities: Scalar,
         weighting: Scalar, heat_capacity: Scalar, conductivity: Scalar, cooling: Scalar,
         room_temperature: Scalar, gravity: Scalar, drag: Scalar, hand_pull: float) -> Band:
    """Assembled stiffness, conduction and vertex properties of the hanging band and its weight on the unloaded rest
    triangulation, with the top row clamped."""
    V, E, F = topology.n_elements
    areas, gradients = triangles(topology, rest)
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[F <- V]
    free = topology.reading(np.setdiff1d(np.arange(V), held), 0)              # Scalar+[V_free <- V]
    # Each vertex cell receives one third of the area and mass of its incident triangles.
    dual_areas = corner_mean.adjugate() * areas                               # Scalar+[V], length^2
    masses = corner_mean.adjugate() * (densities * areas)                     # Scalar+[V], extent*mass
    pull = -gravity * mv.y                                                    # [] Upright+, length/time^2
    hand = hand_pull * (corner_mean.adjugate() * (weighting * densities * areas)) * pull   # Upright+[V], extent*length*mass/time^2
    # Evaluating the work form on each triangle's corner gradients gives its room-temperature stiffness map.
    bound = gradients.batch()                                                 # [3, F] Upright+, 1/length
    work = material(bound[:, None], bound[None, :], Spatial, Spatial)         # [3, 3, F] Scalar+ <- (Spatial-, Spatial-)
    elastic = ((Upright & Spatial).solve(work) * (stiffening * areas).batch()).field()   # [3, 3] (Upright+ <- Spatial-)[F]
    # Pulling the unit conductivity back through the corner gradients assembles the vertex conduction map.
    drives = -gradients ^ mv.s                                                # [3] Bivector+[F], 1/length
    local = areas * (mv.s & (gradients[:, None] ^ spreading(drives[None, :])))   # [3, 3] Scalar+[F]
    conduction = conductivity * (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)
    return Band(corners, corner_mean, rest, free.adjugate() * free, masses, dual_areas, areas, elastic, conduction,
                heat_capacity, cooling, room_temperature, pull, drag, hand)


def potentials(band: Band, points: Point) -> Plane:
    """Potentials conjugate to the vertex state, combining minus the elastic force in the plane with the absolute
    temperature along the entropy direction. Because a triangle's entropy equals its thermal entropy minus its
    room-temperature elastic energy divided by room temperature, stretching at fixed entropy raises its temperature and
    scales its restoring forces in proportion."""
    # Separate each vertex's entropy from its spatial shift relative to rest.
    held_entropy = mv.s & points                                              # Scalar+[V], extent*length^2*mass/(temperature*time^2)
    shifts = band.corners * (points.cast(Spatial) - band.rest)                # [3] Spatial-[F], length
    # Room-temperature elastic forces on the corners and the corresponding elastic energy on each triangle.
    springs = -band.elastic(shifts[None, :]).sum(axis=1)                      # [3] Upright+[F], extent*length*mass/time^2
    stretch = -(springs & shifts).sum(axis=0) / 2                             # Scalar+[F], extent*length^2*mass/time^2
    # Mean vertex entropy density and stretch determine each triangle's temperature ratio relative to the room.
    entropy_densities = band.corner_mean * (held_entropy / band.dual_areas)   # Scalar+[F], extent*mass/(temperature*time^2)
    warmth = ((entropy_densities + stretch / (band.areas * band.room_temperature)) / band.heat_capacity).exp()   # Scalar+[F]
    # Area-average the triangle temperatures onto the vertex cells and scale the corner forces by `warmth`.
    temperatures = band.room_temperature * (band.corner_mean.adjugate() * (band.areas * warmth)) / band.dual_areas   # Scalar+[V], temperature
    forces = (band.corners.adjugate() * (warmth * springs)).sum(axis=0)       # Upright+[V], extent*length*mass/time^2
    return temperatures * mv.s - forces


def move(band: Band, points: Point, velocities: Spatial, forces: Upright, pulled: float,
         dt: Scalar) -> tuple[Point, Spatial]:
    """Vertex state and velocities after a mechanical step at fixed entropy under midpoint elastic forces, gravity, air
    drag and the downward hand pull, with the clamped top row held still."""
    # Implicit midpoint air drag updates the free velocities before advancing the positions by the mean velocity.
    loads = forces + band.masses * band.pull + pulled * band.hand             # Upright+[V], extent*length*mass/time^2
    kicked = band.freeing * (((1 - band.drag * dt / 2) * velocities + dt * (loads / band.masses).dual())
                             / (1 + band.drag * dt / 2))                      # Spatial-[V], length/time
    return points + dt * (velocities + kicked) / 2, kicked


def exchange(band: Band, points: Point, temperatures: Scalar, lamp: Spatial, lamp_heat: Scalar, reach: Scalar,
             lit: float, dt: Scalar) -> Scalar:
    """Entropy gained by each vertex cell over one step from lamp heating, Newton cooling to the room, and heat
    conduction across the band."""
    # Lamp intensity falls off with squared spatial distance after projecting out the entropy component from `points`.
    offsets = points.cast(Spatial) - lamp                                     # Spatial-[V], length
    exposures = 1 / (1 + (offsets.dual_inverse() & offsets) / reach ** 2)     # Scalar+[V]
    exchanged = dt * band.dual_areas * (lit * lamp_heat * exposures
                                        - band.cooling * (temperatures - band.room_temperature))   # Scalar+[V], extent*length^2*mass/time^2
    conducted = -dt * (band.conduction * temperatures)                        # Scalar+[V], extent*length^2*mass/time^2
    # Adding heat to a cell of constant heat capacity raises its entropy logarithmically with the temperature ratio.
    capacities = band.heat_capacity * band.dual_areas                         # Scalar+[V], extent*length^2*mass/(temperature*time^2)
    return capacities * (1 + (exchanged + conducted) / (capacities * temperatures)).log()


def step(band: Band, points: Point, velocities: Spatial, lamp: Spatial, lamp_heat: Scalar, reach: Scalar, lit: float,
         pulled: float, dt: Scalar) -> tuple[Point, Spatial]:
    """Vertex state and velocities after one step, combining a reversible mechanical step at fixed entropy with an
    irreversible entropy shift from the lamp, the room and heat conduction."""
    # Evaluate the potentials at the midpoint to split out the temperatures and the elastic forces.
    middle = points + dt / 2 * velocities                                     # Point-[V], length | entropy
    middle_potentials = potentials(band, middle)                              # Plane+[V], force | temperature
    temperatures = middle_potentials & entropy                                # Scalar+[V], temperature
    forces = (-middle_potentials).cast(Upright)                               # Upright+[V], extent*length*mass/time^2
    points, velocities = move(band, points, velocities, forces, pulled, dt)
    gained = exchange(band, middle, temperatures, lamp, lamp_heat, reach, lit, dt)   # Scalar+[V], extent*length^2*mass/(temperature*time^2)
    return points + gained * entropy, velocities


def run(band: Band, points: Point, lamp: Spatial, lamp_heat: Scalar, reach: Scalar, exposure: np.ndarray,
        pulling: np.ndarray, dt: Scalar, n_stride: int) -> Iterator[tuple[Point, Scalar, float]]:
    """Yield the band's vertex state, vertex temperatures and lamp state at the start and after each frame stride."""
    velocities = mv.zeros(Spatial, points.sites.count).field()                # Spatial-[V], 0
    yield points, potentials(band, points) & entropy, exposure[0]
    for exposures, pulls in zip(exposure.reshape(-1, n_stride), pulling.reshape(-1, n_stride)):
        for lit, pulled in zip(exposures, pulls):
            points, velocities = step(band, points, velocities, lamp, lamp_heat, reach, lit, pulled, dt)
        yield points, potentials(band, points) & entropy, exposures[-1]
