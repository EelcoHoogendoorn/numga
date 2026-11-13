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
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# The plane `x+ y+` joined with one degenerate quantity direction `s0` for entropy.
space = Algebra("x+y+")
quantities = Algebra("s0")
ga = space * quantities
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Elastic = ga.gatype((Scalar, Plane, Plane, Point, Point))
# Spatial planes reading a place across and up the band, and thermal drives wedging a spatial slope with entropy.
Upright = Plane.derive.within(space)                                          # Vector+
Drive = ga.gatype.bivector().derive.touching(space, quantities)               # Bivector+
# Unit spatial directions across and up the plane, and a unit of entropy.
across = mv.x.dual()                                                          # [] Point-
up = mv.y.dual()                                                              # [] Point-
directions = stack((across, up))                                              # [directions] Point-
entropy = mv.s.dual()                                                         # [] Point-
# Isotropic unit thermal conductivity: each spatial direction joined with entropy pairs with a drive to carry entropy.
sweeps = directions & entropy                                                 # [directions] Antibivector-
spreading = (sweeps * (sweeps & Drive)).sum(axis=0)                           # [] Antibivector- <- Bivector+
# Isotropic plane-elastic tractions: dilation and rotation-free shear on displacement gradients.
dilation_traction = Plane * (Plane & Point)                                   # [] Plane+ <- (Plane+, Plane+, Point-)
direct_shear = (Plane | Plane) * Point.dual_inverse()                         # [] Plane+ <- (Plane+, Plane+, Point-)
shear_traction = 2 * direct_shear - (Plane | (Plane ^ Point.dual_inverse()))  # [] Plane+ <- (Plane+, Plane+, Point-)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, places: Point) -> tuple[Scalar, Plane]:
    """Triangle areas and corner simplex gradients for the unloaded vertices; each corner gradient reads one from the
    facing side to its corner and zero along entropy."""
    sides = topology.boundary[2, 1] * (topology.d[0] * places)                # [corners] Point-[faces]
    # Joining two sides with a unit of entropy gives twice the triangle's signed area.
    double_areas = (sides[0] & sides[1]) & entropy                            # Scalar-[faces]
    gradients = (entropy & sides) / double_areas                              # [corners] Plane+[faces]
    return double_areas.norm() / 2, gradients


def constitutive(lame_lambda: float, lame_mu: float) -> Elastic:
    """Isotropic plane-elastic work form on pairs of displacement gradients for the given Lamé moduli."""
    traction = lame_lambda * dilation_traction + lame_mu * shear_traction     # [] Plane+ <- (Plane+, Plane+, Point-)
    return traction & Point                                                   # [] Scalar+ <- (Plane+, Plane+, Point-, Point-)


class Band(NamedTuple):
    """Geometry, operators and parameters of the unloaded band at rest and its surrounding room."""
    corners: SparseExtensor                                                   # [corners] Scalar+[faces <- vertices]
    corner_mean: SparseExtensor                                               # Scalar+[faces <- vertices]
    rest: Point                                                               # Point-[vertices]
    freeing: SparseExtensor                                                   # Scalar+[vertices <- vertices]
    masses: Scalar                                                            # Scalar+[vertices]
    cells: Scalar                                                             # Scalar+[vertices]
    areas: Scalar                                                             # Scalar+[faces]
    elastic: Extensor                                                         # [corners, corners] (Plane+ <- Point-)[faces]
    conduction: SparseExtensor                                                # Scalar+[vertices <- vertices]
    heat_capacity: float
    cooling: float
    room_temperature: float
    pull: Plane                                                               # [] Plane+
    drag: float
    hand: Plane                                                               # Plane+[vertices]


def band(topology: Topology, rest: Point, held: np.ndarray, material: Elastic, stiffening: Scalar, densities: Scalar,
         weighting: Scalar, heat_capacity: float, conductivity: float, cooling: float, room_temperature: float,
         gravity: float, drag: float, hand_pull: float) -> Band:
    """Assembled stiffness, conduction and vertex properties of the hanging band and its weight on the unloaded rest
    triangulation, with the top row clamped."""
    areas, gradients = triangles(topology, rest)
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[faces <- vertices]
    free = topology.reading(np.setdiff1d(np.arange(topology.n_vertices), held), 0)
    # Each vertex cell receives one third of the area and mass of its incident triangles.
    cells = ~corner_mean * areas                                              # Scalar+[vertices]
    masses = ~corner_mean * (densities * areas)                               # Scalar+[vertices]
    pull = -gravity * mv.y                                                    # [] Plane+
    hand = hand_pull * (~corner_mean * (weighting * densities * areas)) * pull   # Plane+[vertices]
    # Evaluating the work form on each triangle's corner gradients gives its room-temperature stiffness map.
    bound = gradients.batch()                                                 # [corners, faces] Plane+
    work = material(bound[:, None], bound[None, :], Point, Point)             # [corners, corners, faces] Scalar+ <- (Point-, Point-)
    elastic = ((Plane & Point).solve(work) * (stiffening * areas).batch()).field()   # [corners, corners] (Plane+ <- Point-)[faces]
    # Pulling the unit conductivity back through the corner gradients assembles the vertex conduction map.
    drives = -gradients ^ mv.s                                                # [corners] Bivector+[faces]
    local = areas * (mv.s & (gradients[:, None] ^ spreading(drives[None, :])))   # [corners, corners] Scalar+[faces]
    conduction = conductivity * (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar+[vertices <- vertices]
    return Band(corners, corner_mean, rest, ~free * free, masses, cells, areas, elastic, conduction, heat_capacity,
                cooling, room_temperature, pull, drag, hand)


def potentials(band: Band, points: Point) -> Plane:
    """Potentials conjugate to the vertex state, combining minus the elastic force in the plane with the absolute
    temperature along the entropy direction. Because a triangle's entropy equals its thermal entropy minus its
    room-temperature elastic energy divided by room temperature, stretching at fixed entropy raises its temperature and
    scales its restoring forces in proportion."""
    # Separate each vertex's entropy from its spatial shift relative to rest.
    held_entropy = mv.s & points                                              # Scalar+[vertices]
    shifts = band.corners * ((points - held_entropy * entropy) - band.rest)   # [corners] Point-[faces]
    # Room-temperature elastic forces on the corners and the corresponding elastic energy on each triangle.
    springs = -band.elastic(shifts[None, :]).sum(axis=1)                      # [corners] Plane+[faces]
    stretch = -(springs & shifts).sum(axis=0) / 2                             # Scalar+[faces]
    # Mean vertex entropy density and stretch determine each triangle's temperature ratio relative to the room.
    entropy_densities = band.corner_mean * (held_entropy / band.cells)        # Scalar+[faces]
    warmth = ((entropy_densities + stretch / (band.areas * band.room_temperature)) / band.heat_capacity).exp()   # Scalar+[faces]
    # Area-average the triangle temperatures onto the vertex cells and scale the corner forces by `warmth`.
    temperatures = band.room_temperature * (~band.corner_mean * (band.areas * warmth)) / band.cells   # Scalar+[vertices]
    forces = (~band.corners * (warmth * springs)).sum(axis=0)                 # Plane+[vertices]
    return temperatures * mv.s - forces


def move(band: Band, points: Point, velocities: Point, forces: Plane, pulled: float, dt: float) -> tuple[Point, Point]:
    """Vertex state and velocities after a mechanical step at fixed entropy under midpoint elastic forces, gravity, air
    drag and the downward hand pull, with the clamped top row held still."""
    # Implicit midpoint air drag updates the free velocities before advancing the positions by the mean velocity.
    loads = forces + band.masses * band.pull + pulled * band.hand             # Plane+[vertices]
    kicked = band.freeing * (((1 - band.drag * dt / 2) * velocities + dt * (loads / band.masses).dual())
                             / (1 + band.drag * dt / 2))                      # Point-[vertices]
    return points + dt * (velocities + kicked) / 2, kicked


def exchange(band: Band, points: Point, temperatures: Scalar, lamp: Point, lamp_heat: float, reach: float, lit: float,
             dt: float) -> Scalar:
    """Entropy gained by each vertex cell over one step from lamp heating, Newton cooling to the room, and heat
    conduction across the band."""
    # Lamp intensity falls off with squared spatial distance after stripping the entropy component from `points`.
    offsets = points - (mv.s & points) * entropy - lamp                       # Point-[vertices]
    exposures = 1 / (1 + (offsets.dual_inverse() & offsets) / reach ** 2)     # Scalar+[vertices]
    exchanged = dt * band.cells * (lit * lamp_heat * exposures
                                   - band.cooling * (temperatures - band.room_temperature))   # Scalar+[vertices]
    conducted = -dt * (band.conduction * temperatures)                        # Scalar+[vertices]
    # Adding heat to a cell of constant heat capacity raises its entropy logarithmically with the temperature ratio.
    capacities = band.heat_capacity * band.cells                              # Scalar+[vertices]
    return capacities * (1 + (exchanged + conducted) / (capacities * temperatures)).log()


def step(band: Band, points: Point, velocities: Point, lamp: Point, lamp_heat: float, reach: float, lit: float,
         pulled: float, dt: float) -> tuple[Point, Point]:
    """Vertex state and velocities after one step, combining a reversible mechanical step at fixed entropy with an
    irreversible entropy shift from the lamp, the room and heat conduction."""
    # Evaluate the potentials at the midpoint to split out the temperatures and the elastic forces.
    middle = points + dt / 2 * velocities                                     # Point-[vertices]
    middle_potentials = potentials(band, middle)                              # Plane+[vertices]
    temperatures = middle_potentials & entropy                                # Scalar+[vertices]
    forces = temperatures * mv.s - middle_potentials                          # Plane+[vertices]
    points, velocities = move(band, points, velocities, forces, pulled, dt)
    gained = exchange(band, middle, temperatures, lamp, lamp_heat, reach, lit, dt)   # Scalar+[vertices]
    return points + gained * entropy, velocities


def run(band: Band, points: Point, lamp: Point, lamp_heat: float, reach: float, exposure: np.ndarray,
        pulling: np.ndarray, dt: float, n_stride: int) -> Iterator[tuple[Point, Scalar, float]]:
    """Yield the band's vertex state, vertex temperatures and lamp state at the start and after each frame stride."""
    velocities = 0 * points                                                   # Point-[vertices]
    yield points, potentials(band, points) & entropy, exposure[0]
    for exposures, pulls in zip(exposure.reshape(-1, n_stride), pulling.reshape(-1, n_stride)):
        for lit, pulled in zip(exposures, pulls):
            points, velocities = step(band, points, velocities, lamp, lamp_heat, reach, lit, pulled, dt)
        yield points, potentials(band, points) & entropy, exposures[-1]
