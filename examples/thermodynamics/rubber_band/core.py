"""A rubber band lifting a weight by heat, the Gough-Joule effect, on a strip of triangles that moves with the band, in
the plane joined with a direction for the entropy the band holds.

A rubber band's tension comes from its entropy: stretching it lines its chains up, so a stretched band holds less
entropy than a slack one at the same temperature, and it pulls back by its temperature times the entropy it would
regain. At a fixed stretch its tension is therefore proportional to its absolute temperature. A band holding up a
weight and warmed by a lamp pulls harder, shortens and lifts the weight; cooling, it lengthens and lets the weight down
again: a heat engine that turns some of the lamp's heat into the weight's potential energy. Stretched quickly, it warms;
let go, it cools. The material is the simplest that does this: elastic moduli linear in the displacement gradients and
proportional to the absolute temperature, and a heat capacity, so that the band's internal energy is its heat alone.

The algebra is the plane, x+y+, joined with one degenerate direction for entropy: x+y+s0. Each vertex is one point of
it, its place and the entropy it holds, `place + entropies * entropy`, both spread across each triangle by its corners'
hats. A corner's gradient is the plane along the facing side and the entropy over the triangle's size, `(entropy &
sides) / double_areas`. The rubber's stretch is an isotropic elastic work form on pairs of displacement gradients, as
for seismic waves, bound to each triangle's corner gradients. Each triangle's internal energy is one function of its
corners' points, of its stretch through their places and of its entropy through theirs; its change as a corner's point
moves is that corner's potentials, a plane whose spatial part is minus the force on the corner and whose part along
the entropy is the temperature. Stretched at fixed entropy, a triangle warms by its equation of state alone: the heat
of a pull is in the triangles that stretch, and nowhere else. Heat put in, taken out or conducted changes the vertices'
entropy by the heat over their temperature; conduction, as for convection, moves heat and makes none, and the entropy
it produces is never negative.

One step moves the vertices at fixed entropy by position Verlet, the forces taken at the middle of the move, which keeps
the energy without drift, and then changes the entropy by the heat the lamp, the room and the conduction move. The
operators are built on the band at rest, the stiffness and the conduction of the rubber's shape before it is loaded.

Lengths are in units of the band's width, temperatures in units of the room's absolute temperature, and forces in
units of the band's shear modulus at room temperature times its width; gravity's acceleration is one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.extensor import Extensor
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# The plane, and the entropy the band holds, a direction of its own with no metric.
ga = Algebra("x+y+") * Algebra("s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Force = ga.gatype.vector()                                                    # Vector+
Elastic = ga.gatype((Scalar, Plane, Plane, Point, Point))
# The planes reading a place's run across the band and up, and a temperature's drive, a slope wedged with the entropy.
Upright = ga.gatype.from_blades("x y")                                        # Vector+
Drive = ga.gatype.from_blades("xs ys")                                        # Bivector+
# The plane's two directions, across and up, and a unit of entropy.
across = mv.x.dual()                                                          # [] Point-
up = mv.y.dual()                                                              # [] Point-
directions = stack((across, up))                                              # [2] Point-
entropy = mv.s.dual()                                                         # [] Point-
# The rubber's conductivity at a unit rate, the same in every direction: each of the plane's directions swept through
# the entropy, read by a drive and driving the entropy along it.
sweeps = directions & entropy                                                 # [2] Flux-
spreading = (sweeps * (sweeps & Drive)).sum(axis=0)                          # [] Flux- <- Drive+
# The tractions of a plane isotropic material, as for seismic waves: the dilation's, and the shear's, which removes the
# rotation from the displacement gradient.
dilation_traction = Plane * (Plane & Point)                                   # [] Plane+ <- (Plane+, Plane+, Point-)
direct_shear = (Plane | Plane) * Point.dual_inverse()
shear_traction = 2 * direct_shear - (Plane | (Plane ^ Point.dual_inverse()))   # [] Plane+ <- (Plane+, Plane+, Point-)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, places: Point) -> tuple[Scalar, Plane]:
    """Each triangle's area `Scalar+[faces]` and each corner's gradient `[corners] Plane+[faces]`, the plane along the
    facing side and the entropy over twice the area, reading one from the side to the corner, for the triangles with
    their corners at these places `Point-[vertices]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * places)                # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, at a unit of entropy.
    double_areas = (sides[0] & sides[1]) & entropy                            # Scalar-[faces]
    gradients = (entropy & sides) / double_areas                              # [corners] Plane+[faces]
    return double_areas.norm() / 2, gradients


def constitutive(lame_lambda: float, lame_mu: float) -> Elastic:
    """A plane isotropic elastic work form on pairs of displacement gradients, of these Lamé moduli."""
    traction = lame_lambda * dilation_traction + lame_mu * shear_traction
    return traction & Point                                                   # [] Scalar+ <- (Plane+, Plane+, Point-, Point-)


class Band(NamedTuple):
    """The band and its room, built once on the band at rest: its triangles' corners and their mean, its places at
    rest, the reading that frees all but its held top row, each vertex's mass and cell, each triangle's area and elastic
    blocks at the room's temperature, the conduction, the rubber's heat capacity per area, the room's cooling,
    temperature, gravity and drag, and a hand's pull on the weight at its full strength."""
    corners: SparseExtensor                                                   # [corners] Scalar+[faces <- vertices]
    corner_mean: SparseExtensor                                               # Scalar+[faces <- vertices]
    rest: Point                                                               # Point-[vertices]
    freeing: SparseExtensor                                                   # Scalar+[vertices <- vertices]
    masses: Scalar                                                            # Scalar+[vertices]
    cells: Scalar                                                             # Scalar+[vertices]
    areas: Scalar                                                             # Scalar+[faces]
    elastic: Extensor                                                         # [corners, corners] (Force+ <- Point-)[faces]
    conduction: SparseExtensor                                                # Scalar+[vertices <- vertices]
    heat_capacity: float
    cooling: float
    room_temperature: float
    pull: Plane                                                               # [] Plane+
    drag: float
    hand: Plane                                                               # Force+[vertices]


def band(topology: Topology, rest: Point, held: np.ndarray, material: Elastic, stiffening: Scalar, densities: Scalar,
         weighting: Scalar, heat_capacity: float, conductivity: float, cooling: float, room_temperature: float,
         gravity: float, drag: float, hand_pull: float) -> Band:
    """The band of these triangles at rest at the places `rest` `Point-[vertices]`, its vertices `held` fixed, of this
    material at the room's temperature made stiffer by `stiffening` and of these densities `Scalar+[faces]`, the
    weight's triangles those `weighting` reads as one, of this heat capacity and conductivity per area, in a room
    cooling it by `cooling` per area and degree, at this temperature, with this gravity and the air's drag per mass; a
    hand pulls the weight down at full strength as hard as `hand_pull` times its weight."""
    areas, gradients = triangles(topology, rest)
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    corner_mean = corners.mean(axis=0)                                        # Scalar+[faces <- vertices]
    free = topology.reading(np.setdiff1d(np.arange(topology.n_vertices), held))
    # Each corner's hat owns a third of its triangle: its cell and its mass.
    cells = ~corner_mean * areas                                              # Scalar+[vertices]
    masses = ~corner_mean * (densities * areas)                               # Scalar+[vertices]
    pull = -gravity * mv.y                                                    # [] Plane+
    hand = hand_pull * (~corner_mean * (weighting * densities * areas)) * pull   # Force+[vertices]
    # The work form bound to each triangle's test and trial corner gradients: a form on the corners' displacements, as
    # a map to the forces on them, over the triangle's area.
    bound = gradients.batch()                                                 # [corners, faces] Plane+
    work = material(bound[:, None], bound[None, :], Point, Point)
    elastic = ((Force & Point).solve(work) * (stiffening * areas).batch()).field()   # [corners, corners] (Force+ <- Point-)[faces]
    # The conduction: each corner's hat's drive on the entropy, mapped to its flux and read by the other hats for its
    # entropy, over the triangle, gathered onto the vertices.
    drives = -(gradients ^ mv.s)                                              # [corners] Drive+[faces]
    local = areas * (mv.s & (gradients[:, None] ^ spreading(drives[None, :])))   # [corners, corners] Scalar+[faces]
    conduction = conductivity * (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)
    return Band(corners, corner_mean, rest, ~free * free, masses, cells, areas, elastic, conduction, heat_capacity,
                cooling, room_temperature, pull, drag, hand)


def potentials(band: Band, places: Point, entropies: Scalar) -> tuple[Scalar, Plane]:
    """Each triangle's temperature `Scalar+[faces]` and the forces on its corners `[corners] Force+[faces]`, the two
    parts of its internal energy's change as its corners' points move, with the vertices at these places
    `Point-[vertices]` holding these entropies `Scalar+[vertices]`. A triangle's free energy is its temperature over the
    room's times its stretch's elastic energy at the room's, and its heat capacity's; its entropy, spread from its
    corners by their hats, is its heat's less the stretch's elastic energy over the room's temperature, so that its
    temperature rises as it stretches at fixed entropy, and its forces are its elastic blocks scaled by it."""
    shifts = band.corners * (places - band.rest)                              # [corners] Point-[faces]
    # The forces of the stretch at the room's temperature, and its elastic energy, half the forces read on the shifts.
    springs = -band.elastic(shifts[None, :]).sum(axis=1)                      # [corners] Force+[faces]
    stretch = -(springs & shifts).sum(axis=0) / 2                             # Scalar+[faces]
    # The entropy per area each triangle holds, its corners' spread by their hats, and the temperature it and the
    # stretch give: the heat's entropy is the heat capacity times the logarithm of the temperature over the room's.
    held = band.corner_mean * (entropies / band.cells)                        # Scalar+[faces]
    warmth = ((held + stretch / (band.areas * band.room_temperature)) / band.heat_capacity).exp()   # Scalar+[faces]
    return band.room_temperature * warmth, warmth * springs


class Budget(NamedTuple):
    """The band's energy at a state, `[] Scalar+` each: the vertices' kinetic energy, their potential energy in
    gravity, the band's internal energy, its heat; and, over the steps that led to it, the heat the lamp and the room
    put in, the energy the air's drag dissipated, and the work the hand did on the weight."""
    kinetic: Scalar
    potential: Scalar
    thermal: Scalar
    heated: Scalar
    dissipated: Scalar
    handed: Scalar


def heating(band: Band, places: Point, entropies: Scalar, lamp: Point, lamp_heat: float, reach: float,
            exposure: np.ndarray, pulling: np.ndarray, dt: float, n_stride: int
            ) -> Iterator[tuple[Point, Scalar, Scalar, float, Budget]]:
    """The band's places `Point-[vertices]`, the entropies they hold and their temperatures `Scalar+[vertices]`,
    whether the lamp is on, and the budget, every `n_stride` steps of `dt`, one step for each of the lamp's exposures
    and the hand's pulls `[steps]`, each from none to full, from the band still at these places holding these
    entropies; the lamp at its place puts in this heat per area and time where it shines brightest, half as much this
    reach away."""
    corners, masses = band.corners, band.masses
    velocities = 0 * places                                                   # Point-[vertices]
    heated = dissipated = handed = 0 * entropies.sites.sum()                  # [] Scalar+
    # Each step's exposure and pull, and the last again, so that the state after the last step is yielded too.
    for step, (lit, pulled) in enumerate(zip(np.append(exposure, exposure[-1]), np.append(pulling, pulling[-1]))):
        if step % n_stride == 0:
            # Each triangle's temperature, and each vertex's, its triangles' around it by their areas, its cell's share.
            warmth, _ = potentials(band, places, entropies)                   # Scalar+[faces]
            temperatures = (~band.corner_mean * (band.areas * warmth)) / band.cells   # Scalar+[vertices]
            kinetic = (masses * (velocities.dual_inverse() & velocities)).sites.sum() / 2   # [] Scalar+
            potential = -(masses * (band.pull & places)).sites.sum()          # [] Scalar+
            thermal = band.heat_capacity * (band.areas * warmth).sites.sum()  # [] Scalar+
            yield places, entropies, temperatures, lit, Budget(kinetic, potential, thermal, heated, dissipated, handed)
        # The move at fixed entropy: the triangles' temperatures and forces at the middle of the move, the kick by them,
        # gravity and the hand against the drag of the mean of the velocities before and after it, and the step by that
        # mean.
        middle = places + dt / 2 * velocities                                 # Point-[vertices]
        warmth, pulls = potentials(band, middle, entropies)                   # Scalar+[faces], [corners] Force+[faces]
        forces = (~corners * pulls).sum(axis=0) + masses * band.pull + pulled * band.hand   # Force+[vertices]
        kicked = band.freeing * (((1 - band.drag * dt / 2) * velocities + dt * (forces / masses).dual())
                                 / (1 + band.drag * dt / 2))                  # Point-[vertices]
        steps = dt * (velocities + kicked) / 2                                # Point-[vertices]
        # The heat the lamp, the room and the conduction move into each vertex's cell over the step, at its temperature
        # at the middle of the move; the lamp's exposure falls off with the square of the distance from it.
        temperatures = (~band.corner_mean * (band.areas * warmth)) / band.cells   # Scalar+[vertices]
        offsets = middle - lamp                                               # Point-[vertices]
        exposures = 1 / (1 + (offsets.dual_inverse() & offsets) / reach ** 2)   # Scalar+[vertices]
        exchanged = dt * band.cells * (lit * lamp_heat * exposures
                                       - band.cooling * (temperatures - band.room_temperature))   # Scalar+[vertices]
        conducted = -dt * (band.conduction * temperatures)                    # Scalar+[vertices]
        # The heat warms a cell by itself over the cell's heat capacity, and its entropy grows by the heat capacity
        # times the logarithm of the warmer temperature over the cooler.
        capacities = band.heat_capacity * band.cells                          # Scalar+[vertices]
        entropies = entropies + capacities * (1 + (exchanged + conducted) / (capacities * temperatures)).log()
        heated = heated + exchanged.sites.sum()
        dissipated = dissipated + band.drag * (masses * (steps.dual_inverse() & steps)).sites.sum() / dt
        handed = handed + pulled * (band.hand & steps).sites.sum()
        places, velocities = places + steps, kicked
