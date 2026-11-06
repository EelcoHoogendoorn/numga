"""Superfluid helium's two fluids, on a mesh of triangles, in the plane joined with two directions, for the mass and the
entropy a parcel of liquid holds.

Below the lambda point liquid helium flows as two fluids through each other: a superfluid, which carries mass but no
entropy and has no viscosity, and a normal fluid, which carries the rest of the mass and all of the entropy. Heat
cannot spread through it by diffusing: a warm place pushes its normal fluid out and draws superfluid in, and the
counterflow runs on as a wave of temperature, second sound. Where a fine powder holds the normal fluid still, only
the superfluid moves, and the same heat drives it through the powder as a wave of pressure and temperature together,
fourth sound, ten times faster. The normal fluid's viscosity, its conduction of heat and the mutual friction between the
two fluids are left out, so nothing damps the motion.

The algebra is the plane, x+y+, joined with two degenerate directions, one for mass and one for entropy: x+y+m0s0. A
parcel is a point weighed by both, `mass * mv.m.dual() + entropy * mv.s.dual()`, and a superfluid parcel's entropy is
zero. The liquid's potentials are one vector, the chemical potential per mass along m and the temperature along s,
`chemical_potential * mv.m + temperature * mv.s`. Their drive is minus their gradient wedged with them, a field of
bivectors each pairing a direction of the plane with one of the two degenerate directions; as a translation's
generator moves a point by its weight alone, a drive's commutator with a parcel pushes it by what it holds:
`drive.commutator(parcel)` is the force on it, its mass times the fall of the chemical potential and its entropy times
the fall of the temperature, the two-fluid model's forces on its two components. A parcel moving at a velocity
carries what it holds along the line it moves on, the flux `velocity.regressive(parcel)`, and a hat gradient wedged
with a flux, `gradient ^ flux`, is what the flux carries out of the corner's cell. The thermodynamic stiffness, the map
from the amounts a cell holds to its potentials, is a sum of two dyads, compression and heating, symmetric as
Maxwell's relations require.

Lengths are in millimetres and times in milliseconds, so that speeds read in metres per second; amounts are per unit
volume, their changes small against the liquid's own.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from examples.util.topology import Topology

# The plane, and the mass and the entropy a parcel holds, each a direction of its own with no metric.
space = Algebra("x+y+")
amounts = Algebra("m0s0")
ga = space * amounts
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# What a parcel holds, the points that span the plane; the potentials that read it.
Amount = Plane.derive.within(amounts).derive.dual                             # Antivector-
Potential = Plane.derive.within(amounts)                                      # Vector+
Drive = ga.gatype.bivector().derive.touching(space, amounts)                  # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
# A unit of mass, the weight of a place, and a unit of entropy.
mass = mv.m.dual()                                                            # [] Amount-
entropy = mv.s.dual()                                                         # [] Amount-


# --- plumbing -------------------------------------------------------------------------
def cavity(n_columns: int, n_rows: int, width: float, height: float) -> tuple[Topology, np.ndarray]:
    """A rectangle of triangles `width` across and `height` up, its vertices in rows: its topology, and its vertices'
    places across and up `[vertices, 2]`."""
    places = np.meshgrid(np.linspace(0, width, n_columns), np.linspace(0, height, n_rows))
    return Topology.grid(n_rows, n_columns), np.stack([place.ravel() for place in places], axis=-1)


# --- math -----------------------------------------------------------------------------
def stiffness_of(density: np.ndarray, first_sound: np.ndarray, temperature: np.ndarray, entropy_per_mass: np.ndarray,
                 heat_capacity: np.ndarray) -> Extensor:
    """The liquid's thermodynamic stiffness `[...] Potential+ <- Amount-`, the changes of its potentials under changes
    of the mass and entropy per volume it holds: compression raises the chemical potential by the squared first sound
    over the density per unit of mass, and heating, a change of the entropy per mass, raises the temperature by the
    temperature over the heat capacity and lowers the chemical potential by the entropy per mass times that rise."""
    # A unit rise of the entropy per mass, read as the change of entropy beyond what the mass carries in.
    heating = mv.s - entropy_per_mass * mv.m                                  # [...] Potential+
    return (first_sound ** 2 / density) * (mv.m * (mv.m & Amount)) + (
        temperature / (heat_capacity * density)) * (heating * (heating & Amount))


def sound_speeds(stiffness: Extensor, parcels: Point, masses: np.ndarray) -> Scalar:
    """The squared speeds of sound `[..., 2] Scalar+`, slowest first, of the liquid with this stiffness `[...]
    Potential+ <- Amount-` whose components move as these parcels per volume `[..., components] Amount-` of these
    masses `[..., components]`: the pencil of the components' kinetic form on potentials, each parcel read by the
    potentials and weighed by its inverse mass, against the liquid's compliance."""
    yielding = (parcels * (Potential & parcels) / masses).sum(axis=-1)       # [...] Amount- <- Potential+
    compliance = stiffness.inverse()                                          # [...] Amount- <- Potential+
    speeds_squared, _ = (Potential & yielding(Potential)).eigh(Potential & compliance(Potential))
    return speeds_squared


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Each triangle's area `Scalar+[faces]`, each corner's hat gradient `[corners] Plane+[faces]`, the plane through the
    facing side at unit mass and entropy over twice the area, reading one from the side to the corner, and each
    vertex's share of the area around it `Scalar+[vertices]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)[..., None]  # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read at unit entropy and mass.
    double_areas = ((sides[..., 0] & sides[..., 1]) & entropy) & mass         # Scalar-[faces]
    areas = double_areas.norm() / 2                                           # Scalar+[faces]
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    return areas, (entropy & mass & sides) / double_areas[..., None], (~corners * (areas / 3)).sum(axis=-1)


def stepped(topology: Topology, vertices: Point, stiffness: Extensor, parcels: Point, masses: np.ndarray, held: Scalar,
            amounts: Point, supply: Point, dt: float, n_steps: int, stride: int) -> Iterator[Point]:
    """The mass and entropy per volume over the cavity `Point-[vertices]`, from these at rest, every `stride` steps of
    `dt`, with this supply of them per volume and time `Point-[vertices]`, its components moving as these parcels
    `[components] Amount-` of these masses `[components]` with the normal fluid held still where `held` `Scalar+[faces]`
    is one: each step the potentials' drive pushes each component's parcels, and the flux they carry empties the cells
    it runs out of into those it runs into."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    areas, gradients, cells = triangles(topology, vertices)
    # How freely each component moves in each triangle: the superfluid everywhere, the normal fluid but where held.
    freedom = (1 - held[..., None] * np.array([0.0, 1.0])) / masses           # [components] Scalar+[faces]
    velocities = 0 * (axis * freedom)                                         # [components] Point-[faces]
    for step in range(n_steps + 1):
        if step % stride == 0:
            yield amounts
        potentials = stiffness(amounts)                                       # Potential+[vertices]
        drive = -(gradients ^ (corners * potentials[..., None])).sum(axis=-1)   # Drive+[faces]
        velocities = velocities + dt * freedom * drive.commutator(parcels)    # [components] Point-[faces]
        flux = velocities.regressive(parcels).sum(axis=-1)                    # Flux-[faces]
        # What the flux carries out of each corner's cell, and what the supply adds to it.
        amounts = amounts + dt * (supply - (~corners * (areas * (gradients ^ flux[..., None]))).sum(axis=-1) / cells)
