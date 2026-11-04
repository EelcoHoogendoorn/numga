"""Galvanomagnetic and thermomagnetic effects: charge and heat carried through a conductor in a magnetic field by two
kinds of carrier, electrons and holes, as the balance between them runs from electrons only to holes only; on meshes of
triangles, in planar spacetime joined with a direction for entropy.

Each carrier holds a charge and an entropy, so a fall of the electrochemical potential pushes it by its charge and a
fall of temperature by its entropy, and it drifts where the drag of its collisions balances the push, Drude's steady
state; a current carries each kind's entropy per charge along with it, the Peltier effect, and the same entropy per
charge is the Seebeck voltage's per fall of temperature, Kelvin's relation. A magnetic field across the conductor turns
each carrier's drift sideways as it moves, by an angle that grows with the field, the Hall angle, and the opposite
charges of electrons and holes turn their drifts opposite ways. A current along a bar with insulating sides piles charge
onto one side until the voltage that builds up across the bar, the Hall voltage, holds the carriers on course; with a
single kind of carrier it undoes the turning entirely, so that the bar's resistance along it does not change. Where
electrons and holes are balanced, as in a semimetal such as bismuth, their turned currents cancel, no Hall voltage
builds up to hold the carriers on course, and the resistance rises with the field; both kinds are turned to the same
side and carry their entropy there, which warms it, the Ettingshausen effect. A fall of temperature alone drives both
kinds the same way, and the field parts them to opposite sides, where their charges pile up into a transverse voltage,
the Nernst effect, strongest where the Hall voltage vanishes; a single kind of carrier stands still in it, held by its
own Seebeck voltage, and where the kinds are unbalanced the heat they carry is turned across the bar too, the
Righi-Leduc effect. A Corbino ring, its contacts on its rims, has no sides for anything to pile up on: its carriers
spiral, and a fall of temperature between its rims drives a current around it.

The algebra is planar spacetime, x+y+t-, joined with one degenerate direction for entropy: x+y+t-s0. A unit of charge
is a direction along time, `-mv.t.dual()`, and a unit of entropy `mv.s.dual()`, which is also the weight of a place; a
carrier's parcel holds both, `valence * (time + entropy_per_charge * entropy)`. The potentials are one vector, the
electrochemical potential along t and the temperature along s, and their drive is minus their gradient wedged with
them. The magnetic field is a plane of space, `strength * mv.xy`, and with the drive it pushes a moving carrier by one
commutator, `(magnetic + drive).commutator(parcel + valence * velocity)`: through space the drive's push on the parcel
at rest and the field's turning of the drift, with no cross product and no magnetic term of its own, and along time
the power spent on the carrier, none of it the magnetic field's. For a drift through space the field's commutator is
their geometric product, so that Drude's steady state, the drag balancing the push, is solved by inverting one element,
`1 - valence * mobility * magnetic`, a scalar plus a bivector: a rotor scaled by the secant of the kind's Hall angle,
the angle opposite for opposite charges.
A carrier carries what it holds as the flux `velocity.regressive(parcel)`, and the response, the map from a drive to the
flux it drives, is the sum over the kinds of their drifts per drive joined with their parcels, with the lattice's
conduction of entropy. One sum of turned drives is read two ways: through the planes that read charge, the kinds' turns
cancel at compensation, and through those that read entropy, which both kinds carry along their drift, they add.
Reversing the field turns the response into its adjugate, the Onsager-Casimir reciprocity, and a drive's reading of the
flux it drives, `drive & flux`, the dissipation, is never negative, a pairing every observer reads alike.

A hat's gradient wedged with a flux, `gradient ^ flux`, is what the flux carries out of the corner's cell, charge and
entropy; the steady state is the linear response, the potentials at which no free cell gains either, the contacts held
and the rest of the border insulated, the entropy the drives produce left out. The conductor is driven two ways: by a
current between its contacts, both at one temperature, and by heat, its contacts at two temperatures, passing no
current, as with the circuit between them open. Potentials are measured from their values in equilibrium; lengths are
in the bar's width or the ring's outer radius, charges in a carrier's charge, densities in the carriers' total, and the
field in the inverse of the mobility, so that its strength is the tangent of the Hall angle.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

# Planar spacetime, and the entropy a parcel holds, a direction of its own with no metric.
ga = Algebra("x+y+t-") * Algebra("s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# What a parcel holds, charge along time and entropy, the points that span the plane; and a direction through space.
Parcel = Point.derive.containing("x y")                                       # Antivector-
Spatial = Point.derive.containing("t s")                                      # Antivector-
# The planes reading a place's run along the conductor and across it; the potentials, the electrochemical potential
# along t and the temperature along s; their drive; the magnetic field, a plane of space; a flux; a carrier's velocity
# per drive; and the response, the map from a drive to the flux it drives.
Upright = ga.gatype.from_blades("x y")                                        # Vector+
Potential = Plane.derive.within("t s")                                        # Vector+
Drive = ga.gatype.from_blades("xt yt xs ys")                                  # Bivector+
Magnetic = ga.gatype.from_blades("xy")                                        # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
Drifting = ga.gatype((Spatial, Drive))                                        # Antivector- <- Bivector+
Response = ga.gatype((Flux, Drive))                                           # Antibivector- <- Bivector+
axis = mv.x.dual()                                                            # [] Point-
across = mv.y.dual()                                                          # [] Point-
# A unit of charge, a direction along time, and a unit of entropy, the weight of a place.
time = -mv.t.dual()                                                           # [] Parcel-
entropy = mv.s.dual()                                                         # [] Parcel-


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Each triangle's area `Scalar+[faces]`, and each corner's hat gradient `[corners] Plane+[faces]`, the plane
    through the facing side, along time and at unit entropy, over twice the area, so that it reads one from the side to
    the corner."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, read through time at unit entropy.
    double_areas = ((sides[0] & sides[1]) & time) & entropy                   # Scalar-[faces]
    return double_areas.norm() / 2, (time & entropy & sides) / double_areas


def drifts(magnetic: Magnetic, valences: np.ndarray, entropies_per_charge: np.ndarray, mobilities: np.ndarray
           ) -> Drifting:
    """Each kind of carrier's velocity per drive `[..., kinds] Spatial- <- Drive+` in each magnetic field `[...]
    Bivector+`, a plane of space: the kind's charge, the entropy it carries per charge and its mobility, the velocity a
    unit push gives it. The velocity is the drive's push on the kind's parcel at rest, turned through the kind's Hall
    angle and shortened by its cosine: Drude's steady state, in which the drag balances the push and the field's
    turning of the drift."""
    parcels = valences * (time + entropies_per_charge * entropy)             # [kinds] Parcel-
    # The whole push on a moving carrier is one commutator, `(magnetic + drive).commutator(parcel + valence *
    # velocity)`. Through space it is the drive's push on the parcel at rest, its charge down the fall of the
    # electrochemical potential and its entropy down the fall of temperature, and the field's turning of the drift.
    push = Drive.commutator(parcels)                                          # [kinds] Spatial- <- Drive+
    # The field turns a drift through space by their geometric product, by the parcel's charge, so the drift balances
    # `velocity == mobility * (push + valence * magnetic * velocity)`: one less the mobility times the field, a scalar
    # plus a bivector, a rotor scaled by the secant of the Hall angle, times the velocity is the mobility times the push.
    hall = 1 - valences * mobilities * magnetic[..., None]                   # [..., kinds] Even+
    return hall.inverse() * (mobilities * push)


def response(magnetic: Magnetic, valences: np.ndarray, entropies_per_charge: np.ndarray, mobilities: np.ndarray,
             densities: np.ndarray, conductivity: float) -> Response:
    """The conductor's response `[...] Flux- <- Drive+`, the flux a drive drives, in each magnetic field `[...]
    Bivector+`: each kind of carrier, at its density per area `[..., kinds]`, drifting as `drifts` has it and carrying
    what it holds; and the lattice, conducting entropy down a fall of temperature at this conductivity."""
    parcels = valences * (time + entropies_per_charge * entropy)             # [kinds] Parcel-
    drifting = drifts(magnetic, valences, entropies_per_charge, mobilities)  # [..., kinds] Spatial- <- Drive+
    carrying = (densities * drifting.regressive(parcels)).sum(axis=-1)       # [...] Flux- <- Drive+
    conducting = conductivity * Drive.commutator(entropy).regressive(entropy)   # [] Flux- <- Drive+
    return carrying + conducting


def outflow(topology: Topology, areas: Scalar, gradients: Plane, response: Response) -> SparseExtensor:
    """What each vertex's cell loses as the potentials rise at each vertex, charge and entropy, `(Parcel- <-
    Potential+)[vertices <- vertices]`, under this response `[] Flux- <- Drive+`: the drive of a hat's rise, minus its
    gradient wedged with the potentials, drives a flux, which the other hats read as what it carries out of their
    cells, over the triangle, gathered onto the vertices."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    driving = -(gradients ^ Potential)                                        # [corners] (Drive+ <- Potential+)[faces]
    local = areas * (gradients[:, None] ^ response(driving)[None, :])        # [corners, corners] (Parcel- <- Potential+)[faces]
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)


def steady(topology: Topology, areas: Scalar, gradients: Plane, response: Response, high: np.ndarray, low: np.ndarray,
           current: float, temperature_difference: float) -> tuple[Potential, Potential]:
    """The conductor's potentials in its two steady states `Potential+[vertices]` under this response `[] Flux- <-
    Drive+`, its contacts at the vertices `high` `[high]` and `low` `[low]` and the rest of its border insulated: driven
    by this current, entering at `high` and leaving at `low`, both contacts at one temperature; and driven by heat,
    `high` warmer than `low` by this difference of temperature, no current passing between them. Each holds no free
    cell gaining charge or entropy, the linear response."""
    shares = np.zeros(topology.n_vertices)
    shares[high], shares[low] = 1, -1
    towards_high = mv.scalar(shares[:, None]).field()                         # Scalar+[vertices]
    # The two drives at the contacts, a unit fall of the electrochemical potential and a unit fall of the temperature
    # from `high` to `low`, nothing elsewhere; and the reading of the free vertices.
    held = towards_high * stack((mv.t, mv.s)) / 2                             # [drives] Potential+[vertices]
    free = topology.reading(np.setdiff1d(np.arange(topology.n_vertices), np.concatenate([high, low])), 0)
    losing = outflow(topology, areas, gradients, response)                    # (Parcel- <- Potential+)[vertices <- vertices]
    loose = free * losing * ~free                                             # (Parcel- <- Potential+)[free <- free]
    potentials = held + ~free * loose.solve(-(free * losing(held)))           # [drives] Potential+[vertices]
    # The charge entering the conductor at the high contact, what the contact's cells lose to the rest.
    currents = mv.t & (topology.reading(high, 0) * losing(potentials)).sites.sum()   # [drives] Scalar+
    # Driven by a current: the fall of the potential scaled to pass it. Driven by heat: the fall of temperature, less
    # the fall of the potential that passes its current back, scaled to the difference of temperature.
    electric, thermal = potentials
    electric_current, thermal_current = currents
    return (electric * (current / electric_current),
            (thermal - electric * (thermal_current / electric_current)) * temperature_difference)


def sweep(topology: Topology, areas: Scalar, gradients: Plane, high: np.ndarray, low: np.ndarray, current: float,
          temperature_difference: float, magnetic: Magnetic, valences: np.ndarray, entropies_per_charge: np.ndarray,
          mobilities: np.ndarray, densities: np.ndarray, conductivity: float
          ) -> Iterator[tuple[np.ndarray, Potential, Spatial, Potential, Spatial]]:
    """The conductor's two steady states, as `steady` finds them at this current and this difference of temperature,
    at each frame's magnetic field `[frames] Bivector+` and densities of the carriers `[frames, kinds]` in turn: the
    densities `[kinds]`, and driven by the current and then by heat, the potentials `Potential+[vertices]` and each kind
    of carrier's velocity `[kinds] Spatial-[faces]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    for frame, frame_densities in enumerate(densities):
        drifting = drifts(magnetic[frame], valences, entropies_per_charge, mobilities)   # [kinds] Spatial- <- Drive+
        conducting = response(magnetic[frame], valences, entropies_per_charge, mobilities, frame_densities,
                              conductivity)                                   # [] Flux- <- Drive+
        electric, thermal = steady(topology, areas, gradients, conducting, high, low, current, temperature_difference)
        # Each triangle's drive: its corners' potentials on their hats' gradients.
        potentials = stack((electric, thermal))                               # [drives] Potential+[vertices]
        drives = -(gradients ^ (corners * potentials[:, None])).sum(axis=-1)  # [drives] Drive+[faces]
        yield frame_densities, electric, drifting(drives[0]), thermal, drifting(drives[1])
