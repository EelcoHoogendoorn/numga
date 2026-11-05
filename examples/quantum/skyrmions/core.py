"""Magnetic skyrmions in a thin chiral magnet: spins on a triangular lattice, their exchange carried across each bond by
a rotor, settle from noise into a lattice of skyrmions, vanish into a ferromagnet and give way to a spiral as the field
changes, and drift with a current; in the algebra of space.

A chiral magnet is a crystal without a mirror symmetry, in which neighbouring spins prefer to stand a little turned
against each other, always in the same sense, while the exchange prefers them parallel. The compromise is a spiral, and
in a field a lattice of skyrmions: small whirls in which the spin turns over once, from along the field at the rim to
against it at the core. A skyrmion is a knot in the spin texture: its spins point every way once, so that no smooth
change of the texture undoes it, and the texture's winding counts the skyrmions.

The algebra is the algebra of space, VGA3D. A spin is modelled as the patch of the plane its moment turns in, `Spin`,
intrinsic, and a field as the plane it turns spins in, `Field`, extrinsic; their pairing, `field & spin`, is minus the
energy of the one in the other. The exchange pairs a spin with the field its neighbour exerts on it, `exchange =
Spin.dual()`, after the neighbour's spin has been carried across the bond by a rotor. The bonds' rotors are a
connection: a rotor turning about its own bond, `twist * bonds` with the twist per unit length an intrinsic scalar whose
sense is the crystal's handedness, makes the exchange exactly the Dzyaloshinskii-Moriya interaction, with an exchange
anisotropy of second order beside it. Carried around a triangle, the rotors turn a spin in the film's plane, by about
the twist squared times the triangle's area: the connection is curved, so no turning of the spins removes the twist, and
the texture is left its whirls.

The effective field on a spin is written out: the applied field, plus on every bond at the site the field the spin at
the bond's other end exerts, once carried across the bond, a tail's spin to the head by `carriers >> tails` and a head's
back to the tail by `carriers << heads`, the sandwich that undoes it. The energy is bilinear in the spins, so that sum
is the energy's derivative with respect to each spin, paired back through `pairing = Field & Spin`, the Zeeman energy's
own pairing; the tests confirm it against JAX's derivative. Each spin precesses in its field's plane; Gilbert's damping
adds that plane's quarter turn about the spin, `turning.commutator(spins.dual())`, which turns the spin toward its
field; a current carries the texture along, turning each spin in the plane through it and its change along the drift;
and a thermal field kicks it. Every step turns every spin by a rotor, so the spins stay unit by construction. A
triangle's topological charge is the solid angle its three spins enclose: the shortest turns from spin to spin around it
compose to a turn about the first spin by that angle, as parallel transport around a triangle of the sphere does. The
stepping is jitted and scanned by JAX.

Energies are in units of the exchange, lengths in lattice spacings and times in the exchange's precession time.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import stack, where
from numga.algebras import VGA3D as ga
from numga.backend.jax import JaxContext
from examples.util.topology import Topology

jax.config.update("jax_enable_x64", True)

mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# A place in the film, a step from its origin, and a bond, a step from site to site.
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# A spin: the patch of the plane its moment turns in. A field: the plane it turns spins in.
Spin = ga.gatype.antibivector()                                               # Antibivector-
Field = ga.gatype.bivector()                                                  # Bivector+
Rotor = ga.gatype.rotor()                                                     # Rotor+
# The film's plane, which the applied field turns spins in, and the step off the film.
film = mv.xy                                                                  # [] Field+
off_film = mv.z.dual()                                                        # [] Point-
# The exchange, in units of its strength: the field a spin exerts on a neighbour, turning it in the spin's own plane.
exchange = Spin.dual()                                                        # [] Field+ <- Spin-
# A field's pairing with a spin, minus the energy of the one in the other.
pairing = Field & Spin                                                        # [] Scalar+ <- (Field+, Spin-)


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, bonds: Point) -> tuple[Scalar, Plane, Scalar]:
    """Each triangle's area `Scalar-[faces]`; each corner's hat gradient `[corners] Plane+[faces]`, the plane through
    the facing side and the step off the film over twice the area, which reads one from the side to the corner; and each
    site's share of the area around it `Scalar-[vertices]`; for the bonds `Point-[edges]`, each the step from its tail to
    its head."""
    sides = topology.boundary[2, 1] * bonds                                   # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined with the step off the film.
    double_areas = (sides[0] & sides[1]) & off_film                           # Scalar-[faces]
    gradients = (off_film & sides) / double_areas                             # [corners] Plane+[faces]
    areas = double_areas / 2                                                  # Scalar-[faces]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=0)


def energy(topology: Topology, carriers: Rotor, applied: Field, spins: Spin) -> Scalar:
    """The energy `[] Scalar+` of the spins `Spin-[vertices]` in the applied field: on every bond, the field its tail's
    spin exerts on its head's spin, the tail's spin carried across the bond by the bond's rotor `Rotor+[edges]`, paired
    with the head's spin; and the applied field paired with every spin; each taken negative."""
    tails, heads = topology.incidence[1, 0] * spins                           # Spin-[edges], Spin-[edges]
    exchange_fields = exchange(carriers >> tails)                             # Field+[edges]
    return -(exchange_fields & heads).sites.sum() - (applied & spins).sites.sum()


def effective_field(topology: Topology, carriers: Rotor, applied: Field, spins: Spin) -> Field:
    """The effective field on each spin `Field+[vertices]`, for the spins `Spin-[vertices]` in the applied field: the
    applied field, plus on every bond at the site the field the spin at the bond's other end exerts, carried across the
    bond by its rotor `Rotor+[edges]`; the energy, bilinear in the spins, falls along it."""
    tails, heads = topology.incidence[1, 0] * spins                           # Spin-[edges], Spin-[edges]
    # The field each bond brings to its ends: the other end's spin, carried across the bond, back to the tail by the
    # sandwich that undoes the carrying.
    to_tails = exchange(carriers << heads)                                    # Field+[edges]
    to_heads = exchange(carriers >> tails)                                    # Field+[edges]
    return (~topology.incidence[1, 0] * stack((to_tails, to_heads))).sum(axis=0) + applied


def dynamics(topology: Topology, carriers: Rotor, areas: Scalar, gradients: Plane, cells: Scalar, spins: Spin,
             applied: Field, drifts: Point, time_steps: np.ndarray, damping: float, temperature: float, stride: int,
             seed: int) -> Iterator[Spin]:
    """The spins `Spin-[vertices]` at each frame, `stride` steps apart, from these, with the bonds' rotors
    `Rotor+[edges]` carrying the exchange: each spin precesses in its effective field and the thermal field at the
    `temperature`, the latter drawn from the `seed`, damped by Gilbert's `damping`, while in each frame the field
    `[frames] Field+` is applied, a current's spins drift at the velocity `[frames] Point-`, and every step lasts the
    frame's time step `[frames]`; the triangles have these areas `Scalar-[faces]`, hat gradients `[corners]
    Plane+[faces]` and cells `Scalar-[vertices]`."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]

    def step(spins: Spin, kicks: jax.Array, applied: Field, drift: Point, time_step: float) -> Spin:
        """One step: every spin turned by a rotor, at the rate its effective field, the thermal field and the current
        turn it, damped."""
        field = effective_field(topology, carriers, applied, spins)           # Field+[vertices]
        # The thermal field, its spread in each plane the one at which the damping holds the spins at the temperature.
        thermal = (2 * damping * temperature / time_step) ** 0.5 * mv(Field, kicks).field()   # Field+[vertices]
        # Each spin's change along the drift, on each triangle from its corners' hats, shared out to the corners' cells.
        along_faces = ((gradients & drift) * (corners * spins)).sum(axis=0)   # Spin-[faces]
        along = (~corners * (areas / 3 * along_faces)).sum(axis=0) / cells    # Spin-[vertices]
        # Undamped, each spin turns in its fields' planes, and the current turns it back along its change, in the plane
        # through the spin and that change.
        undamped = field + thermal - (spins ^ along)                          # Field+[vertices]
        # Gilbert's damping adds the turning's quarter turn about the spin, which turns the spin toward its field.
        turning = (undamped + damping * undamped.commutator(spins.dual())) / (1 + damping ** 2)   # Field+[vertices]
        rotors = (turning * (-time_step / 2)).exp()                           # Rotor+[vertices]
        return rotors >> spins

    @jax.jit
    def strided(spins: Spin, key: jax.Array, applied: Field, drift: Point, time_step: float) -> Spin:
        kicks = jax.random.normal(key, (stride, topology.n_vertices, 3))
        return jax.lax.scan(lambda spins, kick: (step(spins, kick, applied, drift, time_step), None), spins, kicks)[0]

    keys = jax.random.split(jax.random.PRNGKey(seed), len(applied))
    for frame in range(len(applied)):
        yield spins
        spins = strided(spins, keys[frame], applied[frame], drifts[frame], time_steps[frame])


def holonomies(topology: Topology, rotors: Rotor) -> Rotor:
    """Each face's holonomy `Rotor+[faces]`: the rotors on the edges `Rotor+[edges]`, each carrying its edge's tail to
    its head, multiplied around the face, a rotor's reverse where the face runs against its edge; it carries a spin at
    the face's second corner around the face and back."""
    sides = topology.incidence[2, 1] * rotors                                 # [corners] Rotor+[faces]
    sides = where(topology.orientation[1].T > 0, sides, ~sides)               # [corners] Rotor+[faces]
    # Around the face, each side carries on from where the one before it arrived, multiplying from the left.
    return sides.cumprod(axis=0)[-1]


def charges(topology: Topology, spins: Spin) -> Scalar:
    """Each face's topological charge `Scalar+[faces]`, in whole spheres: the solid angle its corners' spins
    `Spin-[vertices]` enclose, over the sphere's. The shortest turns from spin to spin along the face's edges carry the
    spin at its second corner around it and back, turned about itself by that angle, as parallel transport around a
    triangle of the sphere turns by its area."""
    tails, heads = topology.incidence[1, 0] * spins                           # Spin-[edges], Spin-[edges]
    # The shortest turn from each edge's tail spin to its head's: half the turn of their product.
    shortest = (1 + heads * tails).normalized()                               # Rotor+[edges]
    # The holonomy's logarithm, a half turn in the second corner's own plane.
    halves = holonomies(topology, shortest).log()                             # Field+[faces]
    second = topology.incidence[2, 0][1] * spins                              # Spin-[faces]
    return -2 * (halves & second) / (4 * np.pi)
