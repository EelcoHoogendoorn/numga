"""The responses a crystal allows are its measured responses averaged over its point group.

A response looks the same after any of the crystal's symmetries, so the average of a measured response over the
symmetry group keeps exactly the part the crystal allows. A positive heat conductivity, a map from driving fields to
heat flows along directions, averaged over rotation groups, aligns with the planes of the group's half turns, grows
round in the plane of its quarter turns, and under the cube's rotations becomes the same in every direction. The
bonds of a cubic lattice, averaged over the cube's rotations, give that isotropic conduction and yet an anisotropic
elasticity, a form on four planes that read a strain. A rotary power is an intrinsic scalar whose sign is a sense,
so mirrors and the inversion, odd versors, reverse it, and restrict it where they leave conduction alone.

The averaging is Neumann's principle of crystal physics: https://dictionary.iucr.org/Neumann%27s_principle
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, Extensor, NumpyContext, concatenate
from numga.algebras import PGA3D

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                     # Scalar+
# A driving field, the temperature's fall per unit length, is read by crossing it: a vector. Heat flows,
# bonds and the paths of light are carried by geometry: directions, antivectors. The planes that read
# directions, a strain's and a rotary power's, are vectors too.
Field = ga.gatype.vector()                                                      # Vector+
Plane = ga.gatype.vector()                                                      # Vector+
Direction = ga.gatype.antivector()                                              # Antivector-
# A turn per distance with a sense of its own: an intrinsic scalar.
RotaryPower = ga.gatype.antipseudoscalar()                                 # Antipseudoscalar-
Conductivity = ga.gatype((Direction, Field))                                    # Direction- <- Field+
Elasticity = ga.gatype((Scalar, Plane, Plane, Plane, Plane))                    # Scalar+ <- (Plane+, Plane+, Plane+, Plane+)
Gyration = ga.gatype((RotaryPower, Direction, Direction))                       # RotaryPower- <- (Direction-, Direction-)
# A plane of turning, the turns in it, and the improper rotations, the odd versors.
TurningPlane = ga.gatype.bivector()                                             # Bivector+
Rotor = ga.gatype.rotor()                                                       # Rotor+
ImproperRotation = ga.gatype.odd()                                              # Odd+

# The lattice's sites are projective points.
pga_mv = NumpyContext(PGA3D).multivector
Point = PGA3D.gatype.antivector()                                               # Antivector-

# The latitudes and longitudes the sphere of directions is sampled at.
n_latitudes, n_longitudes = 33, 65


# --- plumbing -------------------------------------------------------------------------
def turns(plane: TurningPlane, order: int) -> Rotor:
    """The cyclic group of `order` equal turns in a unit plane, the identity first."""
    # A rotor turns by twice its angle: element k, `(plane * (np.pi * k / order)).exp()`, turns by
    # `2 * np.pi * k / order`.
    return (plane * (np.pi * np.arange(order) / order)).exp()


def cube_rotations() -> Rotor:
    """The 24 rotations of a cube: each of the six faces turned to the top, then four quarter turns in `mv.xy`."""
    tilt = np.pi / 4 * np.array([0, 1, 2, 3, 0, 0])
    roll = np.pi / 4 * np.array([0, 0, 0, 0, 1, -1])
    faces = (mv.yz * tilt + mv.zx * roll).exp()
    return (faces[:, None] * turns(mv.xy, 4)[None, :]).reshape(-1)


def sphere_turns() -> Rotor:
    """Turns that carry x over a sphere, sampled for drawing: up toward z by the latitude, then in `mv.xy` by the
    longitude."""
    longitude = np.linspace(0, 2 * np.pi, n_longitudes)[None, :]
    latitude = np.linspace(-np.pi / 2, np.pi / 2, n_latitudes)[:, None]
    return (mv.xy * (-longitude / 2)).exp() * (mv.xz * (-latitude / 2)).exp()        # [latitudes, longitudes] Rotor+


# A sphere of unit directions, and the unit field across each of them, the plane that reads it.
sphere = sphere_turns() >> mv.x.dual()                                               # [latitudes, longitudes] Direction-
unit_fields = sphere_turns() >> mv.x                                                 # [latitudes, longitudes] Field+


def lattice_samples() -> tuple[Point, Point, Point, Point]:
    """PGA sites of a simple-cubic lattice, by their shells about the central site: the centre, its axial
    neighbours, its face-diagonal ones, and the corners."""
    i, j, k = np.indices((3, 3, 3)) - 1
    sites = pga_mv.xyz + pga_mv.wzy * i + pga_mv.wxz * j + pga_mv.wyx * k
    shell = i * i + j * j + k * k
    return sites[shell == 0], sites[shell == 1], sites[shell == 2], sites[shell == 3]


# --- math -----------------------------------------------------------------------------
def conductivities(axes: Direction, gains: np.ndarray, groups: list[Rotor]) -> Conductivity:
    """A positive conductivity along principal axes, followed by its average over each group.

    A conductivity maps the driving field, the temperature's fall, to heat flow; in an anisotropic material the
    flow need not run along the field. A symmetry moves the conductivity as it moves anything,
    `rotor >> conductivity(rotor << Field)`, and the average over a group is the Reynolds projection onto the
    responses the group leaves unchanged: the closest such response in the Frobenius norm, still positive.
    Symmetry constrains the response, not its gains.
    """
    # Each principal axis reads the driving field by their pairing and contributes heat flow along
    # itself. Positive gains make this a passive conductivity.
    conductivity = (axes * (Field & axes) * gains).sum(axis=0)                     # [] Direction- <- Field+
    # Pull the field into each rotated frame, apply the conductivity, and turn the flow back. The mean is
    # unchanged by any rotation in the group: those rotations merely permute the terms.
    return Extensor.stack([conductivity] + [(group >> conductivity(group << Field)).mean(axis=0) for group in groups])


def rotary_powers(planes: Plane, powers: RotaryPower, groups: list[tuple[Rotor, ImproperRotation]]) -> Gyration:
    """A candidate rotary power with principal planes, followed by its average over each point group.

    Light crossing an optically active crystal along a unit direction `d` has its plane of polarization
    turned, in proportion to the distance, by the rotary power `gyration(d, d)`. The turn has a sense,
    so the rotary power is an intrinsic scalar: the mirror image of a crystal turns light the other way.
    Each principal plane reads the light's direction, `plane & d`.

    A crystal's point group holds its rotations and, where the crystal has mirrors or a centre of
    inversion, its improper rotations, the odd versors. An improper rotation reverses the sense of an
    intrinsic output, so a group allows less optical activity than its rotations alone, and none where
    it contains the inversion.
    """
    gyration = ((planes & Direction) * (planes & Direction) * powers).sum(axis=0)   # [] RotaryPower- <- (Direction-, Direction-)

    def average(rotations: Rotor, improper: ImproperRotation) -> Gyration:
        # Turn both directions into each element's frame, and carry the rotary power back out: an improper
        # rotation reverses its sense.
        return concatenate([element >> gyration(element << Direction, element << Direction) for element in (rotations, improper)]).mean(axis=0)

    return Extensor.stack([gyration] + [average(rotations, improper) for rotations, improper in groups])


def lattice_responses(seeds: Direction, cube: Rotor) -> tuple[Conductivity, Elasticity]:
    """The conductivity and the elasticity of a simple-cubic lattice's bonds, averaged over the cube's rotations.

    A simple-cubic lattice has axial and face-diagonal bond families, and the 24 cube rotations turn a seed bond
    of each family, `mv.x.dual()` and `(mv.x + mv.y).dual().normalized()`, into every bond in it. Every member
    has the same weight, so another cube rotation only permutes the terms in the average: this is the
    microscopic meaning of the group projection.

    The two families carry equal total weights. For the elastic model these are unstressed central springs at
    lattice spacing 1, of axial stiffness 1/3 and face-diagonal stiffness 1/12, whose bond counts and squared
    lengths give each family weight 1; the diagonal springs provide the shear stiffness. This model has equal
    elastic constants c12 and c44, a relation a general cubic crystal need not have.

    A unit bond reads a driving field through their pairing, `field & bond`. A uniaxial strain of size s across
    a unit plane, `s * plane * (plane & Direction)`, stretches the bond by the fraction `s * (plane & bond) ** 2`,
    and the spring energy, its square, makes the elasticity a form on four planes. Its value
    `elasticity(plane, plane, plane, plane)` is the stiffness, with energy density
    `s ** 2 * elasticity(plane, plane, plane, plane) / 2`; it holds transverse strain fixed, and is not the
    relaxed Young's modulus.

    Cubic elasticity background: https://www.ctcms.nist.gov/oof/oof1/Manual/node152.html
    """
    # These are free bond directions; the actual lattice sites are projective points.
    bonds = cube[:, None] >> seeds[None, :]                                      # [rotations, families] Direction-
    # Each bond read by a plane; a uniaxial strain across the plane stretches the bond by the reading's square.
    readings = Plane & bonds                                                     # [rotations, families] Scalar+ <- Plane+

    # Average over the group, then add the two families. No direction is privileged within either
    # orbit. Conductivity reads the field along each bond and returns flow along it.
    conductivity = (bonds * (Field & bonds)).mean(axis=0).sum(axis=0)            # [] Direction- <- Field+
    # Four independent inputs encode the spring stiffness. The same group average
    # leaves cubic structure here, although it made the conduction isotropic.
    elasticity = (readings * readings * readings * readings).mean(axis=0).sum(axis=0)   # [] Scalar+ <- (Plane+, Plane+, Plane+, Plane+)
    return conductivity, elasticity
