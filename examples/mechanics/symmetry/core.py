"""Which responses survive averaging over an object's rotation symmetries?

Heat conduction: a positive conductivity, averaged over rotation groups, and its heat-flow ellipsoids.
Crystal lattice: the same cubic bond orbits give isotropic conduction and anisotropic elasticity.
Optical activity: a rotary power has a sense, so mirrors and the inversion restrict it where they leave conduction alone.

Crystal-symmetry motivation: https://dictionary.iucr.org/Neumann%27s_principle
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, Extensor, NumpyContext, concatenate
from numga.algebras import PGA3D

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
# A driving field, the temperature's fall per unit length, is read by crossing it: a vector. Heat flows,
# bonds and the paths of light are carried by geometry: directions, antivectors.
Field = ga.gatype.vector()
Direction = ga.gatype.antivector()
Conductivity = ga.gatype((Direction, Field))       # Direction <- Field
Rotor = ga.gatype.rotor()
ImproperRotation = ga.gatype.odd()
Plane = ga.gatype.bivector()

# The lattice's sites are projective points.
pga_mv = NumpyContext(PGA3D).multivector
Point = PGA3D.gatype.antivector()


# --- plumbing -------------------------------------------------------------------------
def turns(plane: Plane, order: int) -> Rotor:
    """The cyclic group of `order` equal turns in a unit plane, the identity first."""
    # A rotor turns by twice its angle: element k, `(plane * (np.pi * k / order)).exp()`, turns by
    # `2 * np.pi * k / order`.
    return (plane * (np.pi * np.arange(order) / order)).exp()


def cube_rotations() -> Rotor:
    """The 24 rotations of a cube: each of the six faces turned to the top, then four quarter turns about z."""
    tilt = np.pi / 4 * np.array([0, 1, 2, 3, 0, 0])
    roll = np.pi / 4 * np.array([0, 0, 0, 0, 1, -1])
    faces = (mv.yz * tilt + mv.zx * roll).exp()
    return (faces[:, None] * turns(mv.xy, 4)[None, :]).reshape(-1)


def sphere_turns() -> Rotor:
    """Turns that carry x over a sphere, sampled for drawing: up toward z by the latitude, then about z by
    the longitude."""
    longitude = np.linspace(0, 2 * np.pi, 65)[None, :]
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 33)[:, None]
    return (mv.xy * (-longitude / 2)).exp() * (mv.xz * (-latitude / 2)).exp()        # [latitudes, longitudes] Rotor


def fields() -> Field:
    """A sphere of unit driving fields."""
    return sphere_turns() >> mv.x                                                    # [latitudes, longitudes] Field


def directions() -> Direction:
    """A sphere of unit directions, each the direction of the field at the same place in `fields`."""
    return sphere_turns() >> mv.x.dual()                                             # [latitudes, longitudes] Direction


def lattice_samples():
    """PGA sites of a simple-cubic lattice, with the central site's two neighbour shells."""
    i, j, k = np.indices((3, 3, 3)) - 1
    sites = pga_mv.xyz + pga_mv.wzy * i + pga_mv.wxz * j + pga_mv.wyx * k
    shell = i * i + j * j + k * k
    return sites.reshape(-1), sites[shell == 1], sites[shell == 2]


# --- math -----------------------------------------------------------------------------
def conductivities(axes: Direction, gains: np.ndarray, groups: list[Rotor]) -> Conductivity:
    """Which heat-conduction responses are compatible with a crystal's symmetry?

    Conductivity maps the driving field, the temperature's fall, to heat flow. In an anisotropic
    material the flow need not run along the field. Start with a positive conductivity,
    rotate the whole response through each symmetry, and average. This Reynolds
    projection gives the closest invariant response in the Frobenius norm, preserving
    positive definiteness. Symmetry constrains the response, not its numerical gains.

    Returns the measured conductivity followed by its average over each group.
    """
    # Each principal axis reads the driving field by their pairing and contributes heat flow along
    # itself. Positive gains make this a passive conductivity.
    conductivity = (axes * (Field & axes) * gains).sum(axis=0)                     # Direction <- Field
    # Pull the field into each rotated frame, apply the conductivity, and turn the flow back. The mean is
    # unchanged by any rotation in the group: those rotations merely permute the terms. This projects a
    # measured response onto the allowed ones.
    return Extensor.stack([conductivity] + [(group >> conductivity(group << Field)).mean(axis=0) for group in groups])


def rotary_powers(axes: Direction, powers: np.ndarray, groups: list[tuple[Rotor, ImproperRotation]]) -> Extensor:
    """Which crystals can rotate polarized light?

    Light crossing an optically active crystal along a unit direction `d` has its plane of polarization
    turned, in proportion to the distance, by the rotary power `gyration(d, d)`. The turn has a sense,
    so the rotary power is an intrinsic scalar: the mirror image of a crystal turns light the other way.
    Its principal values are measured against the lab's handedness, the volume `x ^ y ^ z` read as a
    number.

    A crystal's point group holds its rotations and, where the crystal has mirrors or a centre of
    inversion, its improper rotations, the odd versors. An improper rotation reverses the sense of an
    intrinsic output, so a group allows less optical activity than its rotations alone, and none where
    it contains the inversion.

    Returns the candidate followed by its average over each group.
    """
    handedness = mv.xyz.dual()                                                   # [] Scalar, intrinsic
    gyration = (axes.dot(Direction) * axes.dot(Direction) * powers).sum(axis=0) * handedness   # Scalar <- (Direction, Direction)

    def average(rotations: Rotor, improper: ImproperRotation) -> Extensor:
        # Turn both directions into each element's frame, and carry the rotary power back out: an improper
        # rotation reverses its sense.
        return concatenate([element >> gyration(element << Direction, element << Direction) for element in (rotations, improper)]).mean(axis=0)

    return Extensor.stack([gyration] + [average(rotations, improper) for rotations, improper in groups])


def lattice_responses(seeds: Direction, cube: Rotor) -> tuple[Conductivity, Extensor]:
    """Average microscopic bond responses over a crystal's cubic point group.

    A simple-cubic lattice has axial and face-diagonal bond families. Starting with
    `mv.x.dual()` and `(mv.x + mv.y).dual().normalized()`, the 24 cube rotations generate every
    direction in each family. Every member has the same weight, so another cube
    rotation just permutes the terms in the average. This is the microscopic meaning
    of the group projection.

    Give the two families equal total response weights. For the elastic model, take
    unstressed central springs with lattice spacing 1, axial stiffness 1/3, and
    face-diagonal stiffness 1/12. Bond counts and squared lengths give weight 1 to
    each family. Diagonal springs provide shear stiffness. This particular model has
    equal elastic constants c12 and c44; a general cubic crystal need not have that
    extra relation.

    A unit bond n reads a driving field f through their pairing, `f & n`. For an imposed
    strain of size s along a unit direction d, its fractional extension is `s * n.dot(d) ** 2`,
    measured by the metric between the two directions. Squaring to get spring energy gives
    four factors of `n.dot(d)`, hence a scalar form with four direction inputs. Its value
    `elasticity(d, d, d, d)` is the stiffness, with energy density
    `s ** 2 * elasticity(d, d, d, d) / 2`. It holds transverse strain fixed; it is not
    the relaxed Young's modulus.

    Cubic elasticity background: https://www.ctcms.nist.gov/oof/oof1/Manual/node152.html
    Returns the conductivity and the elasticity.
    """
    # Batch axes are [24 rotations, 2 bond families]. These are free bond directions;
    # the actual lattice sites are projective points.
    bonds = cube[:, None] >> seeds[None, :]                                      # [rotations, families] Direction

    # Average over the group, then add the two families. No direction is privileged within either
    # orbit. Conductivity reads the field along each bond and returns flow along it.
    conductivity = (bonds * (Field & bonds)).mean(axis=0).sum(axis=0)            # Direction <- Field
    projection = bonds.dot(Direction)                                            # [rotations, families] Scalar <- Direction

    # Four independent inputs encode the spring stiffness. The same group average
    # leaves cubic structure here, although it made the conduction isotropic.
    elasticity = (projection * projection * projection * projection).mean(axis=0).sum(axis=0)
    return conductivity, elasticity
