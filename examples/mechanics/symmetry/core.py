"""Material responses allowed by a crystal are measured responses averaged over its point group.

A response looks the same after any of the crystal's symmetries, so averaging a measured response over the point
group keeps the invariant part that the crystal allows. Positive heat conductivity maps driving fields to heat
flows along directions; averaged over rotation groups, it aligns with the planes of half turns, grows round in the
plane of quarter turns, and becomes isotropic under the rotations of a cube. Averaging the bonds of a simple-cubic
lattice over the cube rotations gives isotropic conduction alongside anisotropic cubic elasticity on four strain
planes. Rotary power is an intrinsic scalar whose sign carries a sense of turn, so mirror reflections and spatial
inversion reverse it where they leave conductivity unchanged.

The group average realizes Neumann's principle of crystal physics: https://dictionary.iucr.org/Neumann%27s_principle
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, Extensor, NumpyContext, concatenate

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                     # Scalar+
# Driving fields (such as a temperature slope) and planes that read bond stretch or light directions are
# extrinsic planes; heat flows, bonds, and light paths are intrinsic directions.
Field = ga.gatype.vector()                                                      # Vector+
Plane = ga.gatype.vector()                                                      # Vector+
Direction = ga.gatype.antivector()                                              # Antivector-
# A polarization turn per unit distance carries an intrinsic sense of rotation.
RotaryPower = ga.gatype.antipseudoscalar()                                      # Antipseudoscalar-
Conductivity = ga.gatype((Direction, Field))                                    # Direction- <- Field+
Elasticity = ga.gatype((Scalar, Plane, Plane, Plane, Plane))                    # Scalar+ <- (Plane+, Plane+, Plane+, Plane+)
Gyration = ga.gatype((RotaryPower, Direction, Direction))                       # RotaryPower- <- (Direction-, Direction-)
# Turning planes, proper rotations, and improper rotations (reflections and spatial inversion).
TurningPlane = ga.gatype.bivector()                                             # Bivector+
Rotor = ga.gatype.rotor()                                                       # Rotor+
ImproperRotation = ga.gatype.odd()                                              # Odd+


# --- math -----------------------------------------------------------------------------
def turns(plane: TurningPlane, n_turns: int) -> Rotor:
    """The cyclic group of `n_turns` equal rotations (`[n_turns] Rotor+`) in `plane` starts at the identity."""
    # A rotor turns by twice its bivector angle, so `pi * k / n_turns` turns by `2 * pi * k / n_turns`.
    return (plane * (np.pi * np.arange(n_turns) / n_turns)).exp()               # [n_turns] Rotor+


def cube_rotations() -> Rotor:
    """The 24 proper rotations of a cube (`[24] Rotor+`) bring each of the six faces to the top before four quarter turns."""
    tilt = np.pi / 4 * np.array([0, 1, 2, 3, 0, 0])
    roll = np.pi / 4 * np.array([0, 0, 0, 0, 1, -1])
    faces = (mv.yz * tilt + mv.zx * roll).exp()                                 # [6] Rotor+
    return (faces[:, None] * turns(mv.xy, 4)[None, :]).reshape(-1)              # [24] Rotor+


def conductivities(axes: Direction, gains: np.ndarray, groups: list[Rotor]) -> Conductivity:
    """A positive conductivity along `axes` and its Reynolds averages over `groups` form `[1 + len(groups)]` maps (`Direction- <- Field+`).

    Conductivity maps a driving field to a heat flow that need not align with the field in an anisotropic crystal.
    Averaging `rotor >> conductivity(rotor << Field)` over a finite rotation group projects onto the closest
    group-invariant positive conductivity in the Frobenius norm.
    """
    # Each principal axis pairs with the driving field and carries heat flow along itself.
    conductivity = (axes * (Field & axes) * gains).sum(axis=0)                  # [] Direction- <- Field+
    # Pulling the field into each rotated frame and turning the resulting flow back leaves the group mean invariant.
    return Extensor.stack([conductivity] + [(group >> conductivity(group << Field)).mean(axis=0) for group in groups])


def lattice_responses(seed_bonds: Direction, cube: Rotor) -> tuple[Conductivity, Elasticity]:
    """Averaging `seed_bonds` over `cube` yields the simple-cubic lattice conductivity and four-plane elasticity.

    The 24 cube rotations carry axial and face-diagonal seed bonds onto every bond of their families with equal
    weight, so another cube rotation permutes the terms in the group average. At unit lattice spacing with axial
    spring stiffness 1/3 and face-diagonal stiffness 1/12, both families carry unit total weight and satisfy the
    central-force Cauchy relation `c12 == c44`.

    A uniaxial strain of magnitude `s` across a unit plane stretches a unit bond by `s * (plane & bond) ** 2`, so
    the quadratic spring energy defines a symmetric four-plane form `elasticity(plane, plane, plane, plane)` holding
    transverse strain fixed: https://www.ctcms.nist.gov/oof/oof1/Manual/node152.html
    """
    bonds = cube[:, None] >> seed_bonds[None, :]                                # [..., 2] Direction-
    # Each bond paired with a plane gives the directional stretch factor whose fourth power enters spring energy.
    stretch_readings = Plane & bonds                                            # [..., 2] Scalar+ <- Plane+

    # Averaging over the group and summing the two families makes conduction isotropic while keeping cubic elasticity.
    conductivity = (bonds * (Field & bonds)).mean(axis=0).sum(axis=0)           # [] Direction- <- Field+
    elasticity = (stretch_readings * stretch_readings * stretch_readings * stretch_readings).mean(axis=0).sum(axis=0)  # [] Scalar+ <- (Plane+, Plane+, Plane+, Plane+)
    return conductivity, elasticity


def rotary_powers(planes: Plane, powers: RotaryPower, groups: list[tuple[Rotor, ImproperRotation]]) -> Gyration:
    """A candidate gyration map on `planes` and its point-group averages over `groups` form `[1 + len(groups)]` maps (`RotaryPower- <- (Direction-, Direction-)`).

    Light travelling along a unit direction `d` has its polarization plane turned per unit distance by `gyration(d, d)`.
    Because rotary power is an intrinsic scalar, improper rotations reverse its sense when carrying the output back,
    cancelling optical activity completely whenever the point group contains spatial inversion.
    """
    gyration = ((planes & Direction) * (planes & Direction) * powers).sum(axis=0)  # [] RotaryPower- <- (Direction-, Direction-)

    def group_average(rotations: Rotor, improper_rotations: ImproperRotation) -> Gyration:
        # Improper versors flip the intrinsic sense of `gyration` when sandwiching the output back to the crystal frame.
        return concatenate([
            versor >> gyration(versor << Direction, versor << Direction)
            for versor in (rotations, improper_rotations)
        ]).mean(axis=0)

    return Extensor.stack([gyration] + [group_average(rotations, improper_rotations) for rotations, improper_rotations in groups])
