"""The Hopf fibration: the unit spinors of three-dimensional space, sorted by their spin.

A unit spinor turns the north pole of the unit sphere into another point of that sphere, its spin. The spinor
holds more than its spin: turning it in the xy plane before it acts changes the spinor and not the point it
turns the pole into. The spinors with each spin therefore form a circle, the fibre over that spin; the fibres
fill the three-sphere of unit spinors, and every two of them are linked once. Carried around a closed loop of
spins, a spinor comes back onto its own fibre, turned along it by half the solid angle the loop encloses.

Spinors are modelled as the even multivectors and spins as points of the unit sphere, antivectors like every
position, and the sandwich `Even >> mv.z.dual()` of the pole is an antivector-valued form on pairs of spinors
whose value on one spinor twice is that spinor's spin. Paired with a spin, `spin.dot(hopf)` is a scalar form on
pairs of spinors, with eigenvalues minus one twice and plus one twice; its top eigenspace is the plane of the
fibre over that spin. A turn on the right, `psi * (mv.xy * angle).exp()`, moves a spinor along its fibre, unseen
by the spin, because a turn in the xy plane leaves the pole where it is. Stereographic projection takes a
spinor to a point of space, its bivector part over one plus its scalar part, made a point by a handedness: an
intrinsic scalar, the right-hand rule that takes a plane of turning to a point. The fibres' linking number is a
sum of regressive products of points, an intrinsic scalar: its sign is the side the turns along the fibres act
on, measured against that handedness.

In quaternion notation a spinor reads as a unit quaternion q and its spin as q k q*; in complex coordinates,
as in Hopf's 1931 paper, the map sends a pair (z1, z2) on the unit sphere of C^2 to the point z1 / z2 of the
Riemann sphere; in bra-ket notation it sends the state of a spin one half to its Bloch vector, and the turn
along the fibre is the Berry phase.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                        # Scalar+
# A spinor's spin, a point of the unit sphere, and a point of the space spinors project into.
Antivector = ga.gatype.antivector()                # Antivector-
# A number with a sense of its own: a handedness, or a number measured against one.
HandedScalar = Scalar.derive.flipped_orientation   # Scalar-
Even = ga.gatype.even()                            # Even+
Rotor = ga.gatype.rotor()                          # Rotor+
# The north pole of the unit sphere.
pole = mv.z.dual()                                 # [] Antivector-
# A spinor's spin, the pole it turns, as a form on pairs of spinors.
hopf = Even >> pole                                # [] Antivector- <- (Even+, Even+)
# The right-hand rule, measured against the basis volume x ^ y ^ z: it takes the bivector part of a spinor to a
# point of space.
handedness = mv(HandedScalar, [1.0])               # [] Scalar-


# --- math -----------------------------------------------------------------------------
def fibre_start(spin: Antivector) -> Even:
    """A unit spinor with each spin: the top eigenvector of the form `spin.dot(hopf)`, whose eigenvalues are
    minus one twice and plus one twice."""
    _, spinors = spin.dot(hopf).eigh()                                         # [..., 4] Even+
    return spinors[..., -1]                                                    # [...] Even+


def stereographic(spinor: Even) -> Antivector:
    """The stereographic projection of unit spinors from -1 into space: the bivector part over one plus the
    scalar part, made a point by the handedness."""
    return handedness * spinor.select[2] / (1 + spinor.select[0])              # [...] Antivector-


def transport(spins: Antivector) -> Rotor:
    """The rotors that carry a frame along a curve of unit spins, by the smallest rotation from each spin to
    the next: from the first spin to each of the others."""
    steps = (1 + spins[1:] * ~spins[:-1]).normalized()                         # [steps, ...] Rotor+
    return steps.cumprod(axis=0)                                               # [steps, ...] Rotor+


def linking(first: Antivector, second: Antivector) -> HandedScalar:
    """Gauss's linking number of pairs of closed polygons, the vertices on the last axis: the regressive
    product of each pair of segments with the line between them, over the cube of its length, summed and
    divided by four pi: an intrinsic scalar, its sign measured against a handedness."""
    step_first = first[..., 1:] - first[..., :-1]                                # [..., segments] Antivector-
    step_second = second[..., 1:] - second[..., :-1]                             # [..., segments] Antivector-
    middle_first = 0.5 * (first[..., 1:] + first[..., :-1])                      # [..., segments] Antivector-
    middle_second = 0.5 * (second[..., 1:] + second[..., :-1])                   # [..., segments] Antivector-
    separation = middle_first[..., :, None] - middle_second[..., None, :]        # [..., segments, segments] Antivector-
    volume = separation & step_first[..., :, None] & step_second[..., None, :]   # [..., segments, segments] Scalar-
    distance = separation.norm()                                                 # [..., segments, segments] Scalar+
    return (volume / distance ** 3).sum(axis=(-2, -1)) / (4 * np.pi)             # [...] Scalar-
