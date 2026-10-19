"""The Hopf fibration: the unit spinors of three-dimensional space, sorted by the direction they point.

A unit spinor turns z into a direction, but it holds more than the direction: turning it in the xy plane
before it acts changes the spinor and not where z ends up. The spinors over each direction therefore form
a circle, the fibre over that direction; the fibres fill the three-sphere of unit spinors, and every two of
them are linked once. Carried around a closed loop of directions, a spinor comes back onto its own fibre,
turned along it by half the solid angle the loop encloses.

Spinors are modelled as the even multivectors, and the sandwich `Even >> mv.z` is a vector-valued form on
pairs of spinors whose value on one spinor twice is that spinor's direction. Paired with a direction,
`direction | HOPF` is a scalar form on pairs of spinors, with eigenvalues minus one twice and plus one
twice; its top eigenspace is the plane of the fibre over that direction. A phase is a turn on the right,
`psi * (mv.xy * angle).exp()`, which the direction cannot see because z commutes with xy.

In quaternion notation a spinor reads as a unit quaternion q and its direction as q k q*; in complex
coordinates, as in Hopf's 1931 paper, the map sends a pair (z1, z2) on the unit sphere of C^2 to the point
z1 / z2 of the Riemann sphere; in bra-ket notation it sends the state of a spin one half to its Bloch
vector, and the turn along the fibre is the Berry phase.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
Rotor = ga.gatype.rotor()
# A spinor's direction, as a form on pairs of spinors.
HOPF = Even >> mv.z                                # [] Vector <- (Even, Even)


# --- math -----------------------------------------------------------------------------
def fibre_start(direction: Vector) -> Even:
    """A unit spinor pointing along each direction: the top eigenvector of the form `direction | HOPF`,
    whose eigenvalues are minus one twice and plus one twice."""
    _, spinors = (direction | HOPF).eigh()                                     # [..., 4] Even
    return spinors[..., -1]                                                    # [...] Even


def fibre(start: Even, angles: np.ndarray) -> Even:
    """The spinors pointing the same way as each start: the start turned on the right in the xy plane."""
    return start[..., None] * (mv.xy * angles).exp()       # [..., angles] Even


def stereographic(spinor: Even) -> Vector:
    """The stereographic projection of unit spinors from -1 into space: the bivector part over one plus
    the scalar part, read as the vector it is dual to."""
    return (spinor.select[2] / (1 + spinor.select[0])).dual()                  # [...] Vector


def transport(directions: Vector) -> Rotor:
    """The rotors that carry a frame along a curve of unit directions, by the smallest rotation from
    each direction to the next: from the first direction to each of the others."""
    steps = (1 + directions[1:] * directions[:-1]).normalized()               # [steps, ...] Rotor
    return steps.cumprod(axis=0)                                               # [steps, ...] Rotor


def linking(first: Vector, second: Vector) -> Scalar:
    """Gauss's linking number of pairs of closed polygons, the vertices on the last axis: the volume each
    pair of segments spans with the line between them, over the cube of its length, summed and divided by
    four pi."""
    step_first = first[..., 1:] - first[..., :-1]                                # [..., segments] Vector
    step_second = second[..., 1:] - second[..., :-1]                             # [..., segments] Vector
    middle_first = 0.5 * (first[..., 1:] + first[..., :-1])                      # [..., segments] Vector
    middle_second = 0.5 * (second[..., 1:] + second[..., :-1])                   # [..., segments] Vector
    separation = middle_first[..., :, None] - middle_second[..., None, :]        # [..., segments, segments] Vector
    volume = (separation ^ step_first[..., :, None] ^ step_second[..., None, :]).dual()   # [..., segments, segments] Scalar
    return (volume / (separation | separation).square_root() ** 3).sum(axis=(-2, -1)) / (4 * np.pi)   # [...] Scalar
