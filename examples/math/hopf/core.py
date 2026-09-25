"""The Hopf fibration: the spinors of three-dimensional space, sorted by the direction they point.

A spinor of the geometric algebra of three-dimensional space is an even multivector, four real
numbers, and the unit spinors form the three-sphere. Sandwiching z with a spinor gives a unit vector,
its direction. With the spinor left open, `Even >> mv.z` is that sandwich as a form with two spinor
slots, a vector from a pair of spinors; on one spinor twice it is that spinor's direction.

Turning a spinor on the right in the xy plane leaves its direction alone, since z commutes with xy:
the spinors that point the same way form a circle, the fibre over that direction. The fibre is also
where the form `direction | HOPF`, a scalar from two spinors, is largest: its eigenvalues are minus
one twice and plus one twice, and the eigenspace of plus one is the plane of the circle.

Through the stereographic projection of the three-sphere from the spinor -1, every fibre is a circle
in space, any two of them linked once, and the fibres over a circle of directions fill a torus.
Carrying a spinor around a closed loop of directions, by the smallest rotation from each direction to
the next, brings it back on its own fibre, turned along it by half the solid angle the loop encloses.

In the notation of quaternions a spinor reads as a unit quaternion and its direction as the quaternion
conjugating k, the map of Hopf's 1931 paper; in matrix notation, as a two-component complex spinor and
its Bloch vector, the state of a spin one half, and the turn along the fibre as its Berry phase.
"""

from __future__ import annotations

from itertools import accumulate

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
Rotor = ga.gatype.rotor()
# A spinor's direction, with the spinor left open in both places it appears.
HOPF = Even >> mv.z                                # [] Vector <- (Even, Even)


# --- math -----------------------------------------------------------------------------
def fibre_start(direction: Vector) -> Even:
    """A unit spinor pointing along each direction: the top eigenvector of the form `direction | HOPF`,
    whose eigenvalues are minus one twice and plus one twice."""
    _, spinors = (direction | HOPF).eigh()                                     # [..., 4] Even
    return spinors[..., -1]                                                    # [...] Even


def fibre(start: Even, angles: np.ndarray) -> Even:
    """The spinors pointing the same way as each start: the start turned on the right in the xy plane."""
    return start[..., None] * (mv.xy * mv.scalar(angles[:, None])).exp()       # [..., angles] Even


def stereographic(spinor: Even) -> Vector:
    """The stereographic projection of unit spinors from -1 into space: the bivector part over one plus
    the scalar part, read as the vector it is dual to."""
    return (spinor.select[2] / (1 + spinor.select[0])).dual()                  # [...] Vector


def transport(directions: Vector) -> Rotor:
    """The rotors that carry a frame along a curve of unit directions, by the smallest rotation from
    each direction to the next: from the first direction to each of the others."""
    steps = (1 + directions[1:] * directions[:-1]).normalized()               # [steps, ...] Rotor
    return stack(list(accumulate(steps, lambda carried, step: step * carried)))  # [steps, ...] Rotor


def linking(first: Vector, second: Vector) -> Scalar:
    """Gauss's linking number of two closed polygons: the volume each pair of segments spans with the
    line between them, over the cube of its length, summed and divided by four pi."""
    step_first = first[1:] - first[:-1]                                          # [n] Vector
    step_second = second[1:] - second[:-1]                                       # [m] Vector
    middle_first = 0.5 * (first[1:] + first[:-1])                                # [n] Vector
    middle_second = 0.5 * (second[1:] + second[:-1])                             # [m] Vector
    separation = middle_first[:, None] - middle_second[None, :]              # [n, m] Vector
    volume = (separation ^ step_first[:, None] ^ step_second[None, :]).dual()       # [n, m] Scalar
    return (volume / (separation | separation).square_root() ** 3).sum() / (4 * np.pi)   # [] Scalar
