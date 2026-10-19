"""The Hopf fibration: the unit spinors of three-dimensional space, sorted by their spin.

A unit spinor turns the plane xy into another plane through the origin, its spin, and the unit spins form a
sphere. The spinor holds more than its spin: turning it in the xy plane before it acts changes the spinor and
not the plane it turns xy into. The spinors with each spin therefore form a circle, the fibre over that spin;
the fibres fill the three-sphere of unit spinors, and every two of them are linked once. Carried around a
closed loop of spins, a spinor comes back onto its own fibre, turned along it by half the solid angle the loop
encloses.

Spinors are modelled as the even multivectors, and the sandwich `Even >> mv.xy` is a bivector-valued form on
pairs of spinors whose value on one spinor twice is that spinor's spin. Paired with a spin, `spin.dot(HOPF)` is
a scalar form on pairs of spinors, with eigenvalues minus one twice and plus one twice; its top eigenspace is
the plane of the fibre over that spin. A phase is a turn on the right, `psi * (mv.xy * angle).exp()`, which
the spin cannot see because xy commutes with itself. Stereographic projection takes a spinor to its bivector
part over one plus its scalar part, a point of the three-dimensional space of bivectors, where the fibres'
linking number is a sum of joins: its sign is the side the phase turns on, and nothing else.

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
Scalar = ga.gatype.scalar()
# A spinor's spin, the plane it turns xy into, and a point of the space spinors project into: bivectors.
Bivector = ga.gatype.bivector()
Even = ga.gatype.even()
Rotor = ga.gatype.rotor()
# A spinor's spin, as a form on pairs of spinors.
HOPF = Even >> mv.xy                               # [] Bivector <- (Even, Even)


# --- math -----------------------------------------------------------------------------
def fibre_start(spin: Bivector) -> Even:
    """A unit spinor with each spin: the top eigenvector of the form `spin.dot(HOPF)`, whose eigenvalues are
    minus one twice and plus one twice."""
    _, spinors = spin.dot(HOPF).eigh()                                         # [..., 4] Even
    return spinors[..., -1]                                                    # [...] Even


def fibre(start: Even, angles: np.ndarray) -> Even:
    """The spinors with the same spin as each start: the start turned on the right in the xy plane."""
    return start[..., None] * (mv.xy * angles).exp()       # [..., angles] Even


def stereographic(spinor: Even) -> Bivector:
    """The stereographic projection of unit spinors from -1 into the space of bivectors: the bivector part
    over one plus the scalar part."""
    return spinor.select[2] / (1 + spinor.select[0])                           # [...] Bivector


def transport(spins: Bivector) -> Rotor:
    """The rotors that carry a frame along a curve of unit spins, by the smallest rotation from each spin to
    the next: from the first spin to each of the others."""
    steps = (1 + spins[1:] * ~spins[:-1]).normalized()                         # [steps, ...] Rotor
    return steps.cumprod(axis=0)                                               # [steps, ...] Rotor


def linking(first: Bivector, second: Bivector) -> Scalar:
    """Gauss's linking number of pairs of closed polygons, the vertices on the last axis: the join of each
    pair of segments with the line between them, over the cube of its length, summed and divided by four
    pi."""
    step_first = first[..., 1:] - first[..., :-1]                                # [..., segments] Bivector
    step_second = second[..., 1:] - second[..., :-1]                             # [..., segments] Bivector
    middle_first = 0.5 * (first[..., 1:] + first[..., :-1])                      # [..., segments] Bivector
    middle_second = 0.5 * (second[..., 1:] + second[..., :-1])                   # [..., segments] Bivector
    separation = middle_first[..., :, None] - middle_second[..., None, :]        # [..., segments, segments] Bivector
    volume = separation & step_first[..., :, None] & step_second[..., None, :]   # [..., segments, segments] Scalar
    return (volume / separation.scalar_norm_squared().square_root() ** 3).sum(axis=(-2, -1)) / (4 * np.pi)   # [...] Scalar
