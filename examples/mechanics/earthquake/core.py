"""Earthquake focal mechanisms: the moment of a slip on a fault, the waves it sends out, and the
beachball that seismologists draw of it, in the geometric algebra of three-dimensional space.

Space is north, east and down. A fault's orientation is one rotor, three turns: by the rake within
the fault plane, by the dip about the strike, and by the strike from north. It carries north onto
the slip, the direction the far side moves, and up onto the fault's normal.

A slip on a fault acts on the ground as a moment, a map on directions, `Moment = Vector <- Vector`:
the slip times the normal's component of its argument, plus the normal times the slip's. The first
term alone is a single couple, which would turn the ground; the second, its mirror, cancels the
turning, and the curl of the sum vanishes. The sum is symmetric in the normal and the slip: a fault
and the plane perpendicular to its slip, the auxiliary plane, give the same moment, and no record of
the waves can tell them apart.

Along a direction the P wave's first motion is `direction | moment(direction)`: a push away from the
source where it is positive, a pull where it is negative. Over the lower half of the sphere of
directions, flattened by an equal-area projection, the pushes are shaded: the beachball. Its two
nodal lines are the fault and the auxiliary plane, and the moment's eigenvectors are the pressure
and tension axes, from which the two planes come back.

In the tensor notation of seismology the moment reads as $M_{ij} = M_0\\,(n_i s_j + n_j s_i)$, and
the P wave's amplitude along the unit direction $\\gamma$ as $\\gamma_i M_{ij}\\gamma_j$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Moment = ga.gatype((Vector, Vector))                       # Vector <- Vector

north, east, down = mv.basis()                             # [] Vector each


# --- math -----------------------------------------------------------------------------
def fault_frame(strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray) -> Rotor:
    """The orientation of each fault, from its strike, dip and rake: the turn by the rake in the
    horizontal plane, then the dip about north, then the strike from north towards east. It carries
    north onto the slip and up onto the fault's normal, and the fault dips to the right of its
    strike."""
    rake = ((north ^ east) * (rakes / 2)).exp()                               # [...] Rotor: north towards -east
    dip = ((east ^ down) * (-dips / 2)).exp()                                 # [...] Rotor: east towards down
    strike = ((north ^ east) * (-strikes / 2)).exp()                          # [...] Rotor: north towards east
    return strike * dip * rake                                                # [...] Rotor


def double_couple(normals: Vector, slips: Vector, moments: np.ndarray) -> Moment:
    """The moment of each slip on its fault: the slip's couple across the fault, and the mirrored
    couple that keeps the ground from turning."""
    return moments * (slips * (normals | Vector) + normals * (slips | Vector))   # [...] Moment


def curl(moment: Moment) -> Bivector:
    """How much a moment would turn the ground: zero for any symmetric moment."""
    return (Vector ^ moment(Vector)).contract(1, 2)                           # [...] Bivector


def radiation(moment: Moment, directions: Vector) -> tuple[Scalar, Vector]:
    """Along each direction, the P wave's first motion, positive for a push away from the source,
    and the S wave's, the part of the moment's response across the direction."""
    response = moment(directions)                                             # [...] Vector
    push = directions | response                                              # [...] Scalar
    return push, response - directions * push


def fault_planes(moment: Moment) -> tuple[Vector, Vector]:
    """The two faults with this moment, from its pressure and tension axes: their normals and
    slips `[..., 2]`, each fault's normal the other's slip."""
    _, axes = moment.eigh()                                                   # [..., 3] Vector: pressure, null, tension
    pressure, tension = axes[..., 0], axes[..., 2]                            # [...] Vector each
    first, second = (tension + pressure) / np.sqrt(2), (tension - pressure) / np.sqrt(2)
    return stack([first, second], axis=-1), stack([second, first], axis=-1)   # [..., 2] Vector each


# --- plumbing -------------------------------------------------------------------------
def directions(polar: np.ndarray, azimuths: np.ndarray) -> Vector:
    """Directions on the sphere `[rings, spokes]`, by two turns: away from straight down towards
    north by each polar angle, then about the vertical from north towards east by each azimuth."""
    tilt = ((down ^ north) * (-polar[:, None] / 2)).exp()                     # [rings, 1] Rotor
    spin = ((north ^ east) * (-azimuths[None, :] / 2)).exp()                  # [1, spokes] Rotor
    return (spin * tilt) >> down                                              # [rings, spokes] Vector
