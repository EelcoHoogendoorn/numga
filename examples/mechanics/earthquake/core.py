"""Earthquake focal mechanisms: the moment of a slip on a fault, the waves it sends out, and the
beachball that seismologists draw of it, in the geometric algebra of three-dimensional space.

Space is north, east and down. A fault's orientation is one rotor, three turns: by the rake within
the fault plane, by the dip about the strike, and by the strike from north. It carries the level ground
onto the fault plane, and the prime vertical, the upright plane facing north, onto the auxiliary plane,
the plane across the slip; the metric turns the two into the fault's normal and its slip, the direction
the far side moves.

A slip on a fault acts on the ground as a moment, a map from directions to the planes that read them,
`Moment = Plane <- Point`: the auxiliary plane weighted by a direction's reading by the fault,
`auxiliary * (fault & Point)`, plus the fault weighted by its reading by the auxiliary plane. The first
term alone is a single couple, which would turn the ground; the second, its mirror, cancels the turning,
and the curl of the sum, `(Plane ^ moment(Point)).trace(1, 2)`, vanishes. The sum is symmetric in the
fault and the auxiliary plane: the two give the same moment, and no record of the waves can tell them
apart.

Along a direction the P wave's first motion is the moment's reading of it, `moment(direction) & direction`:
a push away from the source where it is positive, a pull where it is negative. Over the lower half of
the sphere of directions, flattened by an equal-area projection, the pushes are shaded: the beachball.
Its two nodal lines are the fault and the auxiliary plane, and the eigenvectors of the moment's reading
against the metric are the pressure and tension axes, from which the two planes come back.

In the tensor notation of seismology the moment reads as $M_{ij} = M_0\\,(n_i s_j + n_j s_i)$, and
the P wave's amplitude along the unit direction $\\gamma$ as $\\gamma_i M_{ij}\\gamma_j$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Directions are antivectors with intrinsic orientation; faults are planes, with extrinsic orientation.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Moment = ga.gatype((Plane, Point))                         # Plane <- Point

north, east, down = mv.basis().dual()                      # [] Point each
# The level ground, facing up, and the prime vertical, the upright plane facing north: a fault's frame
# turns them onto the fault plane and the auxiliary plane.
ground, prime_vertical = -mv.z, mv.x                       # [] Plane each
# The planes the fault's frame turns in, from one direction towards another: their commutator.
horizontal = east.commutator(north)                        # [] Bivector: north towards east
upright = down.commutator(east)                            # [] Bivector: east towards down


# --- math -----------------------------------------------------------------------------
def fault_frame(strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray) -> Rotor:
    """The orientation of each fault, from its strike, dip and rake: the turn by the rake in the
    horizontal plane, then the dip about north, then the strike from north towards east. It carries
    north onto the slip and up onto the fault's normal, so the prime vertical onto the auxiliary plane
    and the level ground onto the fault plane, and the fault dips to the right of its strike."""
    rake = (horizontal * (rakes / 2)).exp()                                   # [...] Rotor: north towards -east
    dip = (upright * (-dips / 2)).exp()                                       # [...] Rotor: east towards down
    strike = (horizontal * (-strikes / 2)).exp()                              # [...] Rotor: north towards east
    return strike * dip * rake                                                # [...] Rotor


def double_couple(faults: Plane, auxiliaries: Plane, moments: np.ndarray) -> Moment:
    """The moment of each slip, from its fault plane and its auxiliary plane, across the slip: the
    slip's couple across the fault, and the mirrored couple that keeps the ground from turning."""
    return moments * (auxiliaries * (faults & Point) + faults * (auxiliaries & Point))   # [...] Moment


def curl(moment: Moment) -> Bivector:
    """How much a moment would turn the ground: zero for any symmetric moment."""
    # Each plane the moment gives, wedged with the plane that reads its direction: the trace pairs the
    # open plane with the moment's input, plane against direction, with no metric between them.
    return (Plane ^ moment(Point)).trace(1, 2)                               # [...] Bivector


def radiation(moment: Moment, directions: Point) -> tuple[Scalar, Point]:
    """Along each direction, the P wave's first motion, positive for a push away from the source,
    and the S wave's, the part of the moment's response across the direction."""
    reading = moment(directions)                                              # [...] Plane
    push = reading & directions                                               # [...] Scalar
    # The metric turns the moment's reading into the motion it drives.
    return push, reading.dual() - directions * push


def principal_axes(moment: Moment) -> tuple[Scalar, Point]:
    """The principal values and axes, from pressure through the null axis to tension: the eigenvectors
    of the moment's reading of directions, against the metric, the dot of the planes across them."""
    return (moment(Point) & Point).eigh(Point.dual().dot(Point.dual()))      # [..., 3] Scalar, [..., 3] Point


def fault_planes(moment: Moment) -> tuple[Plane, Plane]:
    """The two faults with this moment, from its pressure and tension axes: their fault and auxiliary
    planes `[..., 2]`, each fault's auxiliary plane the other's fault."""
    _, axes = principal_axes(moment)                                          # [..., 3] Point: pressure, null, tension
    pressure, tension = axes[..., 0], axes[..., 2]                            # [...] Point each
    # The two planes' normals, which the metric turns into the planes across them.
    first = ((tension + pressure) / np.sqrt(2)).dual_inverse()               # [...] Plane
    second = ((tension - pressure) / np.sqrt(2)).dual_inverse()              # [...] Plane
    return stack([first, second], axis=-1), stack([second, first], axis=-1)   # [..., 2] Plane each


# --- plumbing -------------------------------------------------------------------------
def directions(polar: np.ndarray, azimuths: np.ndarray) -> Point:
    """Directions on the sphere `[rings, spokes]`, by two turns: away from straight down towards
    north by each polar angle, then about the vertical from north towards east by each azimuth."""
    tilt = (north.commutator(down) * (-polar[:, None] / 2)).exp()             # [rings, 1] Rotor
    spin = (horizontal * (-azimuths[None, :] / 2)).exp()                      # [1, spokes] Rotor
    return (spin * tilt) >> down                                              # [rings, spokes] Point
