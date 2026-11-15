"""Earthquake focal mechanisms represent the moment of a fault slip, its seismic waves, and its beachball diagram in three-dimensional geometric algebra.

Space uses north, east and down axes. A fault orientation is modelled as one rotor composed from three turns by the rake within the fault plane, the dip about the strike, and the strike from north. That rotor carries the level ground onto the fault plane and carries the prime vertical onto the auxiliary plane across the slip, while the metric turns the two planes into the fault normal and slip direction.

A fault slip acts on the surrounding rock as a moment map from directions to reading planes (`Plane+ <- Point-`), formed as `auxiliaries * (faults & Point) + faults * (auxiliaries & Point)`. A single couple `auxiliaries * (faults & Point)` would set the ground turning with curl `(Plane ^ moment(Point)).trace(1, 2)` equal to `faults ^ auxiliaries`, whereas adding the mirrored couple cancels that net torque and makes the moment symmetric between the fault plane and the auxiliary plane.

Along a direction, the first motion of the P wave equals the quadratic form `moment(direction) & direction`, positive for an outward push and negative for an inward pull. Plotting the shaded push regions over an equal-area projection of the lower hemisphere produces the seismological beachball, whose nodal lines trace the fault and auxiliary planes and whose metric eigenvectors give the pressure, null and tension axes.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Directions are intrinsic antivectors; fault and auxiliary planes are extrinsic vectors.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Moment = ga.gatype((Plane, Point))                                            # Plane+ <- Point-

north, east, down = mv.basis().dual()                                         # [] Point-, Point-, Point-
# Level ground faces up and the prime vertical faces north before the fault rotor turns them.
ground, prime_vertical = -mv.z, mv.x                                          # [] Plane+, Plane+
# Reference bivectors orient the horizontal turn from north toward east and the dip turn from east toward down.
horizontal = east.commutator(north)                                           # [] Bivector+
upright = down.commutator(east)                                               # [] Bivector+


# --- math -----------------------------------------------------------------------------
def fault_frame(strikes: np.ndarray, dips: np.ndarray, rakes: np.ndarray) -> Rotor:
    """Three successive turns by `rakes`, `dips` and `strikes` (`[...]`) orient each fault (`[...] Rotor+`), carrying level ground onto the fault plane and north onto the slip direction."""
    # Rake turns north toward -east, dip turns east toward down, and strike turns north toward east.
    rake = (horizontal * (rakes / 2)).exp()                                   # [...] Rotor+
    dip = (upright * (-dips / 2)).exp()                                       # [...] Rotor+
    strike = (horizontal * (-strikes / 2)).exp()                              # [...] Rotor+
    return strike * dip * rake                                                # [...] Rotor+


def double_couple(faults: Plane, auxiliaries: Plane, moments: float) -> Moment:
    """Adding the fault slip couple and its mirror across `faults` and `auxiliaries` (`[...] Plane+`) forms a torque-free seismic moment (`[...] Plane+ <- Point-`)."""
    return moments * (auxiliaries * (faults & Point) + faults * (auxiliaries & Point))   # [...] Plane+ <- Point-


def curl(moment: Moment) -> Bivector:
    """Wedging each output plane of `moment` with its dual input plane traces out the net rotational couple (`[...] Bivector+`), which vanishes for any symmetric moment."""
    # Tracing wedges the output plane against the input plane without invoking the metric.
    return (Plane ^ moment(Point)).trace(1, 2)                                # [...] Bivector+


def radiation(moment: Moment, directions: Point) -> tuple[Scalar, Point]:
    """Evaluating `moment` along unit `directions` (`[...] Point-`) separates the radial P-wave push (`[...] Scalar+`) from the transverse S-wave motion (`[...] Point-`)."""
    reading = moment(directions)                                              # [...] Plane+
    push = reading & directions                                               # [...] Scalar+
    # Dualizing the reading plane converts force traction into motion before projecting out the radial P component.
    return push, reading.dual() - directions * push


def principal_axes(moment: Moment) -> tuple[Scalar, Point]:
    """Diagonalizing `moment(Point) & Point` against the metric yields the pressure, null and tension eigenvalues (`[..., 3] Scalar+`) and unit axes (`[..., 3] Point-`)."""
    return (moment(Point) & Point).eigh(Point.dot(Point))                     # [..., 3] Scalar+, [..., 3] Point-


def fault_planes(moment: Moment) -> tuple[Plane, Plane]:
    """Bisecting the tension and pressure axes of `moment` recovers the two indistinguishable nodal plane pairs (`[..., 2] Plane+`)."""
    # Eigenvalues from `eigh` ascend from pressure through the null axis to tension.
    _, axes = principal_axes(moment)                                          # [..., 3] Point-
    pressure, tension = axes[..., 0], axes[..., 2]                            # [...] Point-, Point-
    # Sum and difference bisect the principal axes at 45 degrees; inverse duality maps normals back to planes.
    first = ((tension + pressure) / np.sqrt(2)).dual_inverse()                # [...] Plane+
    second = ((tension - pressure) / np.sqrt(2)).dual_inverse()               # [...] Plane+
    return stack([first, second], axis=-1), stack([second, first], axis=-1)   # [..., 2] Plane+, [..., 2] Plane+


# --- plumbing -------------------------------------------------------------------------
def sphere_directions(polar: np.ndarray, azimuths: np.ndarray) -> Point:
    """Tilting away from `down` toward `north` by `polar` (`[n_polar]`) and spinning around the vertical by `azimuths` (`[n_azimuths]`) samples a spherical grid of unit directions (`[n_polar, n_azimuths] Point-`)."""
    tilt = (north.commutator(down) * (-polar[:, None] / 2)).exp()             # [n_polar, 1] Rotor+
    spin = (horizontal * (-azimuths[None, :] / 2)).exp()                      # [1, n_azimuths] Rotor+
    return (spin * tilt) >> down                                              # [n_polar, n_azimuths] Point-
