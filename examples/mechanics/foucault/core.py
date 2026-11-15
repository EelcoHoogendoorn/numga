"""Foucault's pendulum turns its swing plane against the site's floor at the fraction of the earth's
turn projected onto that floor, matching the spherical holonomy around the latitude circle.

Space is fixed to the earth: the planes `x` and `y` meet along the rotation axis, `x` passes
through the prime meridian, and `z` is the equatorial plane. Directions such as a site's vertical,
north, and east are intrinsic antivectors (`Point-`), while the earth turns in the bivector plane
`equator`. A pendulum hung at a site swings along a direction in its floor, and because gravity and
the suspension act along the vertical, nothing turns that swing within the floor while the earth
carries the floor underneath it. Projecting `equator` onto the floor gives the sine of the latitude:
all of the earth's turn lies in the floor at the pole, none lies in it at the equator, and in Paris
the swing turns clockwise seen from above through three quarters of a full turn per day.

Carrying a direction step by step around the latitude circle by the minimal rotor between successive
verticals produces the same daily rotation as a geometric holonomy, equal to a full turn minus the
solid angle of the enclosed polar cap.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
# Spatial directions carry intrinsic antivector orientation.
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# Coordinate planes: x and y meet along the rotation axis, x passes through the prime meridian, and z is the equator.
x, y, z = mv.basis()                                                          # [] Plane+, Plane+, Plane+
# Equatorial plane in which the earth turns.
equator = mv.xy                                                               # [] Bivector+


# --- math -----------------------------------------------------------------------------
def compass(latitudes: np.ndarray) -> tuple[Point, Point, Point]:
    """Local vertical, north, and east (`[...] Point-`) frame each site on the prime meridian at `latitudes`."""
    # Tilt the prime-meridian plane normal toward the north pole by the latitude, and a quarter turn further for north.
    up_turn = (mv.xz * (-latitudes / 2)).exp()                                # [...] Rotor+
    north_turn = (mv.xz * (-(latitudes + np.pi / 2) / 2)).exp()               # [...] Rotor+
    vertical = up_turn >> x.dual()                                            # [...] Point-
    north = north_turn >> x.dual()                                            # [...] Point-
    return vertical, north, y.dual() * np.ones_like(latitudes)                # [...] Point-, Point-, Point-


def floor_share(floors: Bivector) -> Scalar:
    """Orthogonal projection of `equator` onto `floors` measures the fraction (`[...] Scalar+`) of the earth's turn felt in each floor."""
    return (equator | floors) / floors.squared()                              # [...] Scalar+


def swing_direction(latitudes: np.ndarray, earth_turns: np.ndarray) -> Point:
    """Starting along north, each pendulum swing (`[...] Point-`) turns backward within its floor by `floor_share` of `earth_turns`."""
    _, north, east = compass(latitudes)                                       # [...] Point-, Point-, Point-
    # Orthogonal directions anticommute, so their commutator spans the oriented horizontal floor plane.
    floor = east.commutator(north)                                            # [...] Bivector+
    swing_turn = (floor * floor_share(floor) * (earth_turns / 2)).exp()       # [...] Rotor+
    return swing_turn >> north                                                # [...] Point-


def parallel_holonomy(latitudes: np.ndarray, n_steps: int) -> Rotor:
    """Minimal rotors between `n_steps` successive verticals around each latitude circle compose into the parallel-transport holonomy (`[...] Rotor+`)."""
    vertical, _, _ = compass(latitudes)                                       # [...] Point-, Point-, Point-
    # Carry the site's vertical through one full rotation of the earth in n_steps equal increments.
    day = (equator * (-np.pi * np.arange(n_steps + 1)[:, None] / n_steps)).exp()  # [n_steps + 1, ...] Rotor+
    sites = day >> vertical                                                   # [n_steps + 1, ...] Point-
    steps = (1 + sites[1:] * ~sites[:-1]).normalized()                        # [n_steps, ...] Rotor+
    return steps.cumprod(axis=0)[-1]                                          # [...] Rotor+


def turn_angle(rotors: Rotor, floors: Bivector) -> Scalar:
    """Pairing twice the bivector logarithm of `rotors` with `floors` reads the clockwise turn angle (`[...] Scalar+`) in radians."""
    return 2 * (rotors.log() | floors)                                        # [...] Scalar+


def day_states(latitude: float, n_frames: int) -> Iterator[tuple[Rotor, Point]]:
    """Yields `(earth, swing)` (`[] Rotor+`, `[] Point-`) across `n_frames` equal intervals of one sidereal day at `latitude`."""
    _, north, east = compass(np.asarray(latitude))                            # [] Point-, Point-, Point-
    floor = east.commutator(north)                                            # [] Bivector+
    share = floor_share(floor)                                                # [] Scalar+
    for index in range(n_frames + 1):
        angle = 2 * np.pi * index / n_frames
        earth = (equator * (-angle / 2)).exp()                                # [] Rotor+
        swing_turn = (floor * share * (angle / 2)).exp()                      # [] Rotor+
        yield earth, swing_turn >> north
