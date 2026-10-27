"""Foucault's pendulum: the plane a pendulum swings in turns against its floor, by the part of the earth's turn
that lies in the floor, in the geometric algebra of three-dimensional space.

Space is the earth's: the planes x and y meet along its axis, x through the prime meridian, and the
plane z is the equator's. Directions, a site's vertical and its north and east, are antivectors. The earth
turns in the plane `x ^ y`. A pendulum hung at a site swings along a line of its floor, the plane through the
site across the local vertical, and nothing on the pendulum can turn that line within the floor: the floor
turns underneath it. The floor turns with the earth, and the part of the earth's turn that lies in the floor's
own plane, the projection of `x ^ y` onto it, is the sine of the latitude: at the pole all of the earth's turn
lies in the floor, at the equator none. So the swing turns against its floor at that fraction of the earth's
rate, backwards, clockwise from above in the north: once a day at the poles, never at the equator, and in Paris
three quarters of a turn a day.

The same turn is a holonomy. The swing's direction, carried around the latitude circle step by step, each
step the least turn that takes one floor to the next, comes back turned: by the area of the polar cap the
circle encloses on the unit sphere, a full turn less the pendulum's.

In the notation of rotating frames the swing turns at the vertical component of the earth's angular velocity,
$-\\Omega \\sin\\varphi$ at latitude $\\varphi$, and the holonomy is the Gauss–Bonnet angle
$2\\pi(1 - \\sin\\varphi)$ of the cap.
"""

from __future__ import annotations


import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Directions are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# The planes x and y, through the axis, and z, the equator's.
x, y, z = mv.basis()                                       # [] Plane+
# The plane the earth turns in.
equator = mv.xy                                            # [] Bivector+


# --- math -----------------------------------------------------------------------------
def compass(latitudes: np.ndarray) -> tuple[Point, Point, Point]:
    """At a site on the prime meridian at each latitude: its vertical, the direction north along its floor,
    and the direction east."""
    # The point along x turned towards z by the latitude, and a quarter turn further.
    up_turn = (mv.xz * (-latitudes / 2)).exp()                                # [...] Rotor+
    north_turn = (mv.xz * (-(latitudes + np.pi / 2) / 2)).exp()               # [...] Rotor+
    vertical = up_turn >> x.dual()                                            # [...] Point-
    north = north_turn >> x.dual()                                            # [...] Point-
    return vertical, north, y.dual() * np.ones_like(latitudes)                # [...] Point-


def felt(floors: Bivector) -> Scalar:
    """How much of the earth's turn lies in each floor's plane: the earth's plane projected onto it, as a
    fraction of the floor's own turn."""
    return (equator | floors) / floors.squared()                       # [...] Scalar+


def swing(latitudes: np.ndarray, earth_turns: np.ndarray) -> Point:
    """The direction a pendulum at each latitude swings along, on its floor, after the earth has turned by
    each angle: started north, and turned back against the floor by the part of the earth's turn the floor
    feels."""
    _, north, east = compass(latitudes)
    # The floor's plane, north towards east: the commutator of two directions is the plane they span.
    floor = east.commutator(north)                                            # [...] Bivector+
    # The turn back in the floor's plane by the felt part of the earth's turn.
    return (floor * felt(floor) * (earth_turns / 2)).exp() >> north           # [...] Point-


def carried(latitudes: np.ndarray, steps: int) -> Rotor:
    """The turn a direction on the floor makes when carried once around each latitude circle, in so many
    steps, each the least turn taking one site's vertical to the next."""
    vertical, _, _ = compass(latitudes)
    # The earth turned through a day, in so many steps, and the site's vertical carried with it.
    day = (equator * (-np.pi * np.arange(steps + 1)[:, None] / steps)).exp()       # [steps + 1, ...] Rotor+
    sites = day >> vertical                                                   # [steps + 1, ...] Point-
    moves = (1 + sites[1:] * ~sites[:-1]).normalized()                         # [steps, ...] Rotor+
    # Each move multiplies on the left, as in a running product.
    return moves.cumprod(axis=0)[-1]                                           # [...] Rotor+
