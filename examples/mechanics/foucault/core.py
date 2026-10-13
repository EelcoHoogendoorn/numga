"""Foucault's pendulum: the plane a pendulum swings in turns against its floor, by the part of the earth's turn
that lies in the floor, in the geometric algebra of three-dimensional space.

Space is the earth's: x and y span the equator, x through the prime meridian, and the axis is z. The earth
turns in the plane `x ^ y`. A pendulum hung at a site swings along a line of its floor, the plane through the
site across the local vertical, and nothing on the pendulum can turn that line within the floor: the floor
turns underneath it. The floor turns with the earth, and the part of the earth's turn that lies in the floor's
own plane, the projection of `x ^ y` onto it, is the sine of the latitude: at the pole all of the earth's turn
lies in the floor, at the equator none. So the swing turns against its floor at that fraction of the earth's
rate, backwards, clockwise from above in the north: once a day at the poles, never at the equator, and in Paris
three quarters of a turn a day.

The same turn is a holonomy. Carry the swing's direction around the latitude circle, step by step, each step
the least turn that takes one floor to the next, and it comes back turned: by the area of the polar cap the
circle encloses on the unit sphere, a full turn less the pendulum's.

In the notation of rotating frames the swing turns at the vertical component of the earth's angular velocity,
$-\\Omega \\sin\\varphi$ at latitude $\\varphi$, and the holonomy is the Gauss–Bonnet angle
$2\\pi(1 - \\sin\\varphi)$ of the cap.
"""

from __future__ import annotations

from functools import reduce

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# The equator's x and y, and the axis z.
x, y, z = mv.basis()                                       # [] Vector
# The plane the earth turns in.
equator = x ^ y                                            # [] Bivector


# --- math -----------------------------------------------------------------------------
def turn(plane: Bivector, angle: np.ndarray) -> Rotor:
    """The turn by an angle in a plane through the origin."""
    return (plane * (-angle / 2)).exp()


def compass(latitudes: np.ndarray) -> tuple[Vector, Vector, Vector]:
    """At a site on the prime meridian at each latitude: its vertical, the direction north along its floor,
    and the direction east."""
    vertical = turn(x ^ z, latitudes) >> x                                    # [...] Vector
    north = turn(x ^ z, latitudes + np.pi / 2) >> x                           # [...] Vector
    return vertical, north, y * np.ones_like(latitudes)                       # [...] Vector each


def felt(floors: Bivector) -> Scalar:
    """How much of the earth's turn lies in each floor's plane: the earth's plane projected onto it, as a
    fraction of the floor's own turn."""
    return equator.scalar_product(floors) / floors.scalar_product(floors)      # [...] Scalar


def swing(latitudes: np.ndarray, earth_turns: np.ndarray) -> Vector:
    """The direction a pendulum at each latitude swings along, on its floor, after the earth has turned by
    each angle: started north, and turned back against the floor by the part of the earth's turn the floor
    feels."""
    _, north, east = compass(latitudes)
    # The floor's plane, north towards east.
    floor = north ^ east                                                      # [...] Bivector
    # The turn back by the felt part of the earth's turn: `turn(floor, -felt * earth_turns)`.
    return (floor * felt(floor) * (earth_turns / 2)).exp() >> north           # [...] Vector


def carried(latitudes: np.ndarray, steps: int) -> Rotor:
    """The turn a direction on the floor makes when carried once around each latitude circle, in so many
    steps, each the least turn taking one site's vertical to the next."""
    vertical, _, _ = compass(latitudes)
    sites = turn(equator, 2 * np.pi * np.arange(steps + 1)[:, None] / steps) >> vertical   # [steps + 1, ...] Vector
    moves = (1 + sites[1:] * sites[:-1]).normalized()                          # [steps, ...] Rotor
    return reduce(lambda total, move: move * total, moves)                    # [...] Rotor
