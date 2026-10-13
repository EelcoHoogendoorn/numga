"""Sundials: the hour lines and the seasons' shadow paths of a horizontal dial, in the geometric algebra of
three-dimensional space.

Space at the dial is east, north and up. The sky turns once a day about the celestial pole, the direction
the earth's axis points, raised above the north by the latitude. The sun rides on that turning at its
declination, its angle from the celestial equator: each day it runs round a circle of the sky at a fixed
angle from the pole, and from day to day the declination swings between the solstices. Every position of
the sun is two turns of a direction on the celestial equator, one towards the pole by the declination and
one about the pole by the hour.

A dial reads the sun by the shadow of a gnomon whose edge, the style, points at the pole. The plane through
the style and the sun at a given hour holds the sun at that hour on every day of the year, so each hour
casts its shadow along one line of the dial, whatever the season. The tip of the gnomon, the nodus, casts
a point whose path over a day is the season's shadow path. The ray from the nodus to the sun keeps its angle
to the pole all day, so the day's rays sweep a cone about the pole, a quadratic form on rays
`pole * (pole | Vector) - sin(declination)**2 * Vector`, and the dial cuts the cone in the shadow path: a
conic. The cone's other nappe holds the rays at the opposite declination, so a summer day and the winter day
as far from the equinox are the two branches of one hyperbola; at the equinoxes the cone flattens into the
plane of the equator, and the shadow runs along a straight line.

In the notation of gnomonics a horizontal dial's hour line makes the angle $\\arctan(\\sin\\varphi \\tan H)$
with the noon line, at latitude $\\varphi$ and hour angle $H$, and the shadow paths are its declination lines.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
# A quadratic form on rays.
Cone = ga.gatype((Vector, Vector))                         # Vector <- Vector

east, north, up = mv.basis()                               # [] Vector each
west = -east


# --- math -----------------------------------------------------------------------------
def turn(plane: Bivector, angle: np.ndarray) -> Rotor:
    """The turn by an angle in a plane through the origin."""
    return (plane * (-angle / 2)).exp()


def sky(latitude: float) -> tuple[Vector, Vector]:
    """The celestial pole, north raised towards up by the latitude, and the celestial equator's highest
    point, a quarter turn further: the sun's place at noon on the equinoxes."""
    return turn(north ^ up, latitude) >> north, turn(north ^ up, latitude + np.pi / 2) >> north


def sun(pole: Vector, equator: Vector, declinations: np.ndarray, hours: np.ndarray) -> Vector:
    """The direction to the sun at each declination and hour angle: the equator's highest point turned
    towards the pole by the declination, then about the pole towards the west by the hour angle."""
    noon = turn(equator ^ pole, declinations) >> equator                      # [...] Vector
    return turn(equator ^ west, hours) >> noon                                # [...] Vector


def shadow(nodus: Vector, suns: Vector) -> Vector:
    """Where the shadow of the nodus falls on the dial, the horizontal plane through the origin: from the
    nodus, away from the sun, as far as the dial."""
    return nodus - suns * ((nodus | up) / (suns | up))                         # [...] Vector


def cone(pole: Vector, declinations: np.ndarray) -> Cone:
    """The cone of rays at a declination's angle from the celestial equator: zero on a ray exactly when its
    angle to the pole is a quarter turn less the declination, or a quarter turn more."""
    return pole * (pole | Vector) - mv.scalar(np.sin(declinations)[..., None] ** 2) * Vector   # [...] Vector <- Vector


def hour_planes(pole: Vector, equator: Vector, hours: np.ndarray) -> Bivector:
    """The plane through the pole and the sun at each hour angle, on every day of the year."""
    return pole ^ sun(pole, equator, np.zeros_like(hours), hours)             # [...] Bivector
