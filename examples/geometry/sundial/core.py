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
to the pole all day, so the day's rays sweep a cone about the pole, a quadric of rays, each ray to the plane
that reads it, `across * (across & Point) - (across & sun).squared() * Point.dual_inverse()` with `across` the
plane across the pole, and the dial cuts the cone in the shadow path: a conic. The cone's other nappe holds the rays at the opposite declination, so a summer day and the winter day
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
# Directions, to the sun, the pole and the compass points, and points of the dial: antivectors with
# intrinsic orientation.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Rotor = ga.gatype.rotor()
# A quadric of rays: the planes that read them.
Cone = ga.gatype((Plane, Point))                           # Plane <- Point

east, north, up = mv.basis().dual()                        # [] Point each
west = -east
# The dial: the horizontal plane through the origin.
dial = mv.z                                                # [] Plane


# --- math -----------------------------------------------------------------------------
def turn(start: Point, towards: Point, angle: np.ndarray) -> Rotor:
    """The turn by an angle in the plane of two directions at right angles, from the first towards the
    second: their commutator is the plane they span."""
    return (towards.commutator(start) * (-angle / 2)).exp()


def sky(latitude: float) -> tuple[Point, Point]:
    """The celestial pole, north raised towards up by the latitude, and the celestial equator's highest
    point, a quarter turn further: the sun's place at noon on the equinoxes."""
    return turn(north, up, latitude) >> north, turn(north, up, latitude + np.pi / 2) >> north


def sun(pole: Point, equator: Point, declinations: np.ndarray, hours: np.ndarray) -> Point:
    """The direction to the sun at each declination and hour angle: the equator's highest point turned
    towards the pole by the declination, then about the pole towards the west by the hour angle."""
    noon = turn(equator, pole, declinations) >> equator                       # [...] Point
    return turn(equator, west, hours) >> noon                                 # [...] Point


def shadow(nodus: Point, suns: Point) -> Point:
    """Where the shadow of the nodus falls on the dial, the horizontal plane through the origin: from the
    nodus, away from the sun, as far as the dial."""
    return nodus - suns * ((dial & nodus) / (dial & suns))                     # [...] Point


def cone(pole: Point, sun: Point) -> Cone:
    """The cone of rays about the pole through a direction to the sun: zero on a ray exactly when it meets
    the pole at the sun's angle, or at its supplement, the other nappe."""
    # The plane across the pole reads a ray's height along it: for a unit sun, the sine of its declination.
    across = pole.dual_inverse()                                               # [...] Plane
    # A ray is on the cone when its height squared is the sun's times its own length squared, the metric.
    return across * (across & Point) - (across & sun).squared() * Point.dual_inverse()   # [...] Plane <- Point


def hour_planes(pole: Point, equator: Point, hours: np.ndarray) -> Plane:
    """The plane through the pole and the sun at each hour angle, on every day of the year: their join."""
    return pole & sun(pole, equator, np.zeros_like(hours), hours)             # [...] Plane
