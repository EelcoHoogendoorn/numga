"""The gnomon of a horizontal sundial casts hour lines that meet at the foot of the style and seasonal shadow paths where the dial plane cuts the sun's daily cone of rays.

Space at the dial is spanned by east, north, and up. The sky turns once a day about the celestial pole, the direction the earth's axis points, raised above the north by the latitude. The sun rides on that turning at its declination, its angle from the celestial equator: each day it runs round a circle of the sky at a fixed angle from the pole, and from day to day the declination swings between the solstices. Every position of the sun is two turns of a direction on the celestial equator, one towards the pole by the declination and one about the pole by the hour.

A dial reads the sun by the shadow of a gnomon whose edge, the style, points at the pole. The plane through the style and the sun at a given hour holds the sun at that hour on every day of the year, so each hour casts its shadow along one line of the dial, whatever the season. The tip of the gnomon, the nodus, casts a point whose path over a day is the season's shadow path. The ray from the nodus to the sun keeps its angle to the pole all day, so the day's rays sweep a cone about the pole, a quadric form from rays to the planes that read them, `across * (across & Point) - (across & sun).squared() * Point.dual_inverse()` with `across` the plane across the pole, and the dial cuts the cone in the shadow path, a conic. The cone's other nappe holds the rays at the opposite declination, so a summer day and the winter day equally far from the equinox trace the two branches of one hyperbola; at the equinoxes the cone flattens into the plane of the equator, and the shadow runs along a straight line.

In the notation of gnomonics a horizontal dial's hour line makes the angle $\\arctan(\\sin\\varphi \\tan H)$ with the noon line at latitude $\\varphi$ and hour angle $H$, and the shadow paths are its declination lines.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Directions to the sun, the pole, and the compass points, and points on the dial are antivectors,
# and planes through the origin are vectors.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# A quadric of rays: the planes that read them.
Cone = ga.gatype((Plane, Point))

east, north, up = mv.basis().dual()                 # [] Point-, Point-, Point-
west = -east                                        # [] Point-
# The horizontal dial plane through the origin.
dial = mv.z                                         # [] Plane+


# --- math -----------------------------------------------------------------------------
def sky(latitude: float) -> tuple[Point, Point]:
    """The celestial pole is north raised towards the zenith by the latitude, and the celestial equator's highest point lies a quarter turn further."""
    # The commutator of two orthogonal unit directions is the bivector of the meridian plane they span.
    meridian = up.commutator(north)                                            # [] Bivector+
    pole_turn = (meridian * (-latitude / 2)).exp()                             # [] Rotor+
    equator_turn = (meridian * (-(latitude + np.pi / 2) / 2)).exp()            # [] Rotor+
    return pole_turn >> north, equator_turn >> north


def sun(pole: Point, equator: Point, declinations: np.ndarray, hour_angles: np.ndarray) -> Point:
    """The direction to the sun is the equator's highest point turned towards the pole by the declination and about the pole towards the west by the hour angle."""
    # The declination turns in the meridian plane, the hour angle in the plane of the celestial equator.
    declination_turns = (pole.commutator(equator) * (-declinations / 2)).exp() # [...] Rotor+
    hour_turns = (west.commutator(equator) * (-hour_angles / 2)).exp()         # [...] Rotor+
    return (hour_turns * declination_turns) >> equator                         # [...] Point-


def shadow(nodus: Point, suns: Point) -> Point:
    """The shadow of the nodus lies on the ray from each sun past the nodus, where its height above the dial vanishes."""
    # Each point's height above the horizontal dial is its pairing with the dial plane.
    return nodus - suns * ((dial & nodus) / (dial & suns))                     # [...] Point-


def cone(pole: Point, sun: Point) -> Cone:
    """The cone of rays about the pole through a sun direction is a quadric whose two nappes hold the rays at the sun's angle to the pole and at its supplement."""
    # The plane across the pole reads a ray's height along it: for a unit sun, the sine of its declination.
    across = pole.dual_inverse()                                               # [...] Plane+
    # A ray lies on the cone when its height squared equals the sun's times its own length squared.
    return across * (across & Point) - (across & sun).squared() * Point.dual_inverse()  # [...] Plane+ <- Point-
