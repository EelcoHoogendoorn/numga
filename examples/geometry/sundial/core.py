"""The gnomon of a horizontal sundial casts hour lines that meet at the foot of the style and seasonal shadow paths where the dial plane cuts the sun's daily cone of rays.

Space at the dial is spanned by east, north, and up. The sky turns once a day about the celestial pole, the direction the earth's axis points, raised above the north by the latitude. The sun rides on that turning at its declination, its angle from the celestial equator. Each day it runs round a circle of the sky at a fixed angle from the pole, and from day to day the declination swings between the solstices. Every position of the sun is two turns of the celestial equator's highest point, first towards the pole by the declination and then about the pole by the hour angle.

A dial reads the sun by the shadow of a gnomon whose edge, the style, points at the pole. Because the plane through the style and the sun at a given hour holds the sun at that hour on every day of the year, each hour casts its shadow along one line of the dial in every season. The tip of the gnomon, the nodus, casts a point whose daily trace marks the season's shadow path. As the ray from the nodus to the sun keeps its angle to the pole all day, the day's rays sweep a cone about the pole, modelled as a duality map `across * (across & Point) - (across & sun).squared() * Point.dual_inverse()` with `across = pole.dual_inverse()` the plane perpendicular to the pole, and the dial cuts that cone in a conic. Because the cone's other nappe holds the rays at the opposite declination, a summer day and the winter day equally far from the equinox trace the two branches of one hyperbola, while at the equinoxes the cone flattens into the plane of the equator and the shadow runs along a straight line.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Directions to the sun, the pole, and the compass points, together with points on the dial, are antivectors;
# planes through the origin are vectors.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+
# A quadric cone maps each ray to its polar plane through the origin.
Cone = ga.gatype((Plane, Point))                    # Plane+ <- Point-

east, north, up = mv.basis().dual()                 # [] Point-, Point-, Point-
west = -east                                        # [] Point-
# Horizontal dial plane through the origin.
dial = mv.z                                         # [] Plane+


# --- math -----------------------------------------------------------------------------
def sky(latitude: float) -> tuple[Point, Point]:
    """The celestial pole lies at the latitude above north in the meridian plane, and the celestial equator's highest point lies a quarter turn further (`[] Point-`, `[] Point-`)."""
    # The commutator of two orthogonal unit directions is the bivector of the meridian plane they span.
    meridian = up.commutator(north)                                            # [] Bivector+
    pole_turn = (meridian * (-latitude / 2)).exp()                             # [] Rotor+
    equator_turn = (meridian * (-(latitude + np.pi / 2) / 2)).exp()            # [] Rotor+
    return pole_turn >> north, equator_turn >> north                           # [] Point-, Point-


def sun(pole: Point, equator: Point, declinations: np.ndarray, hour_angles: np.ndarray) -> Point:
    """Two turns of the celestial equator's highest point place the sun at `declinations` towards the pole and `hour_angles` westwards about the pole (`[...] Point-`)."""
    # Declination turns in the meridian plane, and hour angle turns in the plane of the celestial equator.
    declination_turns = (pole.commutator(equator) * (-declinations / 2)).exp() # [...] Rotor+
    hour_turns = (west.commutator(equator) * (-hour_angles / 2)).exp()         # [...] Rotor+
    turns = hour_turns * declination_turns                                     # [...] Rotor+
    return turns >> equator                                                    # [...] Point-


def shadow(nodus: Point, suns: Point) -> Point:
    """Each ray from `suns` past `nodus` meets the horizontal dial plane where its height above the dial vanishes (`[...] Point-`)."""
    # Pairing a point with the horizontal dial plane reads its height above the dial.
    return nodus - suns * ((dial & nodus) / (dial & suns))                     # [...] Point-


def cone(pole: Point, sun: Point) -> Cone:
    """The daily cone of rays about `pole` through `sun` maps each ray to its polar plane, holding both nappes at opposite declinations (`[...] Plane+ <- Point-`)."""
    # The plane perpendicular to the pole reads a unit ray's sine of declination along the pole.
    across = pole.dual_inverse()                                               # [...] Plane+
    # A ray lies on the cone where its height squared along the pole matches the sun's times its length squared.
    return across * (across & Point) - (across & sun).squared() * Point.dual_inverse()  # [...] Plane+ <- Point-


def year_states(
    pole: Point,
    equator: Point,
    declinations: np.ndarray,
    ray_angles: np.ndarray,
    day_angles: np.ndarray,
) -> Iterator[tuple[Point, Point]]:
    """Yields `(day_rays, day_suns)` (`[n_ray_hours] Point-`, `[n_samples] Point-`) for each declination in `declinations`."""
    for declination in declinations:
        yield sun(pole, equator, declination, ray_angles), sun(pole, equator, declination, day_angles)
