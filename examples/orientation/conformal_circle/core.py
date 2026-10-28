"""A conformal circle carries either the order of its points or of its cutting surfaces.

Conformal geometric algebra models spheres and planes as vectors and points as null antivectors. The circle
through three points is their join, intrinsic; the circle where two surfaces cut is their meet, extrinsic. The
two can share coefficients and still reflect differently.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import CGA3D

ga = CGA3D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Sphere = ga.gatype.vector()                                                     # Vector+
Circle = ga.gatype.bivector()                                                   # Bivector+
Path = Circle.derive.flipped_orientation                                        # Bivector-


# --- plumbing -------------------------------------------------------------------------
def sphere_samples(seed: Point, equator: Circle, meridian: Circle,
                   longitudes: np.ndarray, latitudes: np.ndarray) -> Point:
    """A sphere sampled by two rotations of its seed point, with latitude and longitude axes."""
    longitude = (equator * (-longitudes[None, :] / 2)).exp()                    # [1, longitudes] Rotor+
    latitude = (meridian * (-latitudes[:, None] / 2)).exp()                     # [latitudes, 1] Rotor+
    return (longitude * latitude) >> seed


def disk_samples(direction: Point, origin: Point, infinity: Point, rotation: Circle,
                 radii: np.ndarray, angles: np.ndarray) -> Point:
    """A polar grid of points, with radius and angle axes: the origin carried along a direction by each radius,
    then turned by each angle."""
    # The direction joined with the point at infinity, dualized, generates the translation along it.
    shifts = ((infinity & direction).dual() * (-radii / 2)).exp()               # [radii] Rotor+
    seeds = shifts >> origin                                                    # [radii] Point-
    turns = (rotation * (angles / 2)).exp()                                     # [angles] Rotor+
    return turns[None, :] >> seeds[:, None]


# --- math -----------------------------------------------------------------------------
def constructions(points: Point, plane: Sphere, sphere: Sphere) -> tuple[Path, Circle]:
    """The circle through its ordered points and the circle cut by its ordered surfaces."""
    through_points = (points[0] & points[1] & points[2]).normalized()           # [] Path-
    from_surfaces = (plane ^ sphere).normalized()                               # [] Circle+
    return through_points, from_surfaces
