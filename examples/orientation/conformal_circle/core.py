"""A conformal circle carries either the order of its points or of its cutting surfaces.

Conformal geometric algebra represents spheres and planes as vectors and points as
antivectors. Their joins and intersections can share coefficients while retaining
different orientations under reflection.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+w+e-")
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Sphere = ga.gatype.vector()
Circle = ga.gatype.bivector()
Path = Circle.derive.flipped_orientation


# --- plumbing -------------------------------------------------------------------------
def circle_samples(seed: Point, rotation: Circle, angles: np.ndarray) -> Point:
    """Points following a circle by rotating its seed point."""
    return (rotation * (angles / 2)).exp() >> seed


def sphere_samples(seed: Point, equator: Circle, meridian: Circle,
                   longitudes: np.ndarray, latitudes: np.ndarray) -> Point:
    """A sphere sampled by two rotations of its seed point."""
    longitude = (equator * (-longitudes[None, :] / 2)).exp()
    latitude = (meridian * (-latitudes[:, None] / 2)).exp()
    return (longitude * latitude) >> seed


def disk_samples(direction: Sphere, origin: Sphere, infinity: Sphere, rotation: Circle,
                 radii: np.ndarray, angles: np.ndarray) -> Point:
    """A polar grid lifted to conformal points, with radius and angle axes."""
    positions = direction * radii
    seeds: Point = (origin + positions + positions.squared() * infinity / 2).dual()
    return circle_samples(seeds[:, None], rotation, angles[None, :])


# --- math -----------------------------------------------------------------------------
def constructions(points: Point, plane: Sphere, sphere: Sphere) -> tuple[Path, Circle]:
    """The circle built from its ordered points and its ordered cutting surfaces."""
    through_points: Path = (points[0] & points[1] & points[2]).normalized()
    from_surfaces: Circle = (plane ^ sphere).normalized()
    return through_points, from_surfaces


def reflected_scene(points: Point, curve: Point, surface: Point, plane: Sphere,
                    sphere: Sphere, mirror: Sphere) -> tuple[Point, Point, Point, Sphere, Sphere]:
    """The defining points and surfaces beside their reflected counterparts."""
    return (stack([points, mirror >> points]), stack([curve, mirror >> curve]),
            stack([surface, mirror >> surface]), stack([plane, mirror >> plane]),
            stack([sphere, mirror >> sphere]))
