"""Conformal circles carry either the intrinsic order of three points or the extrinsic order of two cutting surfaces.

Conformal geometric algebra models spheres and planes as extrinsic vectors and points as null intrinsic antivectors. Joining three ordered points with `&` forms an intrinsic circle (`Path-`), whereas intersecting two oriented surfaces with `^` forms an extrinsic circle (`Circle+`). Even when both circles share identical blade coefficients, reflecting in the circle's own plane preserves the point join while reversing the surface intersection, and pairing the intrinsic circle's dual with the sphere at infinity or a mirror plane selects Euclidean rotation or conformal boost generators along the circle.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import CGA3D

ga = CGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                     # Scalar+
Point = ga.gatype.antivector()                                                  # Antivector-
Sphere = ga.gatype.vector()                                                     # Vector+
Circle = ga.gatype.bivector()                                                   # Bivector+
Path = ga.gatype.antitrivector()                                                # Antitrivector-


# --- math -----------------------------------------------------------------------------
def circle_constructions(points: Point, plane: Sphere, sphere: Sphere) -> tuple[Path, Circle]:
    """Joining three ordered `points` (`[3] Point-`) gives an intrinsic circle (`[] Path-`), while
    wedging `plane` and `sphere` (`[] Sphere+`) gives an extrinsic circle (`[] Circle+`)."""
    point_join = (points[0] & points[1] & points[2]).normalized()               # [] Path-
    surface_meet = (plane ^ sphere).normalized()                                # [] Circle+
    return point_join, surface_meet


def circle_motion_states(circle: Path, horizon: Sphere, reflector: Sphere, grid: Point,
                         phases: np.ndarray, glide_amplitude: float) -> Iterator[Point]:
    """Yields paired disk grids (`[2, n_radii + 1, n_angles] Point-`) carried along `circle` by rigid
    rotation and by a hyperbolic glide reflection across `phases` (`[n_frames]`)."""
    # Contracting the circle's dual with the sphere at infinity selects the Euclidean rotation generator about the circle's axis.
    axis = (circle.dual() | horizon).normalized()                               # [] Circle+
    rotations = (axis * (-phases / 2)).exp()                                    # [...] Rotor+
    # Contracting the circle's dual with the reflecting plane selects the commuting conformal boost along the circle.
    boost = reflector | circle.dual()                                           # [] Circle+
    boost_rotors = (boost * (glide_amplitude * np.sin(phases))).exp()           # [...] Rotor+
    glides = reflector * boost_rotors                                           # [...] Odd+
    for rotation, glide in zip(rotations, glides):
        yield stack([rotation >> grid, glide >> grid])


# --- plumbing -------------------------------------------------------------------------
def sphere_samples(seed: Point, equator: Circle, meridian: Circle,
                   longitudes: np.ndarray, latitudes: np.ndarray) -> Point:
    """Two orthogonal rotations carry `seed` across `latitudes` (`[n_parallels]`) and `longitudes`
    (`[n_meridians]`) into a spherical grid (`[n_parallels, n_meridians] Point-`)."""
    longitude_turns = (equator * (-longitudes[None, :] / 2)).exp()              # [1, ...] Rotor+
    latitude_turns = (meridian * (-latitudes[:, None] / 2)).exp()               # [..., 1] Rotor+
    return (longitude_turns * latitude_turns) >> seed


def disk_samples(direction: Point, origin: Point, infinity: Point, axis: Circle,
                 radii: np.ndarray, angles: np.ndarray) -> Point:
    """Translating `origin` along `direction` by `radii` (`[n_radii + 1]`) and turning about `axis` by
    `angles` (`[n_angles]`) samples a polar disk grid (`[n_radii + 1, n_angles] Point-`)."""
    # Dualizing the join of the direction with the point at infinity gives the translation generator along that direction.
    radial_shifts = ((infinity & direction).dual() * (-radii / 2)).exp()        # [...] Rotor+
    radial_seeds = radial_shifts >> origin                                      # [...] Point-
    angular_turns = (axis * (angles / 2)).exp()                                 # [...] Rotor+
    return angular_turns[None, :] >> radial_seeds[:, None]
