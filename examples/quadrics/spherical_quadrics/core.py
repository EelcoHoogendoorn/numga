"""Spherical conics in 3D spherical geometric algebra as quadratic cones cut by the unit sphere.

A polarity maps each point of the sphere to its polar plane, and the conic is the locus where a
point lies on its own polar, `conic(point) & point == 0`. On the sphere that locus is an ellipse
with constant sum of geodesic distances to two foci. Read through tangent great circles instead
of points, the dual polarity maps planes to points with `plane & dual(plane) == 0`, and the
product of the sines of the foci's distances to the tangent circles is constant. Across the whole
sphere, the level sets of `conic(point) & point` are Poinsot's polhodes.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import Spherical3D

ga = Spherical3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()                         # Scalar+
# Points lie on yz, zx, xy as the poles of the coordinate planes x, y, z.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.plane()                           # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+
# A polarity maps a point to its polar plane; its dual maps a plane to its pole.
Polarity = ga.gatype((Plane, Point))
DualPolarity = ga.gatype((Point, Plane))


# --- math -----------------------------------------------------------------------------
def principal_polarity(eigenvalues: np.ndarray) -> Polarity:
    """Weighted dyads of the coordinate planes form the diagonal polarity (`[] Plane+ <- Point-`)."""
    planes = stack([mv.x, mv.y, mv.z])                                          # [3] Plane+
    return (planes * (planes & Point) * eigenvalues).sum(axis=0)                # [] Plane+ <- Point-


def conic_locus(principal: Polarity, azimuths: np.ndarray) -> tuple[Point, Point, Scalar]:
    """Meridian rays around the negative-eigenvalue pole trace the spherical oval (`[...] Point-`), its two foci (`[2] Point-`), and its semi-major arc (`[] Scalar+`)."""
    pole = mv.z.dual()                                                          # [] Point-
    meridian_turns = (mv.xy * (-azimuths / 2)).exp()                            # [...] Rotor+
    equator = meridian_turns >> mv.x.dual()                                     # [...] Point-
    # Along `pole + equator * slopes`, cross terms vanish on principal axes so the cone cancels the pole and equator forms.
    slopes = (-(principal(pole) & pole) / (principal(equator) & equator)).square_root()  # [...] Scalar+
    oval_points = (pole + equator * slopes).normalized().retyped(Point)         # [...] Point-

    # The minor and major vertices lie toward the x and y poles; their cosine ratio sets the focal arc along the major meridian.
    equator_axes = stack([mv.x, mv.y]).dual()                                   # [2] Point-
    axis_slopes = (-(principal(pole) & pole) / (principal(equator_axes) & equator_axes)).square_root()  # [2] Scalar+
    minor_vertex, major_vertex = (pole + equator_axes * axis_slopes).normalized()  # [] Point-, [] Point-
    semi_major = (pole.reverse() | major_vertex).arccos()                       # [] Scalar+
    focal_arc = ((pole.reverse() | major_vertex) / (pole.reverse() | minor_vertex)).arccos()  # [] Scalar+
    focal_turns = (mv.yz * (stack([focal_arc, -focal_arc]) / 2)).exp()          # [2] Rotor+
    foci = (focal_turns >> pole).retyped(Point)                                 # [2] Point-
    return oval_points, foci, semi_major


# --- plumbing -------------------------------------------------------------------------
def sphere_grid(n_longitudes: int, n_latitudes: int) -> Point:
    """Longitude and latitude turns sweep a `[n_latitudes, n_longitudes] Point-` grid across the unit sphere."""
    longitudes, latitudes = np.meshgrid(
        np.linspace(0, 2 * np.pi, n_longitudes, endpoint=False),
        np.linspace(0, np.pi, n_latitudes),
    )
    latitude_turns = (mv.zx * (-latitudes / 2)).exp()                           # [n_latitudes, n_longitudes] Rotor+
    longitude_turns = (mv.xy * (-longitudes / 2)).exp()                         # [n_latitudes, n_longitudes] Rotor+
    grid_turns = longitude_turns * latitude_turns                               # [n_latitudes, n_longitudes] Rotor+
    return (grid_turns >> mv.z.dual()).retyped(Point)                           # [n_latitudes, n_longitudes] Point-
