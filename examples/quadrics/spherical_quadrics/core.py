"""Spherical conics in Cl(3): a quadratic cone through the origin, cut by the unit sphere.

A polarity C maps each point of the sphere to its polar plane, and the conic is where a
point lies on its own polar, P & C(P) == 0. The curve is a spherical ellipse: the sum of
its geodesic distances to two foci is constant. Read through planes instead of points,
the same curve is the envelope of its tangent great circles, the planes with
plane & Q(plane) == 0 for the inverse Q, and the product of the sines of the foci's distances
to those circles is constant. The level sets of P & C(P) are Poinsot's polhodes: the
inertia quadric cut by the momentum sphere.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import Spherical3D

ga = Spherical3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()                         # Scalar+
# Points on yz, zx, xy: the poles of the planes x, y, z.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.plane()                           # Vector+
Rotor = ga.gatype.rotor()                           # Rotor+
# The polarity maps a point to its polar plane, the dual polarity a plane to its pole.
Polarity = ga.gatype((Plane, Point))
DualPolarity = ga.gatype((Point, Plane))


def polarity(eigenvalues: np.ndarray) -> Polarity:
    """The polarity with the given eigenvalues on the basis planes x, y and z is the sum of their weighted dyads."""
    planes = stack([mv.x, mv.y, mv.z])                                         # [3] Plane+
    return (planes * (planes & Point) * eigenvalues).sum(axis=0)


def points(xyz: np.ndarray) -> Point:
    """Points at `(..., 3)` coordinates lie on the basis points yz, zx and xy."""
    return mv(Point, xyz)


def sphere(longitudes: int, latitudes: int) -> Point:
    """A longitude-latitude grid of points covers the unit sphere."""
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, longitudes), np.linspace(0, np.pi, latitudes))
    # The pole xy turned toward x by the polar angle, then about z by the longitude.
    return (mv.xy * (-phi / 2)).exp() * (mv.zx * (-theta / 2)).exp() >> points(np.array([0, 0, 1]))


# --- math -----------------------------------------------------------------------------
def oval(eigenvalues: np.ndarray, t: np.ndarray) -> tuple[Point, Point, float]:
    """The spherical conic of a diagonal polarity with eigenvalues l1 > l2 > 0 > l3 has these points, its two foci,
    and its semi-major arc."""
    l1, l2, l3 = eigenvalues
    # The cone meets the sphere above the ellipse x**2 / x0**2 + y**2 / y0**2 == 1 of the xy plane; its semi-axes
    # as arcs, the semi-minor along x and the semi-major along y, and the focal arc along the major axis y.
    x0 = np.sqrt(-l3 / (l1 - l3))
    y0 = np.sqrt(-l3 / (l2 - l3))
    theta_b = np.arcsin(x0)
    theta_a = np.arcsin(y0)
    theta_c = np.arccos(np.cos(theta_a) / np.cos(theta_b))
    x, y = x0 * np.cos(t), y0 * np.sin(t)
    curve = points(np.stack([x, y, np.sqrt(1.0 - x**2 - y**2)], axis=-1))      # [curve] Point-
    foci = points(np.array([[0.0, np.sin(theta_c), np.cos(theta_c)], [0.0, -np.sin(theta_c), np.cos(theta_c)]]))   # [2] Point-
    return curve, foci, theta_a
