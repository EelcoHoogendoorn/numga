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

from numga import Extensor, NumpyContext
from numga.algebras import Spherical3D

ga = Spherical3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
# Points on yz, zx, xy: the poles of the planes x, y, z.
Point = ga.gatype.point()
Plane = ga.gatype.plane()
Rotor = ga.gatype.rotor()
# The polarity maps a point to its polar plane, the dual polarity a plane to its pole.
Polarity = ga.gatype((Plane, Point))             # Plane <- Point
DualPolarity = ga.gatype((Point, Plane))         # Point <- Plane


def polarity(eigenvalues: np.ndarray) -> Polarity:
    """The polarity with the given eigenvalues on the basis planes x, y and z."""
    planes = Extensor.stack([mv.x, mv.y, mv.z])
    return (planes * (planes & Point) * eigenvalues).sum(axis=0)


def points(xyz: np.ndarray) -> Point:
    """Points at (..., 3) coordinates, on the basis points yz, zx and xy."""
    return mv.yz * xyz[..., 0] + mv.zx * xyz[..., 1] + mv.xy * xyz[..., 2]


def sphere(longitudes: int, latitudes: int) -> Point:
    """A longitude-latitude grid of points on the unit sphere."""
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, longitudes), np.linspace(0, np.pi, latitudes))
    return points(np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1))


# --- math -----------------------------------------------------------------------------
def oval(eigenvalues: np.ndarray, t: np.ndarray):
    """Points of the spherical conic of a diagonal polarity with eigenvalues l1 > l2 > 0 > l3,
    its two foci, and its semi-major arc theta_a."""
    l1, l2, l3 = eigenvalues
    # The cone meets the sphere above the ellipse x**2 / x0**2 + y**2 / y0**2 == 1 of the
    # xy-plane; its semi-axes as arcs, the semi-minor along x and the semi-major along y, and
    # the focal arc along the major axis y.
    x0 = np.sqrt(-l3 / (l1 - l3))
    y0 = np.sqrt(-l3 / (l2 - l3))
    theta_b = np.arcsin(x0)
    theta_a = np.arcsin(y0)
    theta_c = np.arccos(np.cos(theta_a) / np.cos(theta_b))
    curve = cone(eigenvalues, np.ones_like(t), t)
    foci = points(np.array([[0.0, np.sin(theta_c), np.cos(theta_c)], [0.0, -np.sin(theta_c), np.cos(theta_c)]]))
    return curve, foci, theta_a


def cone(eigenvalues: np.ndarray, radius: np.ndarray, t: np.ndarray) -> Point:
    """The quadratic cone P & C(P) == 0 of a diagonal polarity, as points at the given radii."""
    l1, l2, l3 = eigenvalues
    x, y = np.sqrt(-l3 / (l1 - l3)) * np.cos(t), np.sqrt(-l3 / (l2 - l3)) * np.sin(t)
    return points(radius[..., None] * np.stack([x, y, np.sqrt(1.0 - x**2 - y**2)], axis=-1))


def geodesic(start: Point, end: Point, t: np.ndarray) -> Point:
    """The great-circle arc from start to end, at fractions t of its length."""
    angle = (-(start | end)).clip(-1.0, 1.0).arccos()
    return (start * ((1.0 - t) * angle).sin() + end * (t * angle).sin()) / angle.sin()


def great_circles(planes: Plane, through: Point, angles: np.ndarray) -> Point:
    """The great circle of each plane, swept from a point on it by rotations about the plane's normal."""
    return (planes.dual() * (angles / 2.0)).exp() >> through
