"""Solids bounded by quadrics in PGA3D, and the motors that place them and the camera.

A solid quadric is a map from points to planes, `Plane <- Point`, whose pairing with a point,
`quadric(point) & point`, is zero on its surface and negative inside; at a point on the surface the
plane it gives is the tangent plane. A solid is the intersection of a few such quadrics: an ellipsoid,
a ruled hyperboloid between two half-spaces, a bowl that is an ellipsoid less a smaller one, cut by a
plane. A half-space is a quadric too, the product of its plane's pairing with the plane at infinity's,
so the planes that cut and the floor are traced as every other quadric is. Negating a quadric swaps
inside and outside, and subtracts its solid.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points and directions are antivectors, planes are vectors, and a direction is a point at infinity.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))                          # Plane <- Point
origin = mv.point([0, 0, 0])                                 # [] Point
# The planes through the origin square to each axis, and the plane at infinity.
axes = stack((mv.x, mv.y, mv.z))                             # [3] Plane


# --- math -----------------------------------------------------------------------------
def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """A solid ellipsoid about the origin: each axis's squared pairing over its semi-axis squared, less
    the plane at infinity's."""
    return (axes * (axes & Point) / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def hyperboloid(semi_axes: np.ndarray) -> Quadric:
    """A solid ruled hyperboloid of one sheet about the origin, its waist around z: as an ellipsoid, with
    the pairing along z subtracted."""
    signs = np.array([1, 1, -1])
    return (axes * (axes & Point) * signs / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def half_space(plane: Plane) -> Quadric:
    """The points on a plane's negative side, as the quadric that pairs them with the plane and with the
    plane at infinity, symmetrized."""
    return (plane * (mv.w & Point) + mv.w * (plane & Point)) / 2


def placed(quadric: Quadric, motor: Motor) -> Quadric:
    """A quadric moved by a motor: each point moved back, measured, and its plane moved forward."""
    return motor >> quadric(motor << Point)


def translator(offset: Point) -> Motor:
    """The motor carrying every point along a direction, a point at infinity."""
    return ((mv.w ^ offset.dual()) / 2).exp()


def orbit(target: Point, distance: float, elevation: float, azimuth: np.ndarray) -> Motor:
    """A camera's motor, from one at the origin looking along its own -z with y up: backed off by
    `distance`, tilted down to look from `elevation` above the horizon, turned to `azimuth` about the
    vertical, and carried to the target."""
    pitch = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()                          # [] Motor
    turn = (mv.xy * (-azimuth / 2)).exp()                                            # [...] Motor
    return translator(target - origin) * turn * pitch * translator(mv.z.dual() * distance)  # [...] Motor


def ruling_lines(semi_axes: np.ndarray, n_rulings: int) -> tuple[Point, Point, Point]:
    """Points around the waist of a circular hyperboloid, and the directions of the two straight lines
    through each that lie in its surface: the point and its directions turned about z together."""
    radius, _, height = semi_axes
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rulings, endpoint=False) / 2)).exp()   # [rulings] Motor
    waist = turns >> mv.point([radius, 0, 0])                                         # [rulings] Point
    rising = turns >> mv.direction([0, radius, height])                               # [rulings] Point
    falling = turns >> mv.direction([0, radius, -height])                             # [rulings] Point
    return waist, rising, falling


def evaluate(quadric: Quadric, point: Point) -> Scalar:
    """A quadric's pairing with a point: zero on its surface, negative inside."""
    return quadric(point) & point
