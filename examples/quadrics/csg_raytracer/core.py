"""Constructive Solid Geometry on quadrics in PGA3D: ellipsoids, ruled hyperboloids, and clipping planes.

A solid quadric is a map from points to planes, zero on its surface and negative inside.
Bound to a ray, it yields a scalar quadratic in ray distance whose real roots give the entry
and exit points. A ruled hyperboloid of one sheet has signature (+, +, -, -), and contains two
infinite families of straight generator lines. Boolean CSG (intersection, difference) is formed
by interval operations on the ray segment between quadric roots, bounded by clipping half-spaces.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Line = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))
DualQuadric = ga.gatype((Point, Plane))

AXES = stack((mv.x, mv.y, mv.z))


# --- math -----------------------------------------------------------------------------
def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """Solid ellipsoid centered at the origin: sum of squared coordinate plane pairings less infinity."""
    weights = 1.0 / (np.asarray(semi_axes, dtype=float) ** 2)
    return (AXES * (AXES & Point) * weights).sum(axis=-1) - mv.w * (mv.w & Point)


def hyperboloid(semi_axes: np.ndarray) -> Quadric:
    """Solid ruled hyperboloid of one sheet centered at the origin: positive along x and y, negative along z."""
    a, b, c = np.asarray(semi_axes, dtype=float)
    return ((mv.x * (mv.x & Point) / a**2)
            + (mv.y * (mv.y & Point) / b**2)
            - (mv.z * (mv.z & Point) / c**2)
            - mv.w * (mv.w & Point))


def place(body: Quadric, shift_vector: np.ndarray, turn_angle: float = 0.0, turn_axis: str = "xy") -> Quadric:
    """Place a quadric by translating and rotating with a PGA motor."""
    generator = getattr(mv, turn_axis) * (turn_angle / 2)
    translator = (stack((mv.xw, mv.yw, mv.zw)) * (np.asarray(shift_vector, dtype=float) / 2)).sum(axis=-1)
    motor = translator.exp() * generator.exp()
    return motor >> body(motor << Point)


def ruling_lines(semi_axes: np.ndarray, n_rulings: int) -> tuple[Point, Point]:
    """Throat points and direction vectors of the two families of generator lines on a circular hyperboloid."""
    a, _, c = np.asarray(semi_axes, dtype=float)
    angles = np.linspace(0.0, 2 * np.pi, n_rulings, endpoint=False)
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)

    throat_points = (mv.point(np.stack([a * cos_t, a * sin_t, np.zeros_like(angles)], axis=-1)))
    directions_plus = (mv.direction(np.stack([-a * sin_t, a * cos_t, np.full_like(angles, c)], axis=-1)))
    directions_minus = (mv.direction(np.stack([-a * sin_t, a * cos_t, np.full_like(angles, -c)], axis=-1)))

    return throat_points, directions_plus, directions_minus


def evaluate(surface: Quadric, point: Point) -> ga.gatype.scalar():
    """Evaluate quadric form at a point: zero on the boundary, negative inside, positive outside."""
    return surface(point) & point
