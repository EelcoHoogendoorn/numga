r"""A rational quadratic Bézier curve in the plane traces a curve of rays in three-dimensional space whose sweep is the cone of a conic.

A rational Bézier curve is the sum of its control points, each weighted by its weight and its Bernstein polynomial. With the plane modelled as the plane `z = 1` of space, a point `(x, y)` at weight `w` is the antivector `w * (x * mv.x + y * mv.y + mv.z).dual()`, and the sum is a plain quadratic Bézier curve of antivectors. Each point is the line through the origin along it, read where it meets the unit sphere: the projective plane is drawn as a sphere with every point on it twice, at opposite ends of a diameter, and its equator is the line at infinity. A line of the plane is a plane through the origin, the meet of two points. The rays of a quadratic curve sweep a cone, the zero set of a form from points to planes `conic(point) & point == 0` built from the control triangle, whose two opposite loops on the sphere miss, touch or cross the equator for an ellipse, a parabola and a hyperbola.

In the homogeneous notation of computer-aided geometric design the curve reads as $\mathbf C(t) = \sum_i w_i B_i(t)\,\mathbf P_i \big/ \sum_i w_i B_i(t)$, with the Bernstein polynomials $B_i(t) = \binom{n}{i}\, t^i (1-t)^{n-i}$, and the conic as $\mathbf x^\top C\, \mathbf x = 0$ in the coordinates $(x, y, w)$.
"""

from __future__ import annotations

from math import comb

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# A point of the plane is a ray from the origin of space, and a line of the plane is a plane through the origin.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Conic = ga.gatype((Plane, Point))


# --- math -----------------------------------------------------------------------------
def bernstein(degree: int, parameters: np.ndarray) -> np.ndarray:
    """The Bernstein polynomials of the given degree at each parameter lie along a trailing control axis."""
    index = np.arange(degree + 1)
    binomials = np.array([comb(degree, k) for k in index])
    return binomials * parameters[..., None] ** index * (1 - parameters[..., None]) ** (degree - index)  # [samples, controls]


def blend(controls: Point, weights: np.ndarray, parameters: np.ndarray) -> Point:
    """A rational Bézier curve is the sum of its control points weighted by their weights and Bernstein polynomials."""
    basis = bernstein(controls.shape[-1] - 1, parameters) * weights[..., None, :]  # [..., samples, controls]
    return (controls[..., None, :] * basis).sum(axis=-1)                           # [..., samples] Point-


def complementary(weights: np.ndarray) -> np.ndarray:
    """The complementary arc of the same conic has both end weights negated, reversing the tangents at the ends."""
    ends = np.isin(np.arange(weights.shape[-1]), [0, weights.shape[-1] - 1])
    return np.where(ends, -weights, weights)


def conic(controls: Point, weights: np.ndarray) -> Conic:
    """The cone swept by a quadratic curve's rays is a form from rays to the planes of the control triangle that read them."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]      # [...] Point-, Point-, Point-
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    # A line of the plane is a plane through the origin, the meet of two control points.
    chord, before, after = start & end, start & middle, middle & end               # [...] Plane+, Plane+, Plane+
    # The plane through the ends balanced against the symmetrized pair of tangent planes at the ends.
    return first * last * chord * (chord & Point) - 2 * inner**2 * (before * (after & Point) + after * (before & Point))  # [...] Plane+ <- Point-
