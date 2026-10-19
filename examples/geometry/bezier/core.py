"""The cone of a rational Bézier curve: rational quadratic Bézier curves as curves of rays, and the cone of
their conic.

A rational Bézier curve is the sum of its control points, each weighted by its weight and its Bernstein
polynomial. With the plane as the plane z = 1 of space, a point (x, y) at weight w is the antivector
`w * (x * mv.x + y * mv.y + mv.z).dual()`, and the sum is a plain quadratic Bézier curve of antivectors. A point
is the line through the origin along it, read where it meets the unit sphere: the projective plane as a sphere,
every point on it twice, at opposite ends of a diameter, its equator the line at infinity. A line of the plane is
a plane through the origin, the meet of two points. The rays of a quadratic curve sweep a cone: the zero set of a
form from points to planes, `point & conic(point) == 0`, built from the control triangle, whose two opposite loops
on the sphere miss, touch or cross the equator for an ellipse, a parabola and a hyperbola.
"""

from __future__ import annotations

from math import comb

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# A point of the plane, a ray from the origin of space, and a line of the plane, a plane through the origin.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Conic = ga.gatype((Plane, Point))                                              # Plane <- Point


# --- math -----------------------------------------------------------------------------
def bernstein(degree: int, parameters: np.ndarray) -> np.ndarray:
    """The Bernstein polynomials of a degree at each parameter, on a new last axis."""
    index = np.arange(degree + 1)
    binomials = np.array([comb(degree, k) for k in index])
    return binomials * parameters[..., None] ** index * (1 - parameters[..., None]) ** (degree - index)   # [samples, controls]


def blend(controls: Point, weights: np.ndarray, parameters: np.ndarray) -> Point:
    """The rational Bézier curve of the controls on the last axis: each control weighted by its weight and
    its Bernstein polynomial, summed."""
    basis = bernstein(controls.shape[-1] - 1, parameters) * weights[..., None, :]   # [..., samples, controls]
    return (controls[..., None, :] * basis).sum(axis=-1)                      # [..., samples] Point


def complementary(weights: np.ndarray) -> np.ndarray:
    """The weights that trace the rest of the same curve: both end weights negated, which turns the
    tangents at the ends around."""
    ends = np.isin(np.arange(weights.shape[-1]), [0, weights.shape[-1] - 1])
    return np.where(ends, -weights, weights)


def pair(first: Plane, second: Plane) -> Conic:
    """Two planes through the origin as one form: on a point, each plane weighted by the point's pairing with
    the other, so that its pairing with the point is the product of the point's pairings with the two."""
    return (first * (second & Point) + second * (first & Point)) / 2           # [...] Plane <- Point


def conic(controls: Point, weights: np.ndarray) -> Conic:
    """The form whose zero set is the cone a quadratic curve's rays sweep, from its control triangle: the
    plane through the origin and the ends against the two through the origin and an end's tangent, in the
    proportion the weights set."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    chord, before, after = start & end, start & middle, middle & end           # [...] Plane each
    return first * last * pair(chord, chord) - 4 * inner**2 * pair(before, after)   # [...] Plane <- Point


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Points (x, y) at unit weight, in the plane z = 1: the duals of `x * mv.x + y * mv.y + mv.z`."""
    return (mv.x * coords[..., 0] + mv.y * coords[..., 1] + mv.z).dual()
