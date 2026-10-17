"""Rational quadratic Bézier curves as curves of vectors, and the cone of their conic.

A rational Bézier curve is the sum of its control points, each weighted by its weight and its Bernstein
polynomial. With the weight as a third coordinate, a point (x, y) at weight w is the vector
`w * (x * e1 + y * e2 + e3)`, and the sum is a plain Bézier curve of vectors. A point is the line through the
origin along its vector, read where it meets the unit sphere: the projective plane as a sphere, every point on it
twice, at opposite ends of a diameter, its equator the line at infinity. The rays of a quadratic curve sweep a
cone: the zero set of a quadratic form on vectors, built from the control triangle, whose two opposite loops on
the sphere miss, touch or cross the equator for an ellipse, a parabola and a hyperbola.
"""

from __future__ import annotations

from math import comb

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Conic = ga.gatype((Vector, Vector))                                            # Vector <- Vector


# --- math -----------------------------------------------------------------------------
def bernstein(degree: int, parameters: np.ndarray) -> np.ndarray:
    """The Bernstein polynomials of a degree at each parameter, on a new last axis."""
    index = np.arange(degree + 1)
    binomials = np.array([comb(degree, k) for k in index])
    return binomials * parameters[..., None] ** index * (1 - parameters[..., None]) ** (degree - index)   # [samples, controls]


def blend(controls: Vector, weights: np.ndarray, parameters: np.ndarray) -> Vector:
    """The rational Bézier curve of the controls on the last axis: each control weighted by its weight and
    its Bernstein polynomial, summed."""
    basis = bernstein(controls.shape[-1] - 1, parameters) * weights[..., None, :]   # [..., samples, controls]
    return (controls[..., None, :] * basis).sum(axis=-1)                      # [..., samples] Vector


def complementary(weights: np.ndarray) -> np.ndarray:
    """The weights that trace the rest of the same curve: both end weights negated, which turns the
    tangents at the ends around."""
    ends = np.isin(np.arange(weights.shape[-1]), [0, weights.shape[-1] - 1])
    return np.where(ends, -weights, weights)


def pair(first: Vector, second: Vector) -> Conic:
    """The normals of two planes through the origin as one form: on a vector, the product of its inner
    products with the two."""
    return (first * (second | Vector) + second * (first | Vector)) / 2         # [...] Vector <- Vector


def conic(controls: Vector, weights: np.ndarray) -> Conic:
    """The form whose zero set is the cone a quadratic curve's rays sweep, from its control triangle: the
    plane through the origin and the ends against the two through the origin and an end's tangent, in the
    proportion the weights set."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    chord, before, after = (start ^ end).dual(), (start ^ middle).dual(), (middle ^ end).dual()   # [...] Vector each
    return first * last * pair(chord, chord) - 4 * inner**2 * pair(before, after)   # [...] Vector <- Vector


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Vector:
    """Points (x, y) at unit weight: the vectors `x * e1 + y * e2 + e3`."""
    return mv(Vector.derive.within("x y"), coords) + mv.z
