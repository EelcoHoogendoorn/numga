"""A rational quadratic Bézier curve in the plane traces a curve of rays in three-dimensional space whose sweep is the cone of a conic.

A rational Bézier curve sums its control points, each scaled by its weight and its Bernstein polynomial. With the plane modelled at unit height in space, a weighted point sits along its ray from the origin as a three-dimensional antivector, and the rational curve is a plain quadratic Bézier curve of antivectors with nothing divided. Each point stands for its ray, read where the ray meets the unit sphere, so the projective plane appears on the sphere with every point represented twice at opposite ends of a diameter and the line at infinity along the equator. A line of the plane is a plane through the origin formed by the meet `p & q` of two points. The rays of a quadratic curve sweep a cone, the zero set of a symmetric duality map `curve_conic(controls, weights)(point) & point == 0` built from the control triangle, whose two antipodal loops on the sphere miss, touch or cross the equator for an ellipse, a parabola and a hyperbola.
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
Conic = ga.gatype((Plane, Point))                   # Plane+ <- Point-


# --- math -----------------------------------------------------------------------------
def bernstein_basis(n_degree: int, parameters: np.ndarray) -> np.ndarray:
    """The Bernstein polynomials of degree `n_degree` at each parameter form an array of shape `[..., n_degree + 1]`."""
    indices = np.arange(n_degree + 1)
    binomials = np.array([comb(n_degree, index) for index in indices])
    return binomials * parameters[..., None] ** indices * (1 - parameters[..., None]) ** (n_degree - indices)  # [..., n_degree + 1]


def bezier_curve(controls: Point, weights: np.ndarray, parameters: np.ndarray) -> Point:
    """A rational Bézier curve sums its control points weighted by their weights and Bernstein polynomials (`[...] Point-`)."""
    n_degree = controls.shape[-1] - 1
    basis = bernstein_basis(n_degree, parameters) * weights[..., None, :]                # [..., n_degree + 1]
    return (controls[..., None, :] * basis).sum(axis=-1)                                 # [...] Point-


def complementary_weights(weights: np.ndarray) -> np.ndarray:
    """The complementary arc of the same conic negates both end weights, reversing the tangents at the ends."""
    ends = np.isin(np.arange(weights.shape[-1]), [0, weights.shape[-1] - 1])
    return np.where(ends, -weights, weights)


def curve_conic(controls: Point, weights: np.ndarray) -> Conic:
    """The cone swept by a quadratic curve's rays is a symmetric duality map (`[...] Plane+ <- Point-`) built from the three sides of the control triangle."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]      # [...] Point-, Point-, Point-
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    # A line of the plane is a plane through the origin, the meet of two control points.
    chord, before, after = start & end, start & middle, middle & end               # [...] Plane+, Plane+, Plane+
    # The plane through the ends balanced against the symmetrized pair of tangent planes at the ends.
    return first * last * chord * (chord & Point) - 2 * inner**2 * (before * (after & Point) + after * (before & Point))  # [...] Plane+ <- Point-
