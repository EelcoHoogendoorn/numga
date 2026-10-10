"""Rational Bézier curves of points, planes and motors in projective geometric algebra, shown in the plane.

A rational Bézier curve is the sum of its control elements, each weighted by its weight and its Bernstein
polynomial. Elements of one type form a linear space, so the same sum blends points, points at infinity,
planes and motors. Planes are the elements dual to points; in the plane they are lines. A quadratic curve
of points traces a conic: a map from points to planes built from the control polygon, zero on the curve,
whose form on directions tells an ellipse from a parabola and a hyperbola. The curve's tangent planes are
a quadratic curve of planes, and they envelope the same conic through its inverse.
"""

from __future__ import annotations

from math import comb

import numpy as np

from numga import Extensor, NumpyContext, stack
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# Points at infinity, and the size of one: the norm of its dual.
Direction = ga.gatype(ga.subspace("x y").complement())
direction_metric = Direction.dual() | Direction.dual()                        # [] Scalar <- (Direction, Direction)
Conic = ga.gatype((Plane, Point))                                             # Plane <- Point


# --- math -----------------------------------------------------------------------------
def bernstein(degree: int, parameters: np.ndarray) -> np.ndarray:
    """The Bernstein polynomials of a degree at each parameter, on a new last axis."""
    index = np.arange(degree + 1)
    binomials = np.array([comb(degree, k) for k in index])
    return binomials * parameters[..., None] ** index * (1 - parameters[..., None]) ** (degree - index)   # [samples, controls]


def blend(controls: Extensor, weights: np.ndarray, parameters: np.ndarray) -> Extensor:
    """The rational Bézier curve of the controls on the last axis: each control weighted by its weight and
    its Bernstein polynomial, summed. Points, planes and motors alike."""
    basis = bernstein(controls.shape[-1] - 1, parameters) * weights[..., None, :]   # [..., samples, controls]
    return (controls[..., None, :] * basis).sum(axis=-1)                      # [..., samples]


def complementary(weights: np.ndarray) -> np.ndarray:
    """The weights that trace the rest of the same curve: both end weights negated, which turns the
    tangents at the ends around."""
    ends = np.isin(np.arange(weights.shape[-1]), [0, weights.shape[-1] - 1])
    return np.where(ends, -weights, weights)


def pair(first: Plane, second: Plane) -> Conic:
    """Two planes as one conic: its form on a point is the product of the two planes' pairings with it."""
    return (first * (second & Point) + second * (first & Point)) / 2           # [...] Plane <- Point


def conic(controls: Point, weights: np.ndarray) -> Conic:
    """The conic a quadratic curve of points traces, from its control polygon: the chord between its ends
    against the two tangents at them, in the proportion the weights set."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    return first * last * pair(start & end, start & end) - 4 * inner**2 * pair(start & middle, middle & end)   # [...] Plane <- Point


def tangents(controls: Point, weights: np.ndarray) -> tuple[Plane, np.ndarray]:
    """The control planes and weights of the tangent planes of a quadratic curve of points, themselves a
    quadratic curve of planes: the tangent at the start, the chord, and the tangent at the end."""
    start, middle, end = controls[..., 0], controls[..., 1], controls[..., 2]
    first, inner, last = weights[..., 0], weights[..., 1], weights[..., 2]
    planes = stack([start & middle, start & end, middle & end], axis=-1)        # [..., 3] Plane
    return planes, np.stack([first * inner, first * last / 2, inner * last], axis=-1)


def at_infinity(shape: Conic) -> Scalar:
    """The eigenvalues of the conic's form on directions: of one sign for an ellipse, which misses the plane
    at infinity; one zero for a parabola, which touches it; of opposite signs for a hyperbola, which
    crosses it twice."""
    return (Direction & shape(Direction)).eigvalsh(direction_metric)            # [..., 2] Scalar


def geodesic(controls: Motor, parameters: np.ndarray) -> Motor:
    """De Casteljau's construction on the motors of the last axis: each level moves along the screw from
    each control towards the next, by the parameter."""
    controls = controls[..., None, :]                                          # [..., 1, controls] Motor
    fractions = parameters[:, None]                                            # [samples, 1]
    while controls.shape[-1] > 1:
        start, end = controls[..., :-1], controls[..., 1:]
        controls = ((end / start).log() * fractions).exp() * start             # [..., samples, level] Motor
    return controls[..., 0]                                                    # [..., samples] Motor


# --- plumbing -------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Points at unit weight from their x and y coordinates."""
    return (mv("x y", coords) + mv.w).dual()


def heading(coords: np.ndarray) -> Point:
    """Points at infinity in the given directions."""
    return mv("x y", coords).dual()
