"""Least-squares fitting of PGA3D primitives to points, in one pattern.

A point, a line and a plane are fitted to noisy samples with the same three lines: the
join of the samples with the unknown left open, that residual squared and summed into a
quadratic form, and its smallest unit eigenvector. Unit is the unknown's own reverse
product, which is degenerate exactly on the coefficients least squares leaves free. The
roles swap freely: a point fitted to a bundle of lines is their point of closest approach.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor, GAType, NumpyContext
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


def direction(coords: np.ndarray) -> Point:
    """Ideal points: the directions (..., 3), the dual of a weightless vector."""
    return mv("x y z", coords).dual()


# --- math -----------------------------------------------------------------------------
def fit(Unknown: GAType, samples: Extensor) -> Extensor:
    """Fit a PGA primitive to a batch of points or lines by squared join distance.

    Unknown is the point, line or plane slot being solved for.
    """
    samples = samples.normalized()
    # Joining each sample with the open unknown measures its incidence error:
    # point & plane is a scalar, point & line a plane, point & point a line.
    residual = samples & Unknown
    misfit = (residual.reverse() | residual).sum(axis=0)

    # The primitive's reverse product fixes its geometric size, leaving its
    # position free: plane normal, line direction, or point weight has unit norm.
    # The norm is degenerate on ideal components, so those eigenvalues are infinite
    # and the least one belongs to a real, finite mode.
    norm = (mv.rotor() >> Unknown).reverse() | Unknown
    values, modes = misfit.eig(norm)
    return modes[values.real().argmin()].real()


# --- samples ---------------------------------------------------------------------------
def cloud(n: int, spread: float, rng: np.random.Generator) -> Point:
    return point(rng.normal(scale=spread, size=(n, 3)))


def segment(n: int, half_length: float) -> Point:
    """Points along the y axis."""
    return mv.zyx + mv.zxw * np.linspace(-half_length, half_length, n)


def patch(n: int, half_width: float, rng: np.random.Generator) -> Point:
    """Points on a square of the plane z = 0."""
    uv = rng.uniform(-half_width, half_width, size=(n, 2))
    return mv.zyx + mv.yzw * uv[:, 0] + mv.zxw * uv[:, 1]


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Move each point by an independent Gaussian translation."""
    noise = rng.normal(scale=sigma, size=(*points.shape, 3))
    translation = (mv.xw * noise[..., 0] + mv.yw * noise[..., 1] + mv.zw * noise[..., 2]) * 0.5
    return translation.exp() >> points


def bundle(n: int, spread: float, rng: np.random.Generator) -> Line:
    """Lines with unit directions, passing near the origin."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    feet = point(rng.normal(scale=spread, size=(n, 3)))
    return feet & direction(directions)


# --- clipping geometry -----------------------------------------------------------------
def end_planes(half_length: float) -> Plane:
    """The planes y = ±half_length, which cut a segment out of a line along y."""
    return mv.y - mv.w * np.array([half_length, -half_length])


def patch_edges(half_width: float) -> Line:
    """The four edge lines of a square of the plane z = 0."""
    x_planes = mv.x - mv.w * (np.array([1, 1, -1, -1]) * half_width)
    y_planes = mv.y - mv.w * (np.array([1, -1, -1, 1]) * half_width)
    return x_planes ^ y_planes
