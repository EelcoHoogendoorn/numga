r"""Least-squares fitting of projective primitives to noisy samples minimizes the squared join residual in a single eigenproblem.

A point, a line, and a plane are fitted to noisy samples with the same three steps: joining the samples with the unknown, as a map on its type, squaring and summing that residual into a quadratic form, and selecting its smallest unit eigenvector. The unit constraint uses the scalar part of the unknown's reverse product, which is degenerate on the position components that least squares leaves free. Because the join does not care which of its arguments is the sample and which the unknown, fitting a point to a bundle of lines triangulates their point of closest approach with the same function.

In the matrix notation of least squares the four fits read as the mean of the samples, the leading singular vector of the centred samples, the trailing one, and the solution of $\sum_k (I - \mathbf d_k \mathbf d_k^{\mathsf T})\,\mathbf x = \sum_k (I - \mathbf d_k \mathbf d_k^{\mathsf T})\,\mathbf p_k$ for lines through $\mathbf p_k$ along $\mathbf d_k$.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor, GAType, NumpyContext, stack
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                  # Antivector-
# A line through points is their join, intrinsic like them.
Line = ga.gatype.antibivector()                 # Antibivector-
Plane = ga.gatype.vector()                      # Vector+


# --- math -----------------------------------------------------------------------------
def fit(Unknown: GAType, samples: Extensor) -> Extensor:
    """The primitive of type `Unknown` fitted to `samples` is the flattest direction of their summed squared join residual."""
    samples = samples.normalized()
    # Joining each sample with the unknown measures its incidence error:
    # point & plane is a scalar, point & line a plane, point & point a line.
    residual = samples & Unknown
    misfit = residual.scalar_norm_squared().sum(axis=0)                  # [] Scalar+ <- (Unknown, Unknown)

    # The unknown's own metric, the scalar part of the primitive's reverse product, fixes its
    # geometric size, leaving its position free: plane normal, line direction, or point weight
    # has unit norm. For a line it leaves out the condition to be a line, the pseudoscalar part;
    # the fitted moment is the centroid's moment about the fitted direction, which meets that
    # condition. The metric is blind to the ideal components, so for each mode eigh chooses them
    # to minimize the misfit:
    values, modes = misfit.eigh()                                        # [modes] Scalar+, [modes] Unknown
    return modes[values.argmin()]


# --- plumbing -------------------------------------------------------------------------
def cloud(n: int, spread: float, rng: np.random.Generator) -> Point:
    """The cloud's points are scattered normally about the origin."""
    return mv.point(rng.normal(scale=spread, size=(n, 3)))


def segment(n: int, half_length: float) -> Point:
    """The segment's points are spaced evenly along the y axis."""
    return mv.w.dual() + mv.direction([0, 1, 0]) * np.linspace(-half_length, half_length, n)


def patch(n: int, half_width: float, rng: np.random.Generator) -> Point:
    """The patch's points are scattered uniformly over a square of the plane z == 0."""
    in_plane = mv.direction([[1, 0, 0], [0, 1, 0]])                      # [2] Point-
    return mv.w.dual() + (in_plane * rng.uniform(-half_width, half_width, size=(n, 2))).sum(axis=-1)


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Each point is perturbed by an independent Gaussian translation."""
    noise = rng.normal(scale=sigma, size=(*points.shape, 3))
    translation = (stack((mv.xw, mv.yw, mv.zw)) * noise).sum(axis=-1) / 2
    return translation.exp() >> points


def bundle(n: int, spread: float, rng: np.random.Generator) -> Line:
    """The bundle's lines have unit directions and pass near the origin."""
    directions = mv.direction(rng.normal(size=(n, 3)))
    feet = mv.point(rng.normal(scale=spread, size=(n, 3)))
    # A direction has no weight; its length is the norm of its complement.
    return feet & (directions / directions.dual().norm())


def end_planes(half_length: float) -> Plane:
    """The planes y == half_length and y == -half_length cut a segment out of a line along y."""
    return mv.y - mv.w * np.array([half_length, -half_length])


def patch_edges(half_width: float) -> Line:
    """The four edge lines bound a square of the plane z == 0."""
    x_planes = mv.x - mv.w * (np.array([1, 1, -1, -1]) * half_width)
    y_planes = mv.y - mv.w * (np.array([1, -1, -1, 1]) * half_width)
    return x_planes ^ y_planes
