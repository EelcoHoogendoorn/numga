"""Compare two motor fits on the same corresponding PGA3D points.

The centered sandwich alignment gives a Cartesian least-squares rotation;
matching the centroids supplies translation. The one-sided equation
`target * motor - motor * source == 0` fits rotation and translation together, minimizing the residual
against the motor's own metric, followed by motor normalization. That metric
measures the rotor part alone, and with it the one-sided fit reaches the same
least-squares pose as the centered alignment, noisy data included.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# The Euclidean rotation subgroup, fixing the chosen PGA origin.
Rotor = ga.gatype.from_blades("1 yz zx xy")
Vector = ga.gatype.from_blades("x y z")


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


# --- math -----------------------------------------------------------------------------
def fit_motor(source: Point, target: Point) -> Motor:
    """Fit a motor by the coefficient residual of the one-sided equations."""
    source, target = source.normalized(), target.normalized()
    # target * Motor == Motor * source leaves the unknown motor in a single linear slot.
    residual = target * Motor - Motor * source

    # The bulk norm alone would discard translation: the PGA scalar product is degenerate on
    # ideal blades. Adding the weight norm, the bulk norm of the complement, sums the squares
    # of every coefficient, retaining errors in ideal components too.
    bulk = residual.reverse().scalar_product(residual)
    weight = residual.dual().reverse().scalar_product(residual.dual())
    misfit = (bulk + weight).sum(axis=0)
    # Against the motor's own metric, the scalar part of motor * motor.reverse(), only the rotor
    # coefficients are measured: translation carries no unit of its own, so the fit does not depend
    # on where the origin sits or on the scene's scale. The metric is singular on translation, so the
    # general eigenproblem sends those modes to infinity; the least finite mode is real, and
    # normalized() then makes motor * motor.reverse() == 1, its pseudoscalar part included.
    values, motors = misfit.eig()
    values = values.real()
    return motors[values.argmin()].real().normalized()


def fit_rotor(source: Vector, target: Vector) -> Rotor:
    """Maximize sandwich alignment of corresponding Euclidean vectors."""
    alignment = target.scalar_product(Rotor >> source).sum(axis=0)
    values, rotors = alignment.eigh()
    return rotors[values.argmax()].normalized()


def fit_motor_alignment(source: Point, target: Point) -> Motor:
    """Fit a rigid pose by centered alignment and centroid matching."""
    source, target = source.normalized(), target.normalized()
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)

    # Point differences are ideal points. Their duals carry the Euclidean
    # displacements; the spatial slot discards the homogeneous weight component.
    source_vectors = (source - source_mean).dual().select_subspace(Vector.output_subspace)
    target_vectors = (target - target_mean).dual().select_subspace(Vector.output_subspace)
    rotation = fit_rotor(source_vectors, target_vectors)

    # A product of point reflections translates by twice their separation.
    # Its square root carries the rotated source centroid onto the target's.
    translation = (target_mean * (rotation >> source_mean).inverse()).square_root()
    return translation * rotation


# --- samples ---------------------------------------------------------------------------
def cloud(n: int, rng: np.random.Generator) -> Point:
    return point(rng.normal(size=(n, 3)) * [2.0, 1.0, 0.5])


def jitter(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Move each point by an independent Gaussian translation."""
    noise = rng.normal(scale=sigma, size=(*points.shape, 3))
    translation = (mv.xw * noise[..., 0] + mv.yw * noise[..., 1] + mv.zw * noise[..., 2]) * 0.5
    return translation.exp() >> points
