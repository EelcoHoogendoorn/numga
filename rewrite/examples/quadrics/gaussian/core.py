"""Fit a Gaussian and its 1σ quadric directly from homogeneous point moments.

The inverse second moment gives squared Mahalanobis distance plus one on
unit-weight points. Exponentiating gives the Gaussian; subtracting the weight
dyad twice gives its 1σ contour as a homogeneous zero locus. No origin or
principal-axis frame is supplied. This example draws the construction in PGA2D.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Scalar = ga.gatype.scalar()


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 2) coordinates: the dual of the homogeneous vector."""
    return (mv("x y", coords) + mv.w).dual()


# --- math ------------------------------------------------------------------
def fit_gaussian(points: Point, samples: Point) -> tuple[Scalar, Scalar]:
    """Fit unit-weight points and evaluate the Gaussian and its 1σ level set.

    Samples are unit-weight query points in the same affine chart. The returned
    density has peak one; the level vanishes at Mahalanobis distance one.
    """
    # Leave the plane slot open: each point contributes a rank-one Plane -> Point map.
    # Its mean contains both the cloud's location and its spread, without centering.
    moment = (points * (Plane & points)).mean(axis=0)
    precision = moment.inverse()

    # The inverse moment evaluates to 1 + squared Mahalanobis distance.
    # Removing the constant gives a Gaussian with peak density one.
    squared_distance = (precision(samples) & samples) - 1
    density = (-0.5 * squared_distance).exp()

    # Recover the weight plane from the data: weight & p = 1 for every sample.
    # Its dyad evaluates to that constant squared; subtract twice to make d² = 1
    # the zero locus of a Point -> Plane polarity.
    weight = precision(points.mean(axis=0))
    quadric = precision - 2 * weight * (weight & Point)
    level = quadric(samples) & samples
    return density, level
