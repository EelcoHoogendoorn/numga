"""A Gaussian and its 1σ quadric from homogeneous point moments.

The inverse second moment gives squared Mahalanobis distance plus one on unit-weight points. Exponentiating gives
the Gaussian; subtracting the weight dyad twice gives its 1σ contour as a homogeneous zero locus. No origin or
principal-axis frame is supplied. This example draws the construction in PGA2D.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Scalar = ga.gatype.scalar()                         # Scalar+
# A map from points to planes: a precision, or a quadric.
Polarity = ga.gatype((Plane, Point))


# --- math -----------------------------------------------------------------------------
def fit_gaussian(points: Point) -> tuple[Polarity, Polarity]:
    """The precision of unit-weight points comes with their 1σ ellipsoid, a quadric: both maps from points to planes."""
    # Each point contributes its dyad, a rank-one map from planes to points; their mean holds both the cloud's
    # location and its spread, without centring.
    moment = (points * (Plane & points)).mean(axis=0)                          # [] Point- <- Plane+
    precision = moment.inverse()                                               # [] Plane+ <- Point-
    # The plane of unit weight, recovered from the data: weight & point == 1 for every unit-weight point. Its dyad
    # evaluates to that constant squared; subtracted twice, it makes Mahalanobis distance one the zero locus.
    weight = precision(points.mean(axis=0))                                    # [] Plane+
    return precision, precision - 2 * weight * (weight & Point)


def density(precision: Polarity, samples: Point) -> Scalar:
    """The Gaussian at unit-weight points has peak one: the precision evaluates to one plus the squared Mahalanobis
    distance, and the Gaussian is the exponential of minus half of the rest."""
    squared_distance = (precision(samples) & samples) - 1                      # [...] Scalar+
    return (-0.5 * squared_distance).exp()
