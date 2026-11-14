"""Homogeneous point moments determine a Gaussian density and its one-sigma quadric.

Averaging the outer product of unit-weight points with themselves records location and directional spread in a single moment map without centring. Inverting that moment map yields the precision polarity, whose quadratic form gives one plus the squared Mahalanobis distance and shifts by the recovered weight plane to the one-sigma contour.
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
Polarity = ga.gatype((Plane, Point))                # Plane+ <- Point-


def gaussian_fit(points: Point) -> tuple[Polarity, Polarity]:
    """Homogeneous point moments determine the precision polarity (`[] Plane+ <- Point-`) and one-sigma quadric (`[] Plane+ <- Point-`) of a point cloud."""
    # Each point contributes its rank-one dyad from planes to points; their mean holds both location and spread without centring.
    moment = (points * (Plane & points)).mean(axis=0)                          # [] Point- <- Plane+
    precision = moment.inverse()                                               # [] Plane+ <- Point-
    # Pairing the recovered weight plane with any unit-weight point gives 1, so subtracting its dyad twice puts the zero locus at Mahalanobis distance 1.
    weight = precision(points.mean(axis=0))                                    # [] Plane+
    return precision, precision - 2 * weight * (weight & Point)


def gaussian_density(precision: Polarity, samples: Point) -> Scalar:
    """A precision polarity evaluates on unit-weight sample points (`[...] Point-`) to the peak-normalized Gaussian density (`[...] Scalar+`)."""
    squared_distance = (precision(samples) & samples) - 1                      # [...] Scalar+
    return (-0.5 * squared_distance).exp()
