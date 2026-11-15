"""Cayley-Klein geometry in PGA2D builds distances, angles, and spheres from one chosen polarity map.

Projective geometry defines joins and meets without distances or angles. Choosing one quadric as a
polarity map from points to polar planes supplies the entire metric: distances and angles follow from
normalized pairings through that absolute and its inverse, perpendiculars to a plane pass through its
pole, and concentric spheres form quadrics of fixed normalized pairing with their centre. Taking the
unit circle as the absolute yields the hyperbolic Beltrami-Klein disc, while flipping the weight sign
yields elliptic geometry from the same constructions.
"""

from __future__ import annotations

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D
from numga.topology import Topology

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Points are intrinsic antivectors and planes are extrinsic vectors with a side, as polars and mirrors
# carry; joining two points gives an intrinsic line, which becomes a sided plane when multiplied by the
# triangle's intrinsic winding scalar.
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Line = ga.gatype.antibivector()                     # Antibivector-
Polarity = ga.gatype((Plane, Point))                # Plane+ <- Point-


def normalized_pairing(pairing: Extensor, first: Extensor, second: Extensor) -> Scalar:
    """Normalized bilinear pairing of `first` and `second` over the root of their self-pairings (`[...] Scalar+`)."""
    return pairing(first, second) / (pairing(first, first) * pairing(second, second)).square_root()


def triangle_metrics(
    topology: Topology, absolute: Polarity, vertices: Point,
) -> tuple[Plane, Scalar, Scalar, Scalar]:
    """Inward side planes (`[3] Plane+[F]`), corner angles (`[3] Scalar+[F]`), opposite side lengths
    (`[3] Scalar+[F]`), and hyperbolic area (`[] Scalar+[F]`) of triangles against `absolute`."""
    pole_map = absolute.solve(Plane)                                       # [] Point- <- Plane+

    # Cayley's hyperbolic distance is arccosh of the negative normalized point pairing, and the corner
    # angle is arccos of the normalized plane pairing through the inverse polarity.
    distance_pairing = -(absolute(Point) & Point)                          # [] Scalar+ <- (Point-, Point-)
    angle_pairing = Plane & pole_map(Plane)                                # [] Scalar+ <- (Plane+, Plane+)

    corners = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    facing = topology.off[2, 0] * vertices                                 # [2, 3] Point-[F]
    # Multiplying each point join by the corner's intrinsic winding turns the intrinsic line into an
    # extrinsic plane oriented toward the interior; the common winding cancels in every angle ratio.
    corner_windings = corners & facing[0] & facing[1]                      # [] Scalar-[F]
    sides = (facing[0] & facing[1]) * corner_windings                      # [3] Plane+[F]
    first_sides = (corners & facing[0]) * corner_windings                  # [3] Plane+[F]
    second_sides = (corners & facing[1]) * corner_windings                 # [3] Plane+[F]
    angles = normalized_pairing(angle_pairing, first_sides, second_sides).clip(-1.0, 1.0).arccos()  # [3] Scalar+[F]
    lengths = normalized_pairing(distance_pairing, facing[0], facing[1]).clip(1.0, np.inf).arccosh()  # [3] Scalar+[F]
    area = np.pi - angles.sum(axis=0)                                      # [] Scalar+[F]
    return sides, angles, lengths, area


def perpendicular_reflection(
    absolute: Polarity, side: Plane, point: Point,
) -> tuple[Point, Line, Point, Point]:
    """Pole of `side`, perpendicular line through `point`, its foot on `side`, and the harmonic
    reflection of `point` across `side` (`[] Point-`, `[] Line-`, `[] Point-`, `[] Point-`)."""
    # Every line perpendicular to a plane passes through its pole, so joining the point to the pole
    # drops the perpendicular, and harmonic homology centred at the pole reflects across the plane.
    pole = absolute.solve(side)                                            # [] Point-
    normal = point & pole                                                  # [] Line-
    foot = side ^ normal                                                   # [] Point-
    homology_step = 2 * (side & point) / (side & pole)                     # [] Scalar+
    reflected_point = point - pole * homology_step                         # [] Point-
    return pole, normal, foot, reflected_point


def concentric_spheres(absolute: Polarity, centres: Point, radii: np.ndarray) -> Polarity:
    """Concentric sphere quadrics (`[..., n_radii] Plane+ <- Point-`) of `radii` (`[n_radii]`) around `centres`."""
    # Clearing the square root in the fixed-distance invariant turns each sphere into a polarity map
    # whose null locus is P & sphere(P) == 0.
    n_radii = len(radii)
    centre = centres[..., None]                                            # [..., 1] Point-
    polar = absolute(centre)                                               # [..., 1] Plane+
    return polar * (polar & Point) - (polar & centre) * np.cosh(radii) ** 2 * absolute  # [..., n_radii] Plane+ <- Point-
