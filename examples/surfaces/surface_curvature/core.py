"""The principal curvatures and lines of curvature of a quadric surface follow from its polarity and its confocal family.

A quadric surface is modelled as a polarity, a symmetric map from points to planes that sends each point to its polar plane, and its surface is where a point lies on its own polar plane, where that polar plane is the tangent plane. Evaluated on ideal directions, the bilinear form of the polarity acts as the surface's Hessian; restricted to the tangent plane on both sides, it gives the second fundamental form up to the gradient length, and its eigenvalues against the Euclidean metric on directions are the principal curvatures. For a central quadric, the lines of curvature are where the surface meets its confocal quadrics, whose parameters at each point are the eigenvalues of the dual quadric on directions minus the point's own offset dyad.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack, where
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                 # Scalar+
# Finite points carry homogeneous weight, and directions are ideal points of weight zero.
Point = ga.gatype.antivector()                              # Antivector-
Plane = ga.gatype.vector()                                  # Vector+
Direction = Point.derive.degenerate                         # Antivector-
# A polarity maps each point to its polar plane.
Quadric = ga.gatype((Plane, Point))                         # Plane+ <- Point-

# The three coordinate planes x = 0, y = 0, z = 0, and the plane at infinity.
axes = mv(Plane.derive.nondegenerate, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # [3] Plane+
w = mv.w                                                    # [] Plane+

# A direction's dual is the plane through the origin normal to it; their inner product is the Euclidean metric.
metric = Direction.dual() | Direction.dual()                # [] Scalar+ <- (Direction-, Direction-)


# --- math -----------------------------------------------------------------------------
def central_quadric(weights: np.ndarray) -> Quadric:
    """Coordinate plane dyads weighted by `weights` (`[3]`) minus the plane-at-infinity dyad form a central quadric (`[] Plane+ <- Point-`)."""
    return (axes * (axes & Point) * weights).sum() - w * (w & Point)                    # [] Plane+ <- Point-


def nearest_root(quadratic: Scalar, linear: Scalar, discriminant: Scalar) -> Scalar:
    """The smaller positive root of `quadratic * t**2 + 2 * linear * t + constant` gives the first ray step (`[...] Scalar+`), or zero."""
    zero = discriminant * 0                                                             # [...] Scalar+
    root = where(discriminant > 0, discriminant, zero).square_root()                    # [...] Scalar+
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = (-linear - root) / quadratic, (-linear + root) / quadratic      # [...] Scalar+, Scalar+
    positive = where(second > 0, second, zero)                                          # [...] Scalar+
    smaller = where((first > 0) & (~(second > 0) | (first < second)), first, positive)  # [...] Scalar+
    return where(smaller.isfinite(), smaller, zero)                                     # [...] Scalar+


def ray_intersections(surface: Quadric, origins: Point, heading: Direction) -> tuple[Point, Scalar]:
    """Each ray from `origins` along `heading` meets `surface` at its nearer positive root (`[...] Point-`) with quadratic discriminant (`[...] Scalar+`)."""
    form = surface(Point) & Point                                                       # [] Scalar+ <- (Point-, Point-)
    # Evaluating the bilinear form on a ray gives the coefficients of a quadratic in the distance along it.
    quadratic = form(heading, heading)                                                  # [...] Scalar+
    linear = form(heading, origins)                                                     # [...] Scalar+
    constant = form(origins, origins)                                                   # [...] Scalar+
    discriminant = linear * linear - quadratic * constant                               # [...] Scalar+
    hits = (origins + heading * nearest_root(quadratic, linear, discriminant)).normalized()  # [...] Point-
    return hits, discriminant


def tangent_pair(values: Scalar) -> Scalar:
    """Dropping the normal eigenvalue nearest zero from `values` (`[..., 3] Scalar+`) leaves the two ordered tangent eigenvalues (`[..., 2] Scalar+`)."""
    zero_index = values.abs().argmin(axis=-1)
    lower = where(zero_index == 0, values[..., 1], values[..., 0])                      # [...] Scalar+
    upper = where(zero_index == 2, values[..., 1], values[..., 2])                      # [...] Scalar+
    return stack([lower, upper], axis=-1)                                               # [..., 2] Scalar+


def principal_curvatures(surface: Quadric, points: Point) -> Scalar:
    """Ordered principal curvatures (`[..., 2] Scalar+`) come from the tangent-restricted Hessian against `metric`."""
    tangent = surface(points)                                                           # [...] Plane+
    # The normal direction drops the tangent plane's homogeneous weight, and the projector removes its component.
    normal = tangent.dual().cast(Direction)                                             # [...] Direction-
    projector = Direction - normal * ((tangent & Direction) / (tangent & normal))       # [...] Direction- <- Direction-
    # Restricting the Hessian to the tangent plane on both sides and dividing by gradient length gives the second form.
    second_form = (surface(Point) & Point)(projector, projector) / metric(normal, normal).square_root()  # [...] Scalar+ <- (Direction-, Direction-)
    return tangent_pair(second_form.eigvalsh(metric))                                   # [..., 2] Scalar+


def confocal_parameters(surface: Quadric, points: Point) -> Scalar:
    """Ordered confocal family parameters (`[..., 2] Scalar+`) come from the dual shape minus each point's offset dyad."""
    dual_surface = surface.inverse()                                                    # [] Point- <- Plane+
    # The pole of the plane at infinity is the quadric's centre, normalized to unit weight.
    pole = dual_surface(w)                                                              # [] Point-
    centre = pole / (w & pole)                                                          # [] Point-
    # Shift each direction's dual plane to pass through the centre, and dualize its pole under the dual quadric.
    central_planes = Direction.dual() - w * (Direction.dual() & centre)                 # [] Plane+ <- Direction-
    shape = dual_surface(central_planes).dual()                                         # [] Plane+ <- Direction-
    # Each point's displacement from the centre dualizes to the plane through the origin normal to that offset.
    offset_planes = (points - centre).dual()                                            # [...] Plane+
    # Subtract the point's offset dyad from the shape so the eigenvalues give the confocal quadrics through the point.
    confocal_form = -(shape - offset_planes * (offset_planes & Direction)) & Direction  # [...] Scalar+ <- (Direction-, Direction-)
    return tangent_pair(confocal_form.eigvalsh(metric))                                 # [..., 2] Scalar+
