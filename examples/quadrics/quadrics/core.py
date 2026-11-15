"""Ellipsoids as plane-to-point maps in 3D projective geometric algebra.

A dual quadric maps each tangent plane to its contact point, and solving the map against a contact point recovers the tangent plane. Carrying the ellipsoid by a motor pulls planes into the body frame and pushes the resulting points back out.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Motor = ga.gatype.rotor()                           # Rotor+
DualQuadric = ga.gatype((Point, Plane))             # Point- <- Plane+


def ellipsoid(semi_axes: Point, center: Point) -> DualQuadric:
    """Sums the dyads of the ideal `semi_axes` (`[3] Point-`) less the dyad of `center` (`[] Point-`) into a dual quadric (`[] Point- <- Plane+`)."""
    # Ideal points give the directional shape; subtracting the center dyad closes the envelope of tangent planes.
    return (semi_axes * (Plane & semi_axes)).sum(axis=0) - center * (Plane & center)   # [] Point- <- Plane+


def support_plane(quadric: DualQuadric, facing: Plane) -> Plane:
    """Slides the plane facing `facing` (`[...] Plane+`) out from the pole of infinity until it contains its own image under `quadric` (`[...] Plane+`)."""
    normal = facing.normalized()                                                       # [...] Plane+
    # The center of the ellipsoid is the image of the plane at infinity.
    center = quadric(mv.w)                                                             # [] Point-
    through_center = normal - mv.w * ((normal & center) / (mv.w & center))             # [...] Plane+
    # Dividing by the center's weight makes the offset independent of the quadric's overall scale.
    offset = (-(through_center & quadric(through_center)) / (mv.w & center)).square_root()   # [...] Scalar+
    return through_center - mv.w * offset                                              # [...] Plane+
