"""Ellipsoids as plane-to-point maps in PGA3D.

A dual quadric maps each tangent plane to its contact point. Its inverse maps
the contact point back to the tangent plane. Moving the whole ellipsoid means
pulling planes into its body frame and pushing the resulting points back out.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
DualQuadric = ga.gatype((Point, Plane))


# --- math ------------------------------------------------------------------
def ellipsoid(semi_axes: Point, center: Point) -> DualQuadric:
    """The ellipsoid with the given semi-axes, as ideal points, about the given center."""
    # Ideal points encode the semi-axes. Their dyads give the directional shape;
    # subtracting the center dyad makes a closed envelope of tangent planes.
    return (semi_axes * (Plane & semi_axes)).sum(axis=0) - center * (Plane & center)


def support_plane(quadric: DualQuadric, normal: Plane, infinity: Plane) -> Plane:
    """Find an ellipsoid's tangent plane facing a given normal.

    quadric maps planes to points; infinity selects the affine chart.
    The normal's offset is discarded, retaining only its orientation.
    """
    normal = normal.normalized()
    center = quadric(infinity)                       # pole of the plane at infinity
    through_center = normal - infinity * ((normal & center) / (infinity & center))

    # Shift the plane until it contains its own pole: tangent & Q(tangent) = 0.
    # Dividing by the center's weight makes the distance independent of Q's scale.
    radius = (-(through_center & quadric(through_center)) / (infinity & center)).square_root()
    return through_center - infinity * radius
