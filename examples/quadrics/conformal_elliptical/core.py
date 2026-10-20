"""The thirteen planes across the rotation axes of the cube, drawn on the unit sphere.

The cube turns into itself about thirteen axes: three through the centres of opposite faces, six through
the midpoints of opposite edges and four through opposite corners. The plane across each axis, through
the centre, is a vector; a point of the sphere is an antivector, and pairing the two, `plane & point`, is
the point's height above the plane, zero on its great circle. Every rotation of the cube carries the
thirteen planes onto one another. Nine of them, those across the face and edge axes, are also the cube's
mirror planes.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebra import Algebra

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Vector = ga.gatype.vector()
Point = ga.gatype.antivector()

# One direction along each rotation axis of the cube: through its faces, its edges and its corners.
FACE_AXES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
EDGE_AXES = [[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]
CORNER_AXES = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]


# --- math -----------------------------------------------------------------------------
def planes_across_axes() -> Vector:
    """The unit plane across each rotation axis of the cube, through its centre."""
    return mv.vector(FACE_AXES + EDGE_AXES + CORNER_AXES).normalized()                 # [planes] Vector
