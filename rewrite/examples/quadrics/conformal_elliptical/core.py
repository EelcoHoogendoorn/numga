"""The thirteen mirror planes of the octahedral group, drawn on the unit sphere in Cl(3).

A plane through the origin is a vector, its normal. A point of the sphere lies on the
plane's great circle where its inner product with the normal vanishes, so the sign of
that product marks the two hemispheres the circle separates.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebra import Algebra

ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Vector = ga.gatype.vector()


def octahedral_planes() -> Vector:
    """All 13 planes of the octahedral symmetry group: normals along the axes, face and body diagonals."""
    x, y, z = np.array(np.meshgrid(*[[1, 0, -1]] * 3)).reshape(3, -1)[:, :13]
    return (mv.x * x + mv.y * y + mv.z * z).normalized()
