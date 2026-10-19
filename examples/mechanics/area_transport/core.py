"""Area transport: edges, oriented areas and volume carried through a deformation of space, the flat
collapse included.

The deformation is a map on vectors, the identity `Vector` plus a squeeze and a slide. Its outermorphism
moves oriented patches by moving their edges, `areas(a ^ b) == deformation(a) ^ deformation(b)`, and
scales volumes by the determinant. The adjugate of the area map carries a direction back so that it
measures the same volume against a patch, `adjugate(normal) & patch == normal & areas(patch)`: defined
without an inverse, it exists at zero volume too, where the normal it returns pairs to nothing with every
patch through the lost direction, the patches the area map crushes. Its adjoint, the same through the
metric, carries the faces' area normals forward.

In matrix notation the deformation reads as the deformation gradient F, the area map as its second
compound matrix, and the map on area normals as the cofactor matrix, det(F) F^-T for invertible F, as in
Nanson's relation.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Linear = ga.gatype((Vector, Vector))                         # Vector <- Vector
AreaMap = ga.gatype((Bivector, Bivector))                     # Bivector <- Bivector


# --- math -----------------------------------------------------------------------------
def deformation(thickness: np.ndarray, shear: np.ndarray) -> Linear:
    """Squeeze along z while sliding horizontal layers along x."""
    return Vector + (thickness - 1) * mv.z * (mv.z | Vector) + shear * mv.x * (mv.z | Vector)


def cube(side: float) -> tuple[Vector, Vector, Bivector]:
    """A cube centred at the origin, its faces oriented outwards: its corners, its faces' centres, and
    each face's oriented patch, the wedge of its two edge directions wound counter-clockwise seen from
    outside, as large as the face."""
    corners = mv.vector(list(product((-side / 2, side / 2), repeat=3)))       # [vertices] Vector
    outward = concatenate([mv.basis(), -mv.basis()])                           # [faces] Vector
    # The faces at +x, +y and +z; the opposite faces are wound the other way.
    facing = stack([mv.y, mv.z, mv.x]) ^ stack([mv.z, mv.x, mv.y])            # [axes] Bivector
    patches = concatenate([facing, -facing]) * side**2                         # [faces] Bivector
    return corners, outward * (side / 2), patches


def transport(deformation: Linear) -> tuple[AreaMap, Linear, Linear, Scalar]:
    """The deformation's area map, the adjugate that carries complementary measurements back through it,
    the adjoint of that adjugate, which carries area normals forward, and the volume factor."""
    # Move the two edges of each oriented patch together.
    areas = deformation.outermorphism(Bivector)                              # [...] Bivector <- Bivector
    # Pull a complementary volume measurement back through that area map.
    adjugate = areas.adjugate()                                               # [...] Vector <- Vector
    # Its metric adjoint pushes area normals forward, with their lengths intact.
    cofactor = adjugate.adjoint()                                             # [...] Vector <- Vector
    return areas, adjugate, cofactor, deformation.det()
