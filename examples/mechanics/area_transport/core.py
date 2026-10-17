"""Edges, oriented areas and volume carried through a deformation of space, the flat collapse
included.

The deformation is a map on vectors, built by leaving the vector open: `Vector + ...`. Its
outermorphism moves oriented patches by moving their edges, `areas(a ^ b) == deformation(a) ^
deformation(b)`, and scales volumes by the determinant. The adjugate of the area map carries a
direction back so that it measures the same volume against a patch, `adjugate(c) & patch == c &
areas(patch)`: defined without an inverse, it exists at zero volume too, where the normal it returns
pairs to nothing with every patch through the lost direction, the patches the area map crushes. Its
adjoint, the same through the metric, carries the faces' area normals forward.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def cube(side: float) -> Surface:
    """A cube centred at the origin, its faces oriented outwards: its corners, its faces' centres, and
    each face's oriented patch, the wedge of its two edge directions wound counter-clockwise seen from
    outside, as large as the face, with its area normal, the dual of the patch."""
    corners = mv.vector(list(product((-side / 2, side / 2), repeat=3)))       # [vertices] Vector
    outward = concatenate([mv.basis(), -mv.basis()])                           # [faces] Vector
    # The faces at +x, +y and +z; the opposite faces are wound the other way.
    facing = stack([mv.y, mv.z, mv.x]) ^ stack([mv.z, mv.x, mv.y])            # [3] Bivector
    patches = concatenate([facing, -facing]) * side**2                         # [faces] Bivector
    return Surface(corners, outward * (side / 2), patches, patches.dual())


def transport(deformation: Linear) -> Transport:
    """The deformation's areas carried forward, and their complementary measurements carried back."""
    # Move the two edges of each oriented patch together.
    areas = deformation.outermorphism(Bivector)                              # [...] AreaMap
    # Pull a complementary volume measurement back through that area map.
    adjugate = areas.adjugate()                                               # [...] Linear
    # Its metric adjoint pushes area normals forward, with their lengths intact.
    cofactor = adjugate.adjoint()                                             # [...] Linear
    return Transport(deformation, areas, adjugate, cofactor, deformation.det())


def carried(surface: Surface, transport: Transport) -> Surface:
    """The vertices, face centres and oriented face patches moved together."""
    return Surface(
        transport.deformation[..., None](surface.vertices),
        transport.deformation[..., None](surface.centres),
        transport.areas[..., None](surface.patches),
        transport.cofactor[..., None](surface.area_normals),
    )


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Transport:
    deformation: Linear
    areas: AreaMap
    adjugate: Linear
    cofactor: Linear
    volume: Scalar


@dataclass(frozen=True)
class Surface:
    vertices: Vector
    centres: Vector
    patches: Bivector
    area_normals: Vector
