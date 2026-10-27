"""Area transport: edges, oriented areas and volume carried through a deformation of space, the flat
collapse included.

The deformation is a map on points, the identity `Point` plus a squeeze and a slide. Its outermorphism
moves oriented patches by moving their edges, `areas(a & b) == deformation(a) & deformation(b)`: a
patch, the meet of two edges, is a plane through the centre as large as the face, with the face's own
sense. The determinant scales volumes. The adjugate of the area map carries a flux density back so that
it gives the same flux through an undeformed patch as through the deformed one,
`adjugate(flux) & patch == flux & areas(patch)`: defined without an inverse, it exists at zero volume
too, where the flux density it returns gives nothing through every patch along the lost direction, the
patches the area map crushes.

In matrix notation the deformation reads as the deformation gradient F, the area map as its cofactor
matrix, det(F) F^-T for invertible F, as in Nanson's relation, and the map on flux densities as its
adjugate, det(F) F^-1.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points, and edges between them, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
# An oriented patch, the meet of two edges: a plane through the centre as large as its area.
Patch = ga.gatype.vector().derive.flipped_orientation
# A flux density: its pairing with a patch is the flux through it.
Flux = ga.gatype.bivector()
Linear = ga.gatype((Point, Point))                           # Point- <- Point-
AreaMap = ga.gatype((Patch, Patch))                          # Patch- <- Patch-
FluxMap = ga.gatype((Flux, Flux))                            # Flux <- Flux


# --- math -----------------------------------------------------------------------------
def deformation(thickness: np.ndarray, shear: np.ndarray) -> Linear:
    """Squeeze along z while sliding horizontal layers along x: a point's height above the plane z,
    `mv.z & Point`, scaled, and carried along x."""
    return Point + (thickness - 1) * mv.z.dual() * (mv.z & Point) + shear * mv.x.dual() * (mv.z & Point)


def cube(side: float) -> tuple[Point, Point, Patch]:
    """A cube centred at the origin: its corners, its faces' centres, and each face's oriented patch, the
    meet of its two edges, as large as the face, all wound alike around the cube."""
    corners = mv.vector(list(product((-side / 2, side / 2), repeat=3))).dual()   # [vertices] Point-
    outward = concatenate([mv.basis(), -mv.basis()])                           # [faces] Vector+
    centres = outward.dual() * (side / 2)                                      # [faces] Point-
    # The cube's edge along each of its axes; a face's patch is the meet of the two edges it spans.
    edges = mv.basis().dual() * side                                           # [axes] Point-
    meets = edges[[1, 2, 0]] & edges[[2, 0, 1]]                                # [axes] Patch-
    spans = concatenate([meets, meets])                                        # [faces] Patch-
    # Each patch turned so that the volume it spans with its face's centre shares the sense of the cube's
    # own volume, the meet of its three edges: the surface wound consistently.
    sense = edges[0] & edges[1] & edges[2]                                     # [] Scalar-
    facing = (spans & centres) * sense                                         # [faces] Scalar+
    return corners, centres, spans * facing / facing.norm()


def transport(deformation: Linear) -> tuple[AreaMap, FluxMap, Scalar]:
    """The deformation's area map, the adjugate that carries flux densities back through it, and the
    volume factor."""
    # Move the two edges of each oriented patch together.
    areas = deformation.outermorphism(Patch)                                  # [...] Patch- <- Patch-
    # Pull a flux density back through that area map: the flux through every patch is kept.
    adjugate = areas.adjugate()                                               # [...] Flux <- Flux
    return areas, adjugate, deformation.det()
