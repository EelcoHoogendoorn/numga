"""Deformations of space carry oriented surface patches forward by outermorphism and pull flux densities back by adjugate, remaining well defined through a flat collapse."""

from __future__ import annotations

from itertools import product

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import PGA3D, VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                  # Scalar+
# Points, and edge displacements between them, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                               # Antivector-
# An oriented patch, the join of two edge displacements, carries its face's attitude, area, and sense.
Patch = ga.gatype.antibivector()                             # Antibivector-
# A flux density pairs with an oriented patch to give the flux through it.
Flux = ga.gatype.bivector()                                  # Bivector+
Linear = ga.gatype((Point, Point))                           # Point- <- Point-
AreaMap = ga.gatype((Patch, Patch))                          # Patch- <- Patch-
FluxMap = ga.gatype((Flux, Flux))                            # Flux+ <- Flux+

pga = NumpyContext(PGA3D).multivector
ProjectivePlane = PGA3D.gatype.vector()                      # Vector+
ProjectivePoint = PGA3D.gatype.antivector()                  # Antivector-
Placement = PGA3D.gatype((ProjectivePoint, ProjectivePoint)) # ProjectivePoint- <- ProjectivePoint-


# --- math -----------------------------------------------------------------------------
def cube(side: float) -> tuple[Point, Point, Patch]:
    """A centred cube of edge length `side` has eight corners (`[8] Point-`), six face centres (`[6] Point-`), and six outward-wound face patches (`[6] Patch-`)."""
    corners = mv.vector(list(product((-side / 2, side / 2), repeat=3))).dual()  # [8] Point-
    axes = stack((mv.x, mv.y, mv.z))                                            # [3] Plane+
    face_planes = concatenate([axes, -axes])                                    # [6] Plane+
    centres = face_planes.dual() * (side / 2)                                   # [6] Point-
    # Join pairs of edge displacements into face patches, then wind every face outward with the cube's volume.
    edges = axes.dual() * side                                                  # [3] Point-
    x_edge, y_edge, z_edge = edges[0], edges[1], edges[2]                       # [] Point-, Point-, Point-
    axis_patches = stack((y_edge & z_edge, z_edge & x_edge, x_edge & y_edge))   # [3] Patch-
    raw_patches = concatenate([axis_patches, axis_patches])                     # [6] Patch-
    volume_sense = x_edge & y_edge & z_edge                                     # [] Scalar-
    orientation = (raw_patches & centres) * volume_sense                        # [6] Scalar+
    patches = raw_patches * (orientation / orientation.norm())                  # [6] Patch-
    return corners, centres, patches


def deformation(thickness: np.ndarray, shear: np.ndarray) -> Linear:
    """Vertical compression by `thickness` and horizontal layer shear by `shear` map point displacements linearly (`[...] Point- <- Point-`)."""
    height = mv.z & Point                                                       # [] Scalar+ <- Point-
    return Point + (thickness - 1) * mv.z.dual() * height + shear * mv.x.dual() * height


def transport(deformation: Linear) -> tuple[AreaMap, FluxMap, Scalar]:
    """The deformation's outermorphism (`[...] Patch- <- Patch-`), its flux-pullback adjugate (`[...] Flux+ <- Flux+`), and its determinant (`[...] Scalar+`) govern area, flux, and volume transport."""
    # Move the two spanning edges of each oriented patch together.
    areas = deformation.outermorphism(Patch)                                    # [...] Patch- <- Patch-
    # Pull flux densities back across the area pairing without requiring an inverse.
    flux_pullback = areas.adjugate()                                            # [...] Flux+ <- Flux+
    return areas, flux_pullback, deformation.det()


def projective_placement(thickness: float, shear: float, translation: float) -> Placement:
    """Vertical compression by `thickness`, shear by `shear`, and translation along `x` by `translation` combine into one projective point map (`[] ProjectivePoint- <- ProjectivePoint-`)."""
    height = pga.z & ProjectivePoint                                            # [] Scalar+ <- ProjectivePoint-
    squeeze = (ProjectivePoint
               + (thickness - 1) * pga.z.dual() * height
               + shear * pga.x.dual() * height)                                 # [] ProjectivePoint- <- ProjectivePoint-
    translator = (pga.xw * (translation / 2)).exp()                             # [] Rotor+
    return translator >> squeeze


def projective_cube(side: float) -> tuple[ProjectivePoint, ProjectivePlane]:
    """A centred projective cube of edge length `side` has eight corner points (`[8] ProjectivePoint-`) bounded by six face planes (`[6] ProjectivePlane+`)."""
    corner_points = pga.point(list(product((-side / 2, side / 2), repeat=3)))   # [8] ProjectivePoint-
    axes = stack((pga.x, pga.y, pga.z))                                         # [3] ProjectivePlane+
    face_planes = concatenate([axes, -axes]) - pga.w * (side / 2)               # [6] ProjectivePlane+
    return corner_points, face_planes
