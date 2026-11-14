"""Solids bounded by quadrics in projective 3D space, and the motors that place the solids and camera.

Each solid quadric is modeled as a symmetric duality map from points to planes (`Plane+ <- Point-`)
whose regressive pairing `quadric(point) & point` vanishes on the surface and is negative inside,
with `quadric(point)` giving the outward tangent plane on the boundary. Intersecting quadrics cuts
out composite solids such as a ruled hyperboloid between two end planes or a hollow bowl formed by
subtracting an inner ellipsoid and slicing off the top with a half-space. Because a half-space is
itself the symmetrized product of its bounding plane with the plane at infinity, cutting planes and
the floor trace identically to curved quadrics, while negating a quadric swaps inside and outside.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                  # Scalar+
# Points and directions are antivectors, planes are vectors, and a direction is a point at infinity.
Point = ga.gatype.antivector()                               # Antivector-
Plane = ga.gatype.vector()                                   # Vector+
Motor = ga.gatype.rotor()                                    # Rotor+
Quadric = ga.gatype((Plane, Point))                          # Plane+ <- Point-
origin = mv.point([0, 0, 0])                                 # [] Point-
# Coordinate planes through the origin orthogonal to the three spatial axes.
axes = stack((mv.x, mv.y, mv.z))                             # [3] Plane+


# --- math -----------------------------------------------------------------------------
def ellipsoid(semi_axes: np.ndarray) -> Quadric:
    """An origin-centred solid ellipsoid scales each coordinate-plane pairing inversely by the
    squared `semi_axes` (`[3]`) and subtracts the ideal-plane pairing (`[] Plane+ <- Point-`)."""
    return (axes * (axes & Point) / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def hyperboloid(semi_axes: np.ndarray) -> Quadric:
    """An origin-centred ruled hyperboloid of one sheet around the vertical axis subtracts the axial
    pairing from the transverse coordinate-plane pairings (`[] Plane+ <- Point-`)."""
    signs = np.array([1, 1, -1])
    return (axes * (axes & Point) * signs / semi_axes**2).sum(axis=-1) - mv.w * (mv.w & Point)


def half_space(plane: Plane) -> Quadric:
    """Symmetrizing `plane` with the plane at infinity embeds `plane`'s negative half-space as a
    rank-two solid quadric (`[] Plane+ <- Point-`)."""
    return (plane * (mv.w & Point) + mv.w * (plane & Point)) / 2


def ruling_lines(semi_axes: np.ndarray, n_rulings: int) -> tuple[Point, Point, Point]:
    """Rotating one waist point and its two surface-tangent directions evenly around the vertical
    axis generates both straight-line ruling families (`[n_rulings] Point-`) of a circular
    hyperboloid."""
    radius, _, height = semi_axes
    angles = np.linspace(0, 2 * np.pi, n_rulings, endpoint=False)
    turns = (mv.xy * (-angles / 2)).exp()                                             # [n_rulings] Motor+
    waist = turns >> mv.point([radius, 0, 0])                                         # [n_rulings] Point-
    rising = turns >> mv.direction([0, radius, height])                               # [n_rulings] Point-
    falling = turns >> mv.direction([0, radius, -height])                             # [n_rulings] Point-
    return waist, rising, falling


def camera_orbit(target: Point, distance: float, elevation: float, azimuth: np.ndarray) -> Motor:
    """Backing a camera off along its viewing axis by `distance`, pitching to `elevation`, turning
    by `azimuth`, and translating to `target` forms its pose motor (`[...] Motor+`)."""
    back_off = (mv.zw * (distance / 2)).exp()                                         # [] Motor+
    pitch = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()                            # [] Motor+
    turn = (mv.xy * (-azimuth / 2)).exp()                                             # [...] Motor+
    # Translate from the origin to `target` along the ideal point joining their difference.
    translation = ((mv.w ^ (target - origin).dual()) / 2).exp()                       # [] Motor+
    return translation * turn * pitch * back_off
