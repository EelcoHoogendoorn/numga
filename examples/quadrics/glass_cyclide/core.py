"""Conformal quadrics in 3D represent spheres, balls, half-spaces, tori, and Dupin cyclides as polarity maps.

A conformal point is a null antivector placed from the origin by a translator, and a quadric maps each point
to its polar sphere (`Sphere+ <- Point-`). Pairing that polar sphere with the point (`quadric(point) & point`)
vanishes on the surface and turns negative inside the solid, while on the surface itself `quadric(point)` is a
tangent sphere. Sphere inversion is an odd versor that carries the solid torus into a Dupin cyclide, and
intersecting or negating quadrics combines and carves solids directly.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.algebras import CGA3D

ga = CGA3D
mv = NumpyContext(ga).multivector
# Euclidean subspace spanned by x, y, and z.
space = Algebra("x+y+z+")
# Spheres and planes are vectors, while planes through the origin lie within x, y, and z.
# Conformal points are null antivectors, flat directions are dual to origin planes, and motors move space conformally.
Sphere = ga.gatype.vector()                                  # Vector+
OriginPlane = Sphere.derive.within(space)                    # Vector+
Point = ga.gatype.antivector()                               # Antivector-
Direction = OriginPlane.derive.dual                          # Antivector-
Motor = ga.gatype.rotor()                                    # Rotor+
Rotation = Motor.derive.within(space)                        # Rotor+
Quadric = ga.gatype((Sphere, Point))                         # Sphere+ <- Point-
# Conformal origin, point at infinity, and horizon plane whose pairing with a point reads its weight.
origin = (mv.w.dual() + mv.e.dual()) / 2                     # [] Point-
infinity = mv.e.dual() - mv.w.dual()                         # [] Point-
horizon = mv.e - mv.w                                        # [] Sphere+


# --- math -----------------------------------------------------------------------------
def translator(offset: Direction) -> Motor:
    """Exponentiating the dual of `infinity & offset` gives the conformal motor (`[...] Motor+`) translating along `offset`."""
    return (-(infinity & offset).dual() / 2).exp()


def sphere(centre: Point, radius: float) -> Sphere:
    """Subtracting half `radius**2 * horizon` from `centre`'s zero-radius sphere forms the sphere (`[] Sphere+`) positive inside."""
    return centre.dual_inverse() - radius**2 / 2 * horizon


def ball(surface: Sphere) -> Quadric:
    """Symmetrizing a sphere or plane with `horizon` forms the quadric (`[] Sphere+ <- Point-`) negative on its interior."""
    return (surface * (horizon & Point) + horizon * (surface & Point)) / 2


def torus(radius: float, tube: float) -> Quadric:
    """The core sphere, the z plane, and `horizon` combine into the solid torus quadric (`[] Sphere+ <- Point-`) of `radius` and `tube`."""
    core_sphere = sphere(origin, np.hypot(radius, tube))                                             # [] Sphere+
    return core_sphere * (core_sphere & Point) + radius**2 * (mv.z * (mv.z & Point) - tube**2 * horizon * (horizon & Point))


def torus_points(radius: float, tube: float, n_around: int, n_across: int) -> Point:
    """Tube and ring rotations composed with radial translators carry `origin` across a torus surface (`[n_across, n_around] Point-`)."""
    ring_turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_around, endpoint=False) / 2)).exp()          # [n_around] Rotation+
    tube_turns = (mv.xz * (-np.linspace(0, 2 * np.pi, n_across, endpoint=False)[:, None] / 2)).exp()  # [n_across, 1] Rotation+
    motors = ring_turns * translator(mv.x.dual() * radius) * tube_turns * translator(mv.x.dual() * tube)  # [n_across, n_around] Motor+
    return motors >> origin                                                                          # [n_across, n_around] Point-


def outward_normal(tangent: Sphere, point: Point) -> Direction:
    """Flattening a tangent sphere against `horizon` at `point` extracts the surface's outward normal direction (`[...] Direction-`)."""
    # Adding the point's zero-radius sphere weighted by tangent | horizon cancels the conformal curvature term at point,
    # leaving the tangent plane whose Euclidean part dualizes to the outward normal direction.
    return (tangent + (tangent | horizon) * point.dual_inverse()).cast(OriginPlane).dual()


def orbit_cameras(
    target: Point, distance: float, elevation: float, azimuths: np.ndarray,
) -> tuple[Rotation, Point]:
    """Azimuth and elevation rotations place orbiting camera poses (`[...] Rotation+`, `[...] Point-`) at `distance` around `target`."""
    pitch = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()                                           # [] Rotation+
    turns = (mv.xy * (-azimuths / 2)).exp() * pitch                                                  # [...] Rotation+
    offsets = (turns >> mv.z.dual()) * distance                                                      # [...] Direction-
    eyes = translator(offsets) >> target                                                             # [...] Point-
    return turns, eyes
