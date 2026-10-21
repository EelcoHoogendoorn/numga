"""Solids bounded by quadrics of the conformal model: a Dupin cyclide, balls, and half-spaces.

A point of space is carried as the null vector `up(position)`, held by its dual, and a quadric maps a
point to its polar sphere, `Sphere <- Point`; its pairing with a point, `point & quadric(point)`, is
zero on the surface and negative inside, and at a point of the surface its sphere is the tangent sphere.
In this model the quadrics take in the surfaces Euclidean geometry calls quartic: a torus is one, and a
sphere inversion, a versor, carries it to a Dupin cyclide, whose tube swells on one side and narrows on
the other. A ball and a half-space are quadrics too, the product of a sphere's or a plane's pairing with
that of the point at infinity. A solid is the intersection of a few quadrics, and negating one subtracts it.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+w+e-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Spheres and planes are vectors, and so are positions and directions of flat space, those within x y z.
# Points are the duals of null vectors. Motors move and turn space conformally.
Sphere = ga.gatype.vector()
Euclidean = Sphere.derive.within("x y z")
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# A camera only turns: a rotor of flat space.
Rotation = Motor.derive.within("x y z")
Quadric = ga.gatype((Sphere, Point))                         # Sphere <- Point
# The null vectors of the origin and of the point at infinity.
origin = (mv.w + mv.e) / 2                                   # [] Sphere
infinity = mv.e - mv.w                                       # [] Sphere


# --- math -----------------------------------------------------------------------------
def up(position: Euclidean) -> Point:
    """A position of flat space as a point: the dual of its null vector."""
    return (origin + position + (position | position) / 2 * infinity).dual()


def sphere(centre: Euclidean, radius: float) -> Sphere:
    """The sphere about a centre: positive inside, paired with a point."""
    return origin + centre + ((centre | centre) - radius**2) / 2 * infinity


def ball(surface: Sphere) -> Quadric:
    """The solid inside a sphere, or on a plane's negative side: its pairing times that of the point at
    infinity, symmetrized."""
    return (surface * (infinity & Point) + infinity * (surface & Point)) / 2


def torus(radius: float, tube: float) -> Quadric:
    """The solid torus about the z axis: a core circle of `radius` in the xy plane, and a `tube` about it."""
    core_sphere = origin - infinity * ((radius**2 + tube**2) / 2)
    return core_sphere * (core_sphere & Point) + radius**2 * (mv.z * (mv.z & Point) - tube**2 * infinity * (infinity & Point))


def translator(offset: Euclidean) -> Motor:
    """The motor carrying every point along `offset`."""
    return (-(offset ^ infinity) / 2).exp()


def moved(quadric: Quadric, versor: Motor) -> Quadric:
    """A quadric carried by a versor: each point carried back, measured, and its sphere carried forward."""
    return versor >> quadric(versor << Point)


def cyclide(radius: float, tube: float, inversion: Sphere) -> Quadric:
    """A Dupin cyclide: the solid torus inverted in a sphere."""
    return moved(torus(radius, tube), inversion)


def torus_points(radius: float, tube: float, n_around: int, n_across: int) -> Sphere:
    """Null vectors of points on a torus's surface: a point on the tube's circle turned about the core,
    then about z, `[across, around]`."""
    around = (mv.xy * (-np.linspace(0, 2 * np.pi, n_around, endpoint=False) / 2)).exp()      # [around] Rotation
    across = (mv.xz * (-np.linspace(0, 2 * np.pi, n_across, endpoint=False)[:, None] / 2)).exp()  # [across, 1] Rotation
    positions = around >> (mv.x * radius + (across >> mv.x) * tube)                           # [across, around] Euclidean
    return origin + positions + (positions | positions) / 2 * infinity


def outward(tangent: Sphere, position: Euclidean) -> Euclidean:
    """The outward direction square to a surface at a position, from its tangent sphere there."""
    return tangent.cast(Euclidean) + (tangent | infinity) * position
