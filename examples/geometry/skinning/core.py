"""Linear blend skinning two ways in PGA3D: blend the motors, or blend their maps.

A bone's transform is a motor m; its action on points is the extensor `m >> Point`.
Matrix skinning blends the maps and applies the blend; motor skinning blends the motors,
renormalises, and applies the sandwich. In this library those are the same kind of object,
so each blend is one line and the well-known artefact of the matrix version, the collapsing
radius under a twist, is a one-number comparison.

In the matrix notation of computer graphics the point map reads as a four-by-four matrix. In
dual quaternion notation a motor reads as a unit dual quaternion, and motor skinning as dual
quaternion blending.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D


ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Scalar = ga.gatype.scalar()


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


# --- math -----------------------------------------------------------------------------
def blend_skin(vertices: Point, weight: Scalar, root: Motor, tip: Motor):
    """Deform vertices by normalised motor blending, motor slerp, and map blending."""
    # Blend motors and normalise, or blend their point maps directly.
    motor_skin = (root + weight * (tip - root)).normalized() >> vertices
    matrix_skin = ((root >> Point) + weight * ((tip >> Point) - (root >> Point)))(vertices)

    # Slerp follows the geodesic: the log of the relative motor, scaled by the weight and
    # exponentiated. The normalised lerp is a chord of it, so both are rigid but their twist
    # angles are spaced differently along the blend.
    slerp_skin = ((tip / root).log() * weight).exp() * root >> vertices
    return motor_skin, slerp_skin, matrix_skin


def radius(vertices: Point) -> Scalar:
    """Distance of each vertex from the bone axis, the x axis: the norm of the plane joining them."""
    return (vertices.normalized() & mv.yz).norm()


# --- plumbing -------------------------------------------------------------------------
def cylinder(rings: int, around: int) -> tuple[Point, Scalar]:
    """Unit-radius skin around the x axis for x from 0 to 1, and the weight x of the second bone."""
    x, t = np.meshgrid(np.linspace(0.0, 1.0, rings), np.linspace(0.0, 2 * np.pi, around, endpoint=False), indexing="ij")
    # The skin at y = 1 along x, turned about the x axis by each angle.
    skin = (mv.yz * (-t / 2)).exp() >> point(np.stack([x, np.ones_like(x), np.zeros_like(x)], axis=-1))
    return skin.reshape(-1), mv.scalar(x.reshape(-1, 1))
