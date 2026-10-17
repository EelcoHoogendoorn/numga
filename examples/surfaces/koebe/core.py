"""Koebe polyhedra: every triangulation of the sphere as a polyhedron whose edges all touch the unit sphere.

The algebra has the unit sphere as its metric: x, y and z square to one and w to minus one. A point x of the
sphere is the null vector `x + w`, and a circle on the sphere, where the plane `n | x == s` cuts it, is the
vector `n + s * w`, scaled to square to one: a point lies on the circle exactly when its inner product with it
is zero. Two circles touch from outside exactly when their inner product is minus one, and then their sum is
the null vector of the point where they touch. Read as a point, a circle's vector is the apex of the cone that
touches the sphere along it, so circles touching along every edge of a triangulation are the vertices of a
polyhedron whose edges all touch the sphere. The circle through the three touching points of a face is
orthogonal to its vertices' circles, the dual of their wedge, and its apex is a vertex of the dual
polyhedron, which touches the sphere at the same points. The equations fix the circles up to the maps that
keep the sphere, the rotors of the algebra; the centred one has the touching points balanced about the
centre.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+w-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# The unit volume that no motion changes: an oriented volume over it reads as its signed size.
unit_volume = mv.antiscalar([1.0])
Rotor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def caps(directions: Vector, radii: np.ndarray) -> Vector:
    """The circles on the sphere around unit directions, at angular radii: the plane at the cosine along each
    direction, `direction + cos(radius) * w`, scaled to square to one."""
    return (directions + mv.w * np.cos(radii)) * (1 / np.sin(radii))          # [...] Vector


def touching(circles: Vector, edges: np.ndarray) -> Vector:
    """The null vector of the point where the circles at the ends of each edge touch: their sum."""
    return circles[..., edges[:, 0]] + circles[..., edges[:, 1]]               # [..., edges] Vector


def face_circles(circles: Vector, faces: np.ndarray) -> Vector:
    """The circle of each face, through the points where its vertices' circles touch: orthogonal to all three,
    the dual of their wedge."""
    return (circles[..., faces[:, 0]] ^ circles[..., faces[:, 1]] ^ circles[..., faces[:, 2]]).dual()   # [..., faces] Vector


def apexes(circles: Vector) -> Vector:
    """The apex of the cone touching the sphere along each circle, as a point at unit weight along w."""
    return circles / -(circles | mv.w)                                         # [...] Vector


def centring(centre: Vector) -> Rotor:
    """The rotor carrying a unit timelike centre onto w: a boost along the centre's spatial part."""
    return (1 - mv.w * centre).normalized()


def boost(direction: Vector, rapidity: np.ndarray) -> Rotor:
    """The boost along a spatial unit direction by a rapidity: points of the sphere crowd towards the
    direction, circles there shrink and circles opposite grow."""
    return ((direction ^ mv.w) * (rapidity / 2)).exp()


# --- plumbing -------------------------------------------------------------------------
def basis() -> Vector:
    """x, y, z and w, as a batch."""
    return mv.vector(np.eye(4))                                                # [4] Vector
