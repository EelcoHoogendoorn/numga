"""Koebe polyhedra: every triangulation of the sphere as a polyhedron whose edges all touch the unit sphere.

The algebra has the unit sphere as its metric: x, y and z square to one and w to minus one. A circle on the
sphere, where the plane `n | x == s` cuts it, is the vector `n + s * w`, scaled to square to one, and a point x
of the sphere is held by its tangent plane, the null vector `x + w`: a point lies on the circle exactly when
their inner product is zero. Two circles touch from outside exactly when their inner product is minus one, and
then their sum is the tangent plane at the point where they touch. A circle's dual is a point, the apex of the
cone that touches the sphere along it: polarity in the sphere is the dual. So circles touching along every
edge of a triangulation have the vertices of a polyhedron whose edges all touch the sphere as their duals. The circle through the three touching points of a face is
orthogonal to its vertices' circles, the dual of their wedge, and its apex is a vertex of the dual
polyhedron, which touches the sphere at the same points. The equations fix the circles up to the maps that
keep the sphere, the rotors of the algebra; the centred one has the touching points balanced about the
centre.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from numga import Algebra, NumpyContext, concatenate, stack
from numga.sparse import SparseExtensor

ga = Algebra("x+y+z+w-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# A point of space, the dual of a plane: antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Rotor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def caps(directions: Vector, radius: Scalar) -> Vector:
    """The circles on the sphere around unit directions, at an angular radius: the plane at the cosine along
    each direction, `direction + cos(radius) * w`, scaled to square to one."""
    return (directions + mv.w * radius.cos()) / radius.sin()                   # [...] Vector


def touching(circles: Vector, edges: np.ndarray) -> Vector:
    """The tangent plane at the point where the circles at the ends of each edge touch: their sum, a null
    vector."""
    return circles[..., edges[:, 0]] + circles[..., edges[:, 1]]               # [..., edges] Vector


def face_circles(circles: Vector, faces: np.ndarray) -> Vector:
    """The circle of each face, through the points where its vertices' circles touch: orthogonal to all three,
    the dual of their wedge."""
    return (circles[..., faces[:, 0]] ^ circles[..., faces[:, 1]] ^ circles[..., faces[:, 2]]).dual()   # [..., faces] Vector


def apexes(circles: Vector) -> Point:
    """The apex of the cone touching the sphere along each circle, the circle's dual, at unit weight along w."""
    poles = circles.dual()                                                     # [...] Point
    return poles / (mv.w & poles)                                              # [...] Point


def koebe(start: Vector, edges: np.ndarray, n_steps: int) -> Vector:
    """The circles, by Gauss-Newton from the start, that square to one and touch from outside along every
    edge: inner product minus one. The residuals change with the circles by a map from the circles to the
    residuals; each step solves through that map composed with its adjoint, leaving the maps that keep the
    sphere, along which nothing changes, where they were."""
    tails, heads = edges[:, 0], edges[:, 1]
    n_vertices, n_edges = start.shape[-1], len(edges)
    # Each vertex's residual depends on its own circle, each edge's on the circles at its two ends.
    rows = np.concatenate([np.arange(n_vertices), n_vertices + np.arange(n_edges), n_vertices + np.arange(n_edges)])
    columns = np.concatenate([np.arange(n_vertices), tails, heads])
    circles = start                                                            # [vertices] Vector
    for _ in range(n_steps):
        residual = concatenate([circles.squared() - 1, (circles[tails] | circles[heads]) + 1]).field()   # Scalar[vertices + edges]
        jacobian = SparseExtensor.from_indices(
            concatenate([2 * circles | Vector, circles[heads] | Vector, circles[tails] | Vector]),
            rows, columns, (n_vertices + n_edges, n_vertices))                 # [vertices + edges, vertices] Scalar <- Vector
        circles = circles - jacobian.adjoint()(jacobian(jacobian.adjoint()).solve(residual)).batch()
    return circles


def centre(points: Vector, n_iterations: int) -> Vector:
    """The unit timelike vector that minimises the summed logarithms of minus its inner products with the
    tangent planes of points of the sphere: the centre seen from which they balance. There the points, each
    weighted by minus the reciprocal of its inner product with the centre, sum along the centre itself, so the
    centre is the fixed point of that weighted sum, normalized."""
    centre = mv.w                                                              # [] Vector
    for _ in range(n_iterations):
        weighted = (points * (-1 / (centre | points))).sum()                   # [] Vector
        centre = weighted / (-weighted.squared()).square_root()
    return centre


def centring(centre: Vector) -> Rotor:
    """The rotor carrying a unit timelike centre onto w: a boost along the centre's spatial part."""
    return (1 - mv.w * centre).normalized()


def boost(direction: Vector, rapidity: np.ndarray) -> Rotor:
    """The boost along a spatial unit direction by a rapidity: points of the sphere crowd towards the
    direction, circles there shrink and circles opposite grow."""
    return ((direction ^ mv.w) * (rapidity / 2)).exp()


# --- plumbing -------------------------------------------------------------------------
def spiral(n_points: int) -> Vector:
    """Directions spread evenly over the sphere along a Fibonacci spiral: z tilted towards x by each polar
    angle, then turned about z by each azimuth."""
    turns = np.arange(n_points) + 0.5
    polar, azimuth = np.arccos(1 - 2 * turns / n_points), np.pi * (1 + 5**0.5) * turns
    tilts = ((mv.z ^ mv.x) * (-polar / 2)).exp()                               # [points] Rotor
    return (mv.xy * (-azimuth / 2)).exp() >> (tilts >> mv.z)                   # [points] Vector


def beside(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The two faces either side of each edge, [edges, 2]."""
    sides = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2)
    order = np.lexsort(sides.T[::-1])
    return (order // 3).reshape(-1, 2)[np.argsort(np.lexsort(edges.T[::-1]))]


# --- the step outside the algebra -------------------------------------------------------
def hull(points: Vector) -> np.ndarray:
    """The faces `[faces, 3]` of the convex hull of points around the centre, read by their inner products
    with x, y and z."""
    return ConvexHull((points[..., None] | stack((mv.x, mv.y, mv.z))).to_array()).simplices   # [faces, 3]
