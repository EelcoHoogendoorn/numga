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
from examples import instantiate

ga = Algebra("x+y+z+w-")
mv = NumpyContext(ga).multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Scalar = ga.gatype.scalar()                         # Scalar+
# Circles on the sphere, and the tangent planes that hold its points, are vectors.
Vector = ga.gatype.vector()                         # Vector+
# A point of space, the dual of a plane, is an antivector.
Point = ga.gatype.antivector()                      # Antivector-


# --- math -----------------------------------------------------------------------------
def touching(circles: Vector, edges: np.ndarray) -> Vector:
    """The tangent plane at the point where the circles at the ends of each edge touch is their sum, a null
    vector."""
    return circles[..., edges[:, 0]] + circles[..., edges[:, 1]]               # [..., edges] Vector+


def face_circles(circles: Vector, faces: np.ndarray) -> Vector:
    """The circle of each face, through the points where its vertices' circles touch, is orthogonal to all three:
    the dual of their wedge."""
    return (circles[..., faces[:, 0]] ^ circles[..., faces[:, 1]] ^ circles[..., faces[:, 2]]).dual()   # [..., faces] Vector+


def apexes(circles: Vector) -> Point:
    """The apex of the cone touching the sphere along each circle is the circle's dual, at unit weight along w."""
    poles = circles.dual()                                                     # [...] Point-
    return poles / (mv.w & poles)                                              # [...] Point-


def koebe(start: Vector, edges: np.ndarray, n_steps: int) -> Vector:
    """The circles that square to one and touch from outside along every edge, inner product minus one, follow
    by Gauss-Newton from the start; the steps leave the maps that keep the sphere, along which nothing changes,
    where they were."""
    tails, heads = edges[:, 0], edges[:, 1]
    n_vertices, n_edges = start.shape[-1], len(edges)
    # Each vertex's residual depends on its own circle, each edge's on the circles at its two ends.
    rows = np.concatenate([np.arange(n_vertices), n_vertices + np.arange(n_edges), n_vertices + np.arange(n_edges)])
    columns = np.concatenate([np.arange(n_vertices), tails, heads])
    circles = start                                                            # [vertices] Vector+
    for _ in range(n_steps):
        residual = concatenate([circles.squared() - 1, (circles[tails] | circles[heads]) + 1]).field()   # Scalar+[vertices + edges]
        jacobian = SparseExtensor.from_indices(
            concatenate([2 * circles | Vector, circles[heads] | Vector, circles[tails] | Vector]),
            rows, columns, (n_vertices + n_edges, n_vertices))                 # (Scalar+ <- Vector+)[vertices + edges <- vertices]
        # The change that zeroes the linearised residuals, along the circles' own metric: the step commutes with
        # the maps that keep the sphere, so a boosted start gives the boosted circles.
        circles = circles - jacobian.adjoint()(jacobian(jacobian.adjoint()).solve(residual)).batch()
    return circles


def centre(points: Vector, n_iterations: int) -> Vector:
    """The centre from which points of the sphere balance is the unit timelike vector that minimises the summed
    logarithms of minus its inner products with their tangent planes."""
    # There the points, each weighted by minus the reciprocal of its inner product with the centre, sum along
    # the centre itself: the centre is the fixed point of that weighted sum, normalized.
    centre = mv.w                                                              # [] Vector+
    for _ in range(n_iterations):
        weighted_sum = (points * (-1 / (centre | points))).sum()               # [] Vector+
        centre = weighted_sum / (-weighted_sum.squared()).square_root()
    return centre


# --- plumbing -------------------------------------------------------------------------
def spiral(n_points: int) -> Vector:
    """The directions are spread evenly over the sphere along a Fibonacci spiral: z tilted towards x by each polar
    angle, then turned about z by each azimuth."""
    turns = np.arange(n_points) + 0.5
    polar, azimuth = np.arccos(1 - 2 * turns / n_points), np.pi * (1 + 5**0.5) * turns
    tilts = (mv.zx * (-polar / 2)).exp()                               # [points] Rotor+
    return (mv.xy * (-azimuth / 2)).exp() >> (tilts >> mv.z)                   # [points] Vector+


# --- the step outside the algebra -------------------------------------------------------
def hull(points: Vector) -> np.ndarray:
    """The convex hull of points around the centre has faces `[faces, 3]`, counter-clockwise seen from outside,
    found from the points' inner products with x, y and z."""
    coordinates = (points[..., None] | stack((mv.x, mv.y, mv.z))).to_array()  # [points, 3]
    faces = ConvexHull(coordinates).simplices                                  # [faces, 3]
    # Each face turned to run counter-clockwise seen from outside: its corners' volume about the centre positive.
    return np.where((np.linalg.det(coordinates[faces]) > 0)[:, None], faces, faces[:, ::-1])
