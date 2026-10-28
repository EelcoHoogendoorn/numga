"""Koebe polyhedra: every triangulation of the sphere as a polyhedron whose edges all touch the unit sphere.

The algebra has the unit sphere as its metric: x, y and z square to one and w to minus one. A circle on the
sphere, where the plane `n | x == s` cuts it, is the vector `n + s * w`, scaled to square to one. A point is an
antivector, and the points of the sphere are its null antivectors: a point lies on a circle exactly when
`circle & point` is zero. A circle's dual is a point, the apex of the cone that touches the sphere along it:
polarity in the sphere is the dual. Two circles touch from outside exactly when their inner product is minus
one, and then the sum of their poles is the point where they touch, the null point on the line between the
apexes. So circles touching along every edge of a triangulation have the vertices of a polyhedron whose edges
all touch the sphere as their duals. The circle through the three touching points of a face is orthogonal to
its vertices' circles, the dual of their wedge, and its apex is a vertex of the dual polyhedron, which touches
the sphere at the same points. The equations fix the circles up to the maps that keep the sphere, the rotors of
the algebra; the centred one has the touching points balanced about the centre of the sphere.
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
# Circles on the sphere, the planes that cut it, are vectors.
Vector = ga.gatype.vector()                         # Vector+
# A point, of space or of the sphere, is an antivector, the pole of its plane; so is a direction, a point at infinity.
Point = ga.gatype.antivector()                      # Antivector-
# The centre of the sphere is the pole of the plane at infinity, w, which pairs with a point to give its weight.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def touching(circles: Vector, edges: np.ndarray) -> Point:
    """The point where the circles at the ends of each edge touch is the sum of their poles, the null point on
    the line between the apexes."""
    poles = circles.dual()                                                     # [...] Point-
    return poles[..., edges[:, 0]] + poles[..., edges[:, 1]]                   # [..., edges] Point-


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


def centre(points: Point, n_iterations: int) -> Point:
    """The centre from which points of the sphere balance is the point inside it, squaring to minus one, that
    minimises the summed logarithms of minus its inner products with them."""
    # There the points, each weighted by minus the reciprocal of its inner product with the centre, sum along
    # the centre itself: the centre is the fixed point of that weighted sum, normalized.
    centre = origin                                                            # [] Point-
    for _ in range(n_iterations):
        weighted_sum = (points * (-1 / (centre | points))).sum()               # [] Point-
        centre = weighted_sum.normalized()
    return centre


# --- plumbing -------------------------------------------------------------------------
def spiral(n_points: int) -> Point:
    """The directions are spread evenly over the sphere along a Fibonacci spiral: the direction z tilted towards x
    by each polar angle, then turned about z by each azimuth."""
    turns = np.arange(n_points) + 0.5
    polar, azimuth = np.arccos(1 - 2 * turns / n_points), np.pi * (1 + 5**0.5) * turns
    tilts = (mv.zx * (-polar / 2)).exp()                               # [points] Rotor+
    return (mv.xy * (-azimuth / 2)).exp() >> (tilts >> mv.z.dual())            # [points] Point-


# --- the step outside the algebra -------------------------------------------------------
def hull(offsets: Point) -> np.ndarray:
    """The convex hull of points at offsets from the centre has faces `[faces, 3]`, counter-clockwise seen from
    outside, found from the offsets' pairings with the planes x, y and z."""
    coordinates = (stack((mv.x, mv.y, mv.z)) & offsets[..., None]).to_array()  # [points, 3]
    faces = ConvexHull(coordinates).simplices                                  # [faces, 3]
    # Each face turned to run counter-clockwise seen from outside: its corners' volume about the centre positive.
    return np.where((np.linalg.det(coordinates[faces]) > 0)[:, None], faces, faces[:, ::-1])
