"""A convex polyhedron, its polar dual about the unit sphere, and both projected onto the sphere.

Points anywhere in space around the centre span a convex hull. Each face of the hull has a pole with respect
to the unit sphere, the point whose inner product with every point of the face is one, and the poles are the
vertices of the polar dual: a vertex lies on a face exactly when the face's pole has inner product one with
the vertex. Seen from the centre, each point has an image on the sphere; the images of the hull are a
triangle mesh on the sphere, and the images of the poles its dual vertices. How far a point lies from the
centre does not move its image, but it tilts the faces around it, and with them the dual vertices. Along a
fixed direction a point's weight is one over its distance, and every edge's fold is linear in the weights:
the weights that keep every dual edge positive form a convex cone, found by a linear program.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()


# --- math -----------------------------------------------------------------------------
def poles(points: Vector, faces: np.ndarray) -> Vector:
    """The pole of each face with respect to the unit sphere: the point whose inner product with each of the
    face's vertices is one, the face's edges joined to the centre, summed, over the volume of its vertices."""
    a, b, c = (points[..., faces[:, corner]] for corner in range(3))           # [..., faces] Vector each
    return ((a ^ b) + (b ^ c) + (c ^ a)) * (a ^ b ^ c).inverse()               # [..., faces] Vector


def sides(points: Vector, faces: np.ndarray, duals: Vector) -> Scalar:
    """For each face and each of its edges, positive when the face's dual vertex, seen from the centre, lies
    on the face's side of the plane through the centre and that edge, as the face's third vertex does: all
    three positive when the dual vertex lands inside the face."""
    corners = points[..., faces]                                               # [..., faces, 3] Vector
    walls = corners ^ corners[..., [1, 2, 0]]                                  # [..., faces, 3] Bivector
    return (walls ^ duals[..., None]) * (walls ^ corners[..., [2, 0, 1]]).reverse()   # [..., faces, 3] Scalar


def dual_lengths(points: Vector, faces: np.ndarray) -> Scalar:
    """The signed length of each dual edge across its edge, seen from the centre: how much further the pole
    of the face on one side lies to that side of the plane through the centre and the edge than the pole
    of the face on the other side; negative where the two poles have crossed over."""
    edges, beside = adjacency(faces)                                           # [edges, 2] each
    duals = poles(points, faces)                                               # [..., faces] Vector
    wall = points[..., edges[:, 0]] ^ points[..., edges[:, 1]]                 # [..., edges] Bivector
    toward = wall ^ points[..., opposite(faces, edges, beside[:, 1])]          # [..., edges] Trivector
    return (wall ^ (duals[..., beside[:, 1]] - duals[..., beside[:, 0]])) * toward.reverse() / toward.norm()   # [..., edges] Scalar


def folds(directions: Vector, weights: np.ndarray, faces: np.ndarray) -> Scalar:
    """For each edge of the mesh, how far it bends outward, as on a convex polyhedron, or folds inward where
    negative, for points along the directions at the distances one over their weights: the volume of the
    edge's ends, the third vertex of one face beside it and the far vertex of the other, times all four
    weights. That is a sum over the four corners, each weight times the volume of the other three
    directions, with alternating signs: linear in the weights, the weights on the last axis. Each is scaled
    by the face's own volume of directions, positive on a mesh turning counter-clockwise from outside."""
    edges, beside = adjacency(faces)                                           # [edges, 2] each
    corners = np.concatenate([edges, opposite(faces, edges, beside[:, 0])[:, None],
                              opposite(faces, edges, beside[:, 1])[:, None]], axis=-1)   # [edges, 4]
    others = corners[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]          # [edges, 4, 3]
    volumes = directions[others[..., 0]] ^ directions[others[..., 1]] ^ directions[others[..., 2]]   # [edges, 4] Trivector
    face = volumes[:, 3]                                                       # [edges] Trivector
    return (volumes * (weights[..., corners] * [-1, 1, -1, 1])).sum(axis=-1) * face.reverse()   # [..., edges] Scalar


def insides(directions: Vector, weights: np.ndarray, faces: np.ndarray) -> Scalar:
    """For each face and each of its edges, the signed distance of the face's pole from the plane through the
    centre and that edge, positive on the face's side, for points along the directions at the distances one
    over their weights: all three positive when the pole lands inside the face. The pole is the sum of the
    corners' weights times the face's reciprocal frame, the three vectors with inner product one with their
    own corner's direction and zero with the other two, and the planes are fixed by the directions, so this
    is linear in the weights, the weights on the last axis."""
    a, b, c = (directions[faces[:, corner]] for corner in range(3))           # [faces] Vector each
    volume = (a ^ b ^ c).inverse()                                             # [faces] Trivector
    frame = stack([(b ^ c) * volume, (c ^ a) * volume, (a ^ b) * volume], axis=-1)   # [faces, 3] Vector
    duals = (frame * weights[..., faces]).sum(axis=-1)                         # [..., faces] Vector
    corners = directions[faces]                                                # [faces, 3] Vector
    walls = (corners ^ corners[:, [1, 2, 0]]).normalized()                     # [faces, 3] Bivector
    third = walls ^ corners[:, [2, 0, 1]]                                      # [faces, 3] Trivector
    return (walls ^ duals[..., None]) * third.reverse() / third.norm()         # [..., faces, 3] Scalar


# --- plumbing -------------------------------------------------------------------------
def adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge of a closed triangle mesh, and the two faces either side of it."""
    sides = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2)
    paired = np.lexsort(sides.T[::-1]).reshape(-1, 2)                          # each edge twice, once per face
    return sides[paired[:, 0]], paired // 3


def opposite(faces: np.ndarray, edges: np.ndarray, face: np.ndarray) -> np.ndarray:
    """The vertex of each given face across from its edge."""
    return faces[face].sum(axis=-1) - edges.sum(axis=-1)


def weighted(directions: Vector, weights: np.ndarray) -> Vector:
    """Points along the given unit directions at the distances one over their weights, the weights on the
    last axis."""
    return directions * (1 / weights)


def hull(points: Vector) -> np.ndarray:
    """The faces of the convex hull of points around the centre, counter-clockwise from outside. The one
    step outside the algebra: scipy finds the triangles from the points' coordinates, and each is turned
    so that the volume of its vertices turns as x, y, z do, with the centre behind it."""
    faces = ConvexHull(points.cast(ga.subspace("x y z")).kernel).simplices    # [faces, 3]
    a, b, c = (points[faces[:, corner]] for corner in range(3))               # [faces] Vector each
    turning = ((a ^ b ^ c) * mv.xyz.reverse()) > 0                             # [faces]
    return np.where(turning[:, None], faces, faces[:, [0, 2, 1]])
