"""A convex polyhedron, its polar dual about the unit sphere, and both projected onto the sphere.

Points anywhere in space around the centre span a convex hull. Each face lies in a plane, the one that pairs
to one with every point of the face, and the dual of that plane about the unit sphere is the face's pole; the
poles are the vertices of the polar dual. Seen from the centre, each point has an image on the sphere; the images of the hull are a
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
Plane = ga.gatype.vector()
# Points, and directions from the centre, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()


# --- math -----------------------------------------------------------------------------
def poles(points: Point, faces: np.ndarray) -> Point:
    """The pole of each face with respect to the unit sphere, the dual of the face's plane. The plane is the
    one that pairs to one with each of the face's vertices: the planes through the centre and each edge,
    summed, over the volume of its vertices."""
    a, b, c = (points[..., faces[:, corner]] for corner in range(3))           # [..., faces] Point each
    planes = ((a & b) + (b & c) + (c & a)) * (a & b & c).inverse()             # [..., faces] Plane
    return planes.dual()                                                       # [..., faces] Point


def sides(points: Point, faces: np.ndarray, duals: Point) -> Scalar:
    """For each face and each of its edges, positive when the face's dual vertex, seen from the centre, lies
    on the face's side of the plane through the centre and that edge, as the face's third vertex does: all
    three positive when the dual vertex lands inside the face."""
    corners = points[..., faces]                                               # [..., faces, 3] Point
    walls = corners & corners[..., [1, 2, 0]]                                  # [..., faces, 3] Plane
    return (walls & duals[..., None]) * (walls & corners[..., [2, 0, 1]])      # [..., faces, 3] Scalar


def dual_lengths(points: Point, faces: np.ndarray) -> Scalar:
    """The signed length of each dual edge across its edge, seen from the centre: how much further the pole
    of the face on one side lies to that side of the plane through the centre and the edge than the pole
    of the face on the other side; negative where the two poles have crossed over."""
    edges, beside = adjacency(faces)                                           # [edges, 2] each
    duals = poles(points, faces)                                               # [..., faces] Point
    wall = points[..., edges[:, 0]] & points[..., edges[:, 1]]                 # [..., edges] Plane
    toward = wall & points[..., opposite(faces, edges, beside[:, 1])]          # [..., edges] Scalar
    return (wall & (duals[..., beside[:, 1]] - duals[..., beside[:, 0]])) * toward / toward.norm()   # [..., edges] Scalar


def folds(directions: Point, weights: np.ndarray, faces: np.ndarray) -> Scalar:
    """For each edge of the mesh, how far it bends outward, as on a convex polyhedron, or folds inward where
    negative, for points along the directions at the distances one over their weights: the volume of the
    edge's ends, the third vertex of one face beside it and the far vertex of the other, times all four
    weights. That is a sum over the four corners, each weight times the volume of the other three
    directions, with alternating signs: linear in the weights, the weights on the last axis. Each is scaled
    by the volume of the edge's ends and the first face's third vertex, which sets the sense of the centre."""
    edges, beside = adjacency(faces)                                           # [edges, 2] each
    corners = np.concatenate([edges, opposite(faces, edges, beside[:, 0])[:, None],
                              opposite(faces, edges, beside[:, 1])[:, None]], axis=-1)   # [edges, 4]
    others = corners[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]          # [edges, 4, 3]
    volumes = directions[others[..., 0]] & directions[others[..., 1]] & directions[others[..., 2]]   # [edges, 4] Scalar
    face = volumes[:, 3]                                                       # [edges] Scalar
    return (volumes * (weights[..., corners] * [-1, 1, -1, 1])).sum(axis=-1) * face   # [..., edges] Scalar


def insides(directions: Point, weights: np.ndarray, faces: np.ndarray) -> Scalar:
    """For each face and each of its edges, the signed distance of the face's pole from the plane through the
    centre and that edge, positive on the face's side, for points along the directions at the distances one
    over their weights: all three positive when the pole lands inside the face. The face's plane is the sum
    of the corners' weights times the face's reciprocal frame, the three planes that pair to one with their
    own corner's direction and to zero with the other two, and the planes through the edges are fixed by
    the directions, so this is linear in the weights, the weights on the last axis."""
    a, b, c = (directions[faces[:, corner]] for corner in range(3))           # [faces] Point each
    volume = (a & b & c).inverse()                                             # [faces] Scalar
    frame = stack([(b & c) * volume, (c & a) * volume, (a & b) * volume], axis=-1)   # [faces, 3] Plane
    duals = (frame * weights[..., faces]).sum(axis=-1).dual()                  # [..., faces] Point
    corners = directions[faces]                                                # [faces, 3] Point
    walls = (corners & corners[:, [1, 2, 0]]).normalized()                     # [faces, 3] Plane
    third = walls & corners[:, [2, 0, 1]]                                      # [faces, 3] Scalar
    return (walls & duals[..., None]) * third / third.norm()                   # [..., faces, 3] Scalar


# --- plumbing -------------------------------------------------------------------------
def adjacency(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge of a closed triangle mesh, and the two faces either side of it."""
    sides = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2)
    paired = np.lexsort(sides.T[::-1]).reshape(-1, 2)                          # each edge twice, once per face
    return sides[paired[:, 0]], paired // 3


def opposite(faces: np.ndarray, edges: np.ndarray, face: np.ndarray) -> np.ndarray:
    """The vertex of each given face across from its edge."""
    return faces[face].sum(axis=-1) - edges.sum(axis=-1)


def weighted(directions: Point, weights: np.ndarray) -> Point:
    """Points along the given unit directions at the distances one over their weights, the weights on the
    last axis."""
    return directions * (1 / weights)


def hull(points: Point) -> np.ndarray:
    """The faces of the convex hull of points around the centre. The one step outside the algebra: scipy
    finds the triangles from the points' coordinates."""
    return ConvexHull(points.cast(ga.subspace("yz zx xy")).kernel).simplices  # [faces, 3]
