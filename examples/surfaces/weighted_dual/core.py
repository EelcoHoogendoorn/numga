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
from examples import instantiate

mv = NumpyContext(ga).multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Scalar = ga.gatype.scalar()                         # Scalar+
# Points, and directions from the centre, are antivectors.
Point = ga.gatype.antivector()                      # Antivector-


# --- math -----------------------------------------------------------------------------
def poles(points: Point, faces: np.ndarray) -> Point:
    """The pole of each face about the unit sphere is the dual of the face's plane, the plane that pairs to one
    with each of its vertices: the planes through the centre and each edge, summed, over the volume of the
    vertices."""
    a, b, c = (points[..., faces[:, corner]] for corner in range(3))           # [..., faces] Point-, Point-, Point-
    planes = ((a & b) + (b & c) + (c & a)) * (a & b & c).inverse()             # [..., faces] Plane+
    return planes.dual()                                                       # [..., faces] Point-


def sides(points: Point, faces: np.ndarray, duals: Point) -> Scalar:
    """Each of a face's three sides is positive where the face's pole, seen from the centre, lies on the face's
    side of the plane through the centre and an edge, as the face's third vertex does; all three are positive
    when the pole lands inside the face."""
    corners = points[..., faces]                                               # [..., faces, 3] Point-
    walls = corners & corners[..., [1, 2, 0]]                                  # [..., faces, 3] Plane-
    return (walls & duals[..., None]) * (walls & corners[..., [2, 0, 1]])      # [..., faces, 3] Scalar+


def dual_lengths(points: Point, mesh: Mesh) -> Scalar:
    """A dual edge's signed length, seen from the centre, is how much further the pole of the face on the right
    of its edge lies to that side than the pole of the face on the left; it is negative where the two poles have
    crossed over."""
    edges = mesh.edges[mesh.interior_edges]                                    # [edges, 2]
    left, right = mesh.edge_faces.T                                            # [edges], [edges]
    duals = poles(points, mesh.faces)                                          # [..., faces] Point-
    wall = points[..., edges[:, 0]] & points[..., edges[:, 1]]                 # [..., edges] Plane-
    toward = wall & points[..., mesh.edge_opposites[:, 1]]                     # [..., edges] Scalar-
    return (wall & (duals[..., right] - duals[..., left])) * toward / toward.norm()   # [..., edges] Scalar+


def folds(directions: Point, weights: np.ndarray, mesh: Mesh) -> Scalar:
    """An edge's fold is how far it bends outward, as on a convex polyhedron, or inward where negative, for
    points along the directions at the distances one over their weights: the volume of the edge's ends and the
    two vertices across it, times their four weights. That is a sum over the four corners of each weight times
    the volume of the other three directions, with alternating signs, so it is linear in the weights, which lie
    on the last axis. Each fold is scaled by the volume of the edge's ends and the vertex across it on the left,
    which sets the sense of the centre."""
    # Each edge's ends, then the vertices across it on its left and on its right.
    corners = np.concatenate([mesh.edges[mesh.interior_edges], mesh.edge_opposites], axis=-1)   # [edges, 4]
    others = corners[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]]          # [edges, 4, 3]
    volumes = directions[others[..., 0]] & directions[others[..., 1]] & directions[others[..., 2]]   # [edges, 4] Scalar-
    face = volumes[:, 3]                                                       # [edges] Scalar-
    return (volumes * (weights[..., corners] * [-1, 1, -1, 1])).sum(axis=-1) * face   # [..., edges] Scalar+


def insides(directions: Point, weights: np.ndarray, faces: np.ndarray) -> Scalar:
    """The signed distances of a face's pole inside the planes through the centre and each of its edges are
    positive on the face's side, all three when the pole lands inside the face. The face's plane is the corners'
    weights times the face's reciprocal frame, the three planes that pair to one with their own corner's
    direction and to zero with the other two, and the planes through the edges are fixed by the directions, so
    the distances are linear in the weights, which lie on the last axis."""
    a, b, c = (directions[faces[:, corner]] for corner in range(3))           # [faces] Point-, Point-, Point-
    volume = (a & b & c).inverse()                                             # [faces] Scalar-
    frame = stack([(b & c) * volume, (c & a) * volume, (a & b) * volume], axis=-1)   # [faces, 3] Plane+
    duals = (frame * weights[..., faces]).sum(axis=-1).dual()                  # [..., faces] Point-
    corners = directions[faces]                                                # [faces, 3] Point-
    walls = (corners & corners[:, [1, 2, 0]]).normalized()                     # [faces, 3] Plane-
    third = walls & corners[:, [2, 0, 1]]                                      # [faces, 3] Scalar-
    return (walls & duals[..., None]) * third / third.norm()                   # [..., faces, 3] Scalar+


# --- plumbing -------------------------------------------------------------------------
def hull(points: Point) -> np.ndarray:
    """The convex hull of points around the centre has triangle faces, counter-clockwise seen from outside, which
    scipy finds from the points' coordinates, one step outside the algebra."""
    # Each point read by pairing it with the planes across x, y and z.
    coordinates = (stack((mv.x, mv.y, mv.z), axis=-1) & points[..., None]).to_array()   # [points, 3]
    faces = ConvexHull(coordinates).simplices                                  # [faces, 3]
    # Each face turned to run counter-clockwise seen from outside: its corners' volume about the centre positive.
    a, b, c = (points[faces[:, corner]] for corner in range(3))               # [faces] Point-, Point-, Point-
    return np.where(((a & b & c) > 0)[:, None], faces, faces[:, ::-1])


def timing(frames: int) -> np.ndarray:
    """Each frame is so far along the way: at a steady pace for three quarters of the frames, then closing in on
    the end over the last quarter, the way left shrinking by so many e-folds, the pace continuous where it
    starts closing in."""
    closing_start, e_folds = 3 / 4, 3
    rate = e_folds / (1 - closing_start)
    # The way left where closing in starts.
    left = 1 / (1 + closing_start * rate)
    time = np.linspace(0, 1, frames)
    return np.where(time < closing_start, time * left * rate, 1 - left * np.exp(-(time - closing_start) * rate))
