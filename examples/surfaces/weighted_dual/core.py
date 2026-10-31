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

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga
from examples import instantiate

mv = NumpyContext(ga).multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Scalar = ga.gatype.scalar()                         # Scalar+
# Points, and directions from the centre, are antivectors.
Point = ga.gatype.antivector()                      # Antivector-


# --- math -----------------------------------------------------------------------------
def poles(mesh: Mesh, points: Point) -> Point:
    """The pole of each face about the unit sphere is the dual of the face's plane, the plane that pairs to one
    with each of its vertices: the planes through the centre and each edge, summed, over the volume of the
    vertices."""
    corners = mesh.incidence[2, 0] * points                                    # [corners] Point-[faces]
    tails, heads = mesh.incidence[1, 0] * points                               # Point-[edges], Point-[edges]
    planes = (mesh.d[1] * (tails & heads)) / (corners[0] & corners[1] & corners[2])   # Plane+[faces]
    return planes.dual()                                                       # Point-[faces]


def sides(mesh: Mesh, points: Point, duals: Point) -> Scalar:
    """Each of a face's three sides is positive where the face's pole, seen from the centre, lies on the face's
    side of the plane through the centre and an edge, as the face's third vertex does; all three are positive
    when the pole lands inside the face."""
    corners = mesh.incidence[2, 0] * points                                    # [corners] Point-[faces]
    tails, heads = mesh.incidence[1, 0] * points                               # Point-[edges], Point-[edges]
    # The plane through the centre and the edge facing each corner, turned as the face runs along the edge.
    walls = mesh.boundary[2, 1] * (tails & heads)                              # [corners] Plane-[faces]
    return (walls & duals) * (walls & corners)                                 # [corners] Scalar+[faces]


def dual_lengths(mesh: Mesh, points: Point) -> Scalar:
    """A dual edge's signed length, seen from the centre, is how much further the pole of the face on the right
    of its edge lies to that side than the pole of the face on the left; it is negative where the two poles have
    crossed over."""
    inner = mesh.interior_edges                                                # [inner]
    tails, heads = (mesh.incidence[1, 0] * points).batch()[:, inner]           # [inner] Point-, [inner] Point-
    wall = tails & heads                                                       # [inner] Plane-
    toward = wall & (mesh.reading(mesh.edge_opposites[:, 1]) * points).batch()   # [inner] Scalar-
    # The poles summed back onto their faces' edges, each signed as its face runs along the edge, are the pole on
    # the left less the pole on the right: the step across each edge from left to right is their negative.
    crossing = -(~mesh.d[1] * poles(mesh, points)).batch()[inner]              # [inner] Point-
    return (wall & crossing) * toward / toward.norm()                          # [inner] Scalar+


def folds(mesh: Mesh, directions: Point, weights: Scalar) -> Scalar:
    """An edge's fold is how far it bends outward, as on a convex polyhedron, or inward where negative, for
    points along the directions at the distances one over their weights: the volume of the edge's ends and the
    two vertices across it, times their four weights. That is a sum over the four corners of each weight times
    the volume of the other three directions, with alternating signs, so it is linear in the weights. Each fold
    is scaled by the volume of the edge's ends and the vertex across it on the left, which sets the sense of the
    centre."""
    inner = mesh.interior_edges                                                # [inner]
    # Each edge's ends, then the vertices across it on its left and on its right.
    ends, across = mesh.incidence[1, 0], mesh.reading(mesh.edge_opposites.T)   # [2] Scalar+[edges <- points], [2] Scalar+[inner <- points]
    tail, head = (ends * directions).batch()[:, inner]                         # [inner] Point-, [inner] Point-
    left, right = (across * directions).batch()                                # [inner] Point-, [inner] Point-
    face = tail & head & left                                                  # [inner] Scalar-
    volumes = stack([head & left & right, tail & left & right, tail & head & right, face])   # [4, inner] Scalar-
    # The four weights in the same order, on the second axis from the end, and the alternating signs of the sum.
    corner_weights = concatenate([(ends * weights[..., None]).batch()[..., inner],
                                  (across * weights[..., None]).batch()], axis=-2)   # [..., 4, inner] Scalar+
    alternating = np.array([-1, 1, -1, 1])[:, None]
    return (volumes * (corner_weights * alternating)).sum(axis=-2) * face      # [..., inner] Scalar+


def insides(mesh: Mesh, directions: Point, weights: Scalar) -> Scalar:
    """The signed distances of a face's pole inside the planes through the centre and each of its edges are
    positive on the face's side, all three when the pole lands inside the face. The face's plane is the corners'
    weights times the face's reciprocal frame, the three planes that pair to one with their own corner's
    direction and to zero with the other two, and the planes through the edges are fixed by the directions, so
    the distances are linear in the weights."""
    corners = mesh.incidence[2, 0] * directions                                # [corners] Point-[faces]
    tails, heads = mesh.incidence[1, 0] * directions                           # Point-[edges], Point-[edges]
    # The plane through the centre and the edge facing each corner, turned as the face runs along the edge.
    edge_planes = mesh.boundary[2, 1] * (tails & heads)                        # [corners] Plane-[faces]
    frame = edge_planes / (corners[0] & corners[1] & corners[2])               # [corners] Plane+[faces]
    # The weights at each face's corners, on the last axis.
    duals = (frame * (mesh.incidence[2, 0] * weights[..., None])).sum(axis=-1).dual()   # [...] Point-[faces]
    walls = edge_planes.normalized()                                           # [corners] Plane-[faces]
    facing = walls & corners                                                   # [corners] Scalar-[faces]
    return (walls & duals[..., None]) * facing / facing.norm()                 # [..., corners] Scalar+[faces]


# --- plumbing -------------------------------------------------------------------------
def hull(points: Point) -> Mesh:
    """The convex hull of points around the centre has triangle faces, counter-clockwise seen from outside, which
    scipy finds from the points' coordinates, one step outside the algebra."""
    # Each point read by pairing it with the planes across x, y and z.
    coordinates = (stack((mv.x, mv.y, mv.z), axis=-1) & points.batch()[..., None]).to_array()   # [points, 3]
    return Mesh.closed(ConvexHull(coordinates).simplices, coordinates)


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
