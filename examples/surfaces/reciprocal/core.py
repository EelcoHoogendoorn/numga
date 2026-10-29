"""A planar mesh lifted by heights at its vertices, the stresses its creases carry, and its reciprocal diagram.

Heights at the vertices of a planar triangle mesh lift it to a polyhedral terrain, a piecewise linear Airy
stress function. The vertices are points, antivectors, intrinsic like all geometry. Each face is a plane, a
vector, and its height at a point is their pairing, `height = plane & (point + up)`, the point lifted to unit
height; the plane is linear in the heights: the corners' heights times the reciprocal frame of the corners
lifted to unit height, the meets of the corners in pairs over the meet of all three. Across an interior edge
the terrain creases, and the crease is a bar of the planar truss: the jump in slope, over the edge's length, is
its stress, positive in tension where the terrain is convex. The slopes of the faces are the vertices of the
reciprocal diagram, and the jump across each edge is its reciprocal edge, perpendicular to the bar and as long
as the bar's force. Every interior vertex is in equilibrium; the boundary carries the loads. Lifted onto the
paraboloid, the reciprocal diagram is the Voronoi diagram and the stresses are the cotangent weights.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.algebras import VGA3D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                         # Scalar+
# A face of the terrain is a plane, a vector; the vertices are points, antivectors.
Plane = ga.gatype.vector()                          # Vector+
Point = mesh_module.Point                           # Antivector-
# The point one unit above the origin: added to a point of the plane, it lifts it to unit height.
up = mv.z.dual()                                    # [] Point-


# --- math -----------------------------------------------------------------------------
def slopes(mesh: Mesh, points: Point, heights: Scalar) -> Plane:
    """The plane of each face of the terrain, through its corners at their heights, pairs with a point lifted
    to unit height, `point + up`, to the height there: it holds the slope across the plane and the height at
    the origin. It is the corners' heights times the reciprocal frame of the corners lifted to unit height, the
    three planes pairing to one with their own corner and to nothing with the other two, so it is linear in the
    heights."""
    a, b, c = mesh.incidence[2, 0] * (points + up)                             # Point-[faces], Point-[faces], Point-[faces]
    # Two corners meet in a vector that pairs to nothing with either; over the meet of all three, it pairs
    # to one with the third.
    volume = a & b & c                                                         # Scalar-[faces]
    frame = stack([(b & c) / volume, (c & a) / volume, (a & b) / volume], axis=-1)   # [corners] Plane+[faces]
    # The heights at each face's corners, on the last axis.
    corner_heights = mesh.incidence[2, 0] * heights[..., None]                 # [..., corners] Scalar+[faces]
    return (frame * corner_heights).sum(axis=-1)                               # [...] Plane+[faces]


def stresses(mesh: Mesh, points: Point, heights: Scalar) -> Scalar:
    """The stress of each interior edge is the jump in slope between the faces on either side, measured towards
    the corner across the edge, over the edge's length; it is positive in tension, where the terrain bends
    convex. Which face is called left does not matter: swapping them turns both the jump and the side the
    corner lies on."""
    inner = mesh.interior_edges                                                # [inner]
    # The faces' planes summed back onto their edges, each signed as its face runs along the edge: the plane of
    # the face on the left less that of the face on the right.
    jump = (~mesh.d[1] * slopes(mesh, points, heights)).batch()[..., inner]    # [..., inner] Plane+
    # Each bar from its tail to its head.
    tails, heads = (mesh.incidence[1, 0] * points).batch()[:, inner]           # [inner] Point-, [inner] Point-
    bars = heads - tails                                                       # [inner] Point-
    # The corner across each edge in the face on its right, as its offset square to the bar.
    far = (mesh.reading(mesh.edge_opposites[:, 1]) * points).batch()           # [inner] Point-
    corner = far - tails                                                       # [inner] Point-
    across = corner - bars * ((bars | corner) / (bars | bars))                # [inner] Point-
    # The jump in slope paired with that offset: how much faster the terrain rises across the bar.
    return -(jump & across) / (across.norm() * bars.norm())                    # [..., inner] Scalar+


def folds(mesh: Mesh, points: Point, heights: Scalar) -> Scalar:
    """The corner across each interior edge on its right lies above the plane of the face on its left by its
    fold, positive where the terrain bends convex."""
    # The plane of the face on the right passes through that corner at its height, so the fold is the jump from
    # the plane on the left to it, read at the corner lifted to unit height.
    jump = (mesh.reading(mesh.interior_edges, 1) * (~mesh.d[1] * slopes(mesh, points, heights))).batch()   # [..., inner] Plane+
    lifted = (mesh.reading(mesh.edge_opposites[:, 1]) * (points + up)).batch()   # [inner] Point-
    return -(jump & lifted)                                                    # [..., inner] Scalar+
