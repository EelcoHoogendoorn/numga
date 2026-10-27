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
UP = mv.z.dual()                                    # [] Point-


# --- math -----------------------------------------------------------------------------
def slopes(mesh: Mesh, points: Point, heights: np.ndarray) -> Plane:
    """The plane of each face of the terrain, through its corners at their heights, pairs with a point lifted
    to unit height, `point + UP`, to the height there: it holds the slope across the plane and the height at
    the origin. It is the corners' heights times the reciprocal frame of the corners lifted to unit height, the
    three planes pairing to one with their own corner and to nothing with the other two, so it is linear in the
    heights, which lie on the last axis."""
    lifted = points.batch() + UP                                               # [vertices] Point-
    a, b, c = (lifted[mesh.faces[:, corner]] for corner in range(3))          # [faces] Point-, Point-, Point-
    # Two corners meet in a vector that pairs to nothing with either; over the meet of all three, it pairs
    # to one with the third.
    volume = a & b & c                                                         # [faces] Scalar-
    frame = stack([(b & c) / volume, (c & a) / volume, (a & b) / volume], axis=-1)   # [faces, 3] Plane+
    return (frame * heights[..., mesh.faces]).sum(axis=-1)                     # [..., faces] Plane+


def stresses(mesh: Mesh, points: Point, heights: np.ndarray) -> Scalar:
    """The stress of each interior edge is the jump in slope between the faces on either side, measured towards
    the corner across the edge, over the edge's length; it is positive in tension, where the terrain bends
    convex. Which face is called left does not matter: swapping them turns both the jump and the side the
    corner lies on."""
    inner, (left, right) = mesh.interior_edges, mesh.edge_faces.T              # [inner], [inner], [inner]
    planes = slopes(mesh, points, heights)                                     # [..., faces] Plane+
    jump = planes[..., left] - planes[..., right]                              # [..., inner] Plane+
    # Each bar from its tail to its head.
    bars = points.batch()[mesh.edges[inner, 1]] - points.batch()[mesh.edges[inner, 0]]   # [inner] Point-
    # The corner across each edge in the face on its right, as its offset square to the bar.
    far = mesh.edge_opposites[:, 1]                                            # [inner]
    corner = points.batch()[far] - points.batch()[mesh.edges[inner, 0]]       # [inner] Point-
    across = corner - bars * ((bars | corner) / (bars | bars))                # [inner] Point-
    # The jump in slope paired with that offset: how much faster the terrain rises across the bar.
    return -(jump & across) / (across.norm() * bars.norm())                    # [..., inner] Scalar+


def folds(mesh: Mesh, points: Point, heights: np.ndarray) -> Scalar:
    """The corner across each interior edge on its right lies above the plane of the face on its left by its
    fold, positive where the terrain bends convex."""
    left = mesh.edge_faces[:, 0]                                               # [inner]
    far = mesh.edge_opposites[:, 1]                                            # [inner]
    lifted = points.batch()[far] + UP                                          # [inner] Point-
    return heights[..., far] - (slopes(mesh, points, heights)[..., left] & lifted)   # [..., inner] Scalar+
