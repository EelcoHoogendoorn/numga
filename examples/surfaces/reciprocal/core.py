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
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Point = mesh_module.Point
# The point one unit above the origin: added to a point of the plane, it lifts it to unit height.
UP = mv.z.dual()


# --- math -----------------------------------------------------------------------------
def slopes(mesh: Mesh, heights: np.ndarray) -> Vector:
    """The plane of each face of the terrain, through its corners at their heights, as the vector whose
    pairing with a point lifted to unit height, `point + UP`, is the height there: the slope across the
    plane and the height at the origin. The corners' heights times the reciprocal frame of the corners
    lifted to unit height, the three vectors pairing to one with their own corner and to nothing with the
    other two: linear in the heights, the heights on the last axis."""
    lifted = mesh.vertices.batch() + UP                                        # [vertices] Point
    a, b, c = (lifted[mesh.faces[:, corner]] for corner in range(3))          # [faces] Point each
    # Two corners meet in a vector that pairs to nothing with either; over the meet of all three, it pairs
    # to one with the third.
    volume = a & b & c                                                         # [faces] Scalar
    frame = stack([(b & c) / volume, (c & a) / volume, (a & b) / volume], axis=-1)   # [faces, 3] Vector
    return (frame * heights[..., mesh.faces]).sum(axis=-1)                     # [..., faces] Vector


def reciprocal(mesh: Mesh, heights: np.ndarray) -> Vector:
    """The vertices of the reciprocal diagram: the slope of each face, its height at the origin left out."""
    planes = slopes(mesh, heights)                                             # [..., faces] Vector
    return (planes ^ mv.z) * mv.z                                              # [..., faces] Vector


def stresses(mesh: Mesh, heights: np.ndarray) -> Scalar:
    """The stress of each interior edge: the jump in slope between the faces on either side, measured
    towards the corner across the edge, over the edge's length, positive in tension, where the terrain
    bends convex. Which face is called left does not matter: swapping them turns both the jump and the
    side the corner lies on."""
    inner, left, right = beside(mesh)                                          # [inner] each
    planes = slopes(mesh, heights)                                             # [..., faces] Vector
    jump = planes[..., left] - planes[..., right]                              # [..., inner] Vector
    bars = mesh.edge_displacements.batch()[inner]                              # [inner] Point
    # The corner across each edge in the face on its right, as its offset square to the bar.
    far = mesh.faces[right].sum(axis=-1) - mesh.edges[inner].sum(axis=-1)    # [inner]
    corner = mesh.vertices.batch()[far] - mesh.vertices.batch()[mesh.edges[inner, 0]]   # [inner] Point
    across = corner - bars * ((bars | corner) / (bars | bars))                # [inner] Point
    # The jump in slope paired with that offset: how much faster the terrain rises across the bar.
    return -(jump & across) / (across.norm() * bars.norm())                    # [..., inner] Scalar


def folds(mesh: Mesh, heights: np.ndarray) -> Scalar:
    """How far the corner across each interior edge on its right lies above the plane of the face on its
    left: positive where the terrain bends convex."""
    inner, left, right = beside(mesh)                                          # [inner] each
    far = mesh.faces[right].sum(axis=-1) - mesh.edges[inner].sum(axis=-1)    # [inner]
    lifted = mesh.vertices.batch()[far] + UP                                   # [inner] Point
    return heights[..., far] - (slopes(mesh, heights)[..., left] & lifted)     # [..., inner] Scalar


# --- plumbing -------------------------------------------------------------------------
def beside(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The interior edges, and the faces on their left and on their right, seen along the edge from its
    lower vertex to its higher."""
    edge = mesh.face_edges.reshape(-1)                                         # [faces * 3]
    face = np.repeat(np.arange(len(mesh.faces)), 3)                            # [faces * 3]
    along = mesh.face_edge_orientation.reshape(-1) > 0                        # [faces * 3]
    left, right = np.full(len(mesh.edges), -1), np.full(len(mesh.edges), -1)
    left[edge[along]], right[edge[~along]] = face[along], face[~along]
    inner = np.flatnonzero((left >= 0) & (right >= 0))
    return inner, left[inner], right[inner]


def planar(coords: np.ndarray) -> Point:
    """Points of the plane from their x and y coordinates."""
    return mv.antivector(np.concatenate([coords, np.zeros(coords.shape[:-1] + (1,))], axis=-1))
