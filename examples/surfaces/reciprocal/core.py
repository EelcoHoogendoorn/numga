"""A planar mesh lifted by heights at its vertices, the stresses its creases carry, and its reciprocal diagram.

Heights at the vertices of a planar triangle mesh lift it to a polyhedral terrain, a piecewise linear Airy
stress function. Each face is a plane, `height = slope | (point + e3)`, and the slope is linear in the heights:
the corners' heights times the reciprocal frame of the corners lifted to unit height. Across an interior edge
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

mesh_module = instantiate("examples.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()


# --- math -----------------------------------------------------------------------------
def slopes(mesh: Mesh, heights: np.ndarray) -> Vector:
    """The plane of each face of the terrain, through its corners at their heights, as the vector whose inner
    product with a point lifted to unit height, `point + e3`, is the height there: the slope across the plane
    and the height at the origin. The corners' heights times the reciprocal frame of the corners lifted to
    unit height, the three vectors with inner product one with their own corner and zero with the other two:
    linear in the heights, the heights on the last axis."""
    lifted = mesh.vertices.batch() + mv.z                                      # [vertices] Vector
    a, b, c = (lifted[mesh.faces[:, corner]] for corner in range(3))          # [faces] Vector each
    volume = (a ^ b ^ c).inverse()                                             # [faces] Trivector
    frame = stack([(b ^ c) * volume, (c ^ a) * volume, (a ^ b) * volume], axis=-1)   # [faces, 3] Vector
    return (frame * heights[..., mesh.faces]).sum(axis=-1)                     # [..., faces] Vector


def reciprocal(mesh: Mesh, heights: np.ndarray) -> Vector:
    """The vertices of the reciprocal diagram: the slope of each face, its height at the origin left out."""
    planes = slopes(mesh, heights)                                             # [..., faces] Vector
    return (planes ^ mv.z) * mv.z                                              # [..., faces] Vector


def stresses(mesh: Mesh, heights: np.ndarray) -> Scalar:
    """The stress of each interior edge: the jump in slope from the face on its left to the face on its
    right, over the edge's length, positive in tension, where the terrain bends convex."""
    inner, left, right = beside(mesh)                                          # [inner] each
    planes = slopes(mesh, heights)                                             # [..., faces] Vector
    jump = planes[..., left] - planes[..., right]                              # [..., inner] Vector
    bars = mesh.edge_vectors.batch()[inner]                                    # [inner] Vector
    return ((bars ^ jump ^ mv.z) * mv.xyz.reverse()) / (bars | bars)           # [..., inner] Scalar


def folds(mesh: Mesh, heights: np.ndarray) -> Scalar:
    """How far the corner across each interior edge on its right lies above the plane of the face on its
    left: positive where the terrain bends convex."""
    inner, left, right = beside(mesh)                                          # [inner] each
    far = mesh.faces[right].sum(axis=-1) - mesh.edges[inner].sum(axis=-1)    # [inner]
    lifted = mesh.vertices.batch()[far] + mv.z                                 # [inner] Vector
    return heights[..., far] - (slopes(mesh, heights)[..., left] | lifted)     # [..., inner] Scalar


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


def planar(coords: np.ndarray) -> Vector:
    """Points of the plane from their x and y coordinates."""
    return mv("x y", coords)
