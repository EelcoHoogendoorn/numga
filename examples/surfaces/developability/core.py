"""Developable surfaces by flow, in the geometric algebra of three-dimensional space.

A developable surface can be unrolled flat without stretching or tearing, like a sheet of paper bent
into a cylinder or a cone. Stein, Grinspun and Crane, "Developability of triangle meshes" (2018),
measure how far a triangle mesh is from that, and flow it towards it: smooth surfaces crease into
patches that could each be cut from a flat sheet, joined along sharp folds.

A smooth surface is developable where its normals, all around a point, lie on one great circle of the
sphere of directions: the surface bends in one direction only, along the rulings, and the normals turn
about the rulings' direction. At a vertex of a mesh, the normals of the faces around it lie on one
great circle exactly when one direction is perpendicular to all of them. Their spread, the sum of
dyads `normal * (normal | Vector)` over the faces around the vertex, each weighted by the face's angle
at the vertex, is a map whose smallest eigenvalue measures how far they are from that: zero when some
direction `u` meets every normal at a right angle, and growing as the normals spread over the sphere.
The angles make the spread a property of the surface rather than of its triangles: a face split in two
keeps its angle at the vertex. The energy is the sum of those smallest eigenvalues over the vertices.

The flow moves every vertex down the energy's gradient. The smallest eigenvalue is the least value of
the spread's form over unit directions, so its change, to first order, is the change of the form at
the minimizing direction `u`, held fixed: each face's term, its angle times `(u | normal)**2`, changes
as its normal turns and as its angle opens. The vertices and edges are points, antivectors, and two edges
meet in their face's normal, twice its area long. Moving one corner of a face turns the face's normal by
the part of the step across the edge opposite the corner, over twice the face's area, so each corner's
share is the tangential part of the vertices' weighted directions pulled onto the normal, met with the
opposite edge, `across | edge`, over twice the area. A corner's angle opens as either neighbour moves
along the face square to the edge between them, by the step over that edge's length: the edge met with
the corner's normal, `edge | corner`, over the edge's length squared and the normal's size. The gradient
is a vector, the partner a vertex's step pairs with, `gradient & step`; the flow turns it into a step by
the metric, `.dual()`.

In matrix notation the energy reads $E = \\sum_i \\lambda_{\\min}\\left(\\sum_{ijk}
\\theta_i^{jk} N_{ijk} N_{ijk}^{\\mathsf T}\\right)$ over the faces $ijk$ around each vertex $i$, with
$\\theta_i^{jk}$ the face's angle at the vertex.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import VGA3D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar = mesh_module.Mesh, mesh_module.as_scalar
mv = mesh_module.ctx.multivector

Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Point = mesh_module.Point


# --- math -----------------------------------------------------------------------------
def spread(mesh: Mesh, edges: Point, angles: Scalar) -> tuple[Vector, Vector]:
    """The unit normal of each face `Vector[F]`, and the spread of the face normals around each vertex,
    the sum of their dyads, each weighted by the face's angle at the vertex, a map `Vector <- Vector` at
    each vertex."""
    # two edges meet in the face's normal
    normals = (edges[0] & edges[1]).normalized().field()                        # Vector[F]
    around = ~mesh.at_corners(angles)                                           # [V, F] Scalar
    return normals, around * (normals * (normals | Vector))                    # Vector[F], [V] Vector <- Vector


def gradient(mesh: Mesh, edges: Point, angles: Scalar, normals: Vector, directions: Vector) -> Vector:
    """The energy's gradient at each vertex `Vector[V]`, with each vertex's least-spread direction
    `[V] Vector` held fixed. A face's term at a corner is its angle there times the square of the corner's
    direction along its normal, and changes both ways. As the normal turns: for each face, the angle-
    weighted dyads of its corners' directions pulled onto its normal, the tangential part twice, wedged
    with the edge facing each corner over twice the face's area. As the angle opens: each corner's angle
    moves with its two neighbours, each along the face square to the edge from the corner, over that
    edge's length, and with the corner itself against both."""
    double_areas = (edges[0] & edges[1]).norm()                                 # [F] Scalar
    face_normals = normals.batch()                                              # [F] Vector
    corner_directions = directions[mesh.faces.T]                                # [3, F] Vector
    along = corner_directions | face_normals                                    # [3, F] Scalar
    pulled = (corner_directions * (along * angles)).sum(axis=0)                 # [F] Vector
    across = 2 * (pulled - face_normals * (face_normals | pulled))             # [F] Vector
    turning = -(across | edges) / double_areas                                  # [3, F] Vector: one per corner
    # The edges from each corner to the next corner and to the one after, and where they meet.
    ahead, behind = edges[[2, 0, 1]], -edges[[1, 2, 0]]                         # [3, F] Point each
    corner = ahead & behind                                                     # [3, F] Vector
    weight = along * along / corner.norm()                                      # [3, F] Scalar
    # Each edge met with the corner turns it a quarter turn within the face.
    to_next = -(ahead | corner) / ahead.scalar_norm_squared() * weight          # [3, F] Vector
    to_after = (behind | corner) / behind.scalar_norm_squared() * weight        # [3, F] Vector
    opening = to_next[[2, 0, 1]] + to_after[[1, 2, 0]] - to_next - to_after     # [3, F] Vector: one per corner
    return ~mesh.at_corners(turning + opening) * as_scalar(np.ones(len(mesh.faces))).field()   # Vector[V]


def stiffness(mesh: Mesh, edges: Point, angles: Scalar) -> Scalar:
    """How sharply the energy curves at each vertex `Scalar[V]`: moving a corner turns its face's normal by
    the step over the corner's height above the opposite edge, so each face around the vertex adds its
    angle there over that height squared, `(edge | edge) / double_areas**2`."""
    double_areas = (edges[0] & edges[1]).norm()                                 # [F] Scalar
    return ~mesh.at_corners(angles * edges.scalar_norm_squared() / double_areas.squared()) * as_scalar(np.ones(len(mesh.faces))).field()


def flow(mesh: Mesh, vertices: Point, step: float) -> Iterator[tuple[Point, Scalar]]:
    """Gradient descent on the developability energy, each vertex's step the gradient over the energy's
    stiffness there, times `step`: at every step, without end, the vertices `Point[V]` and each vertex's
    smallest eigenvalue `[V] Scalar`. A step of the same size everywhere
    overshoots where triangles are small or thin, and the flow jitters there; over the stiffness, every
    vertex moves by the same share of what its faces can take."""
    while True:
        # The edge facing each corner of each face, and the angle at each corner between the two edges meeting there.
        edges = (mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T   # [3, F] Point
        after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]                                       # [3, F] Point each
        angles = (-(after.dual_inverse() & before) / (after.norm() * before.norm())).arccos()    # [3, F] Scalar
        normals, spreads = spread(mesh, edges, angles)
        values, vectors = spreads.batch().eigh()                                # [V, 3] Scalar, [V, 3] Vector
        yield vertices, values[:, 0]
        # The gradient pairs with a step; the metric turns it into one.
        vertices = vertices - step * (gradient(mesh, edges, angles, normals, vectors[:, 0]) / stiffness(mesh, edges, angles)).dual()
