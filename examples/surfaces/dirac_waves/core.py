"""Waves of a spinor field on a surface, carried by the surface's Dirac operator, in the geometric
algebra of three-dimensional space.

The field is a quaternion on each vertex, an even multivector, and the Dirac operator of the spin
transformations takes it to a quaternion on each face. The operator is a sparse extensor: a field of
the mesh's edges, points, with one cell for each face and each of its corners, the edge that corner
faces over twice the face's area. Its geometric product with a vertex field, `dirac * vertices`,
multiplies each cell into the quaternion at its corner and sums over each face's corners, taking
`Even[V]` to a face field `Even[F]`. Its reverse `~dirac` turns every product with it around, as any reverse does:
every cell reversed, and every coupling running the other way, from a face to its corners.

A wave alternates the two: the face field moved on by the Dirac operator of the vertex field, then the
vertex field moved back by the reverse of the face field, weighted by the areas, a leapfrog that keeps
the field's energy, the area-weighted squared size of both. Twice over the step is the operator
`~dirac * face_areas * dirac` of the spin transformations, taking vertex fields to vertex fields; its
product with `Even`, the space of all quaternion fields, makes it a map `Even[V] <- Even[V]` that the
eigensolver takes, so the standing waves are its eigenfields against the vertex areas,
`vertex_areas * Even`. On the unit sphere their frequencies are the whole numbers, each held eight
times its own.

Lengths are in units of the sphere's radius and times in units of the radius over the waves' speed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.mesh", ga)
Mesh, context = mesh_module.Mesh, mesh_module.ctx
mv = context.multivector
Scalar = ga.gatype.scalar()
Point = mesh_module.Point
Even = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
def dirac(mesh: Mesh) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """The Dirac operator from vertices to faces, the face areas, and the vertex areas and their
    inverses."""
    # diagonal operators: the triangle areas, and the vertex areas and their inverses
    face_areas = spdiag(mesh.triangle_areas)                               # [F, F] Scalar
    inverse_face_areas = spdiag(1 / mesh.triangle_areas)                   # [F, F] Scalar
    vertex_areas = spdiag(mesh.vertex_areas)                               # [V, V] Scalar
    inverse_vertex_areas = spdiag(1 / mesh.vertex_areas)                   # [V, V] Scalar

    # each face takes each corner by the edge it faces, over twice the face's area
    operator = inverse_face_areas * mesh.at_corners(mesh.triangle_edges) * 0.5    # [F, V] Point
    return operator, face_areas, vertex_areas, inverse_vertex_areas


def spread(mesh: Mesh, start: Even, interval: float) -> Iterator[tuple[Even, Even]]:
    """A wave from the given vertex field, without end: the vertex field at each step and the face
    field half a step after it."""
    operator, face_areas, _, inverse_vertex_areas = dirac(mesh)
    vertices, faces = start, operator * start * 0.0                         # Even[V], Even[F]
    while True:
        # the face field moved on by the Dirac operator of the vertex field
        faces = faces + operator * vertices * interval                      # Even[F]
        yield vertices, faces
        # the vertex field moved back by the reverse of the face field, weighted by the areas
        vertices = vertices - inverse_vertex_areas * (~operator * (face_areas * faces)) * interval   # Even[V]


def standing(mesh: Mesh, count: int) -> tuple[Scalar, Even]:
    """The count standing waves of least frequency: their frequencies squared and their vertex fields,
    the eigenfields of `~dirac * face_areas * dirac` against the vertex areas."""
    operator, face_areas, vertex_areas, _ = dirac(mesh)
    return ((~operator * face_areas * operator) * Even).eigh(vertex_areas * Even, count)   # [count] Scalar, [count] Even[V]


def oscillation(mesh: Mesh, waves: Even, frequencies: Scalar, phases: np.ndarray) -> tuple[Even, Even]:
    """Standing waves at the given phases of their periods `[waves, phases]`: the vertex field a
    cosine, and the face field, the Dirac operator of it over the frequency, a sine."""
    operator, _, _, _ = dirac(mesh)
    vertices = waves[:, None] * np.cos(phases)                              # [waves, phases] Even[V]
    faces = (operator * waves / frequencies)[:, None] * np.sin(phases)      # [waves, phases] Even[F]
    return vertices, faces
