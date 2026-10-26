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

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, context = mesh_module.Mesh, mesh_module.ctx
as_ga_sparse, as_scalar = mesh_module.as_ga_sparse, mesh_module.as_scalar
mv = context.multivector
Scalar = ga.gatype.scalar()
Point = mesh_module.Point
Even = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
def spread(mesh: Mesh, points: Point, start: Even, interval: float) -> Iterator[tuple[Even, Even]]:
    """A wave from the given vertex field, without end: the vertex field at each step and the face
    field half a step after it."""
    # The edge facing each corner, from the corner after it to the corner after that: one of the edges, signed as
    # the face runs along it.
    edges = ((mesh.d0 * points).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    # A face's area is half the size of its plane, the commutator of two of its edges; a vertex's is a third
    # of the area of each face at it.
    triangle_areas = edges[1].commutator(edges[0]).norm() / 2              # Scalar[F]
    thirds = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    dual_areas = ~thirds * triangle_areas                                  # Scalar[V]

    face_areas = spdiag(triangle_areas)                                    # [F, F] Scalar
    inverse_face_areas = spdiag(1 / triangle_areas)                        # [F, F] Scalar
    inverse_vertex_areas = spdiag(1 / dual_areas)                          # [V, V] Scalar
    dirac = inverse_face_areas * mesh.at_corners(edges) * 0.5              # [F, V] Point

    vertices, faces = start, dirac * start * 0.0                           # Even[V], Even[F]
    while True:
        # the face field moved on by the Dirac operator of the vertex field
        faces = faces + dirac * vertices * interval                        # Even[F]
        yield vertices, faces
        # the vertex field moved back by the reverse of the face field, weighted by the areas
        vertices = vertices - inverse_vertex_areas * (~dirac * (face_areas * faces)) * interval   # Even[V]


def modes(mesh: Mesh, points: Point, sizes: np.ndarray, probe: Even, frequencies: np.ndarray,
          phases: np.ndarray) -> tuple[Scalar, Even, Even]:
    """The standing waves of the first few frequencies, one of each, over its period: the frequencies
    squared of all the waves found, and the vertex and face fields `[frequencies, phases] Even[V]` and
    `[frequencies, phases] Even[F]`."""
    edges = ((mesh.d0 * points).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    triangle_areas = edges[1].commutator(edges[0]).norm() / 2              # Scalar[F]
    thirds = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    dual_areas = ~thirds * triangle_areas                                  # Scalar[V]

    face_areas = spdiag(triangle_areas)                                    # [F, F] Scalar
    inverse_face_areas = spdiag(1 / triangle_areas)                        # [F, F] Scalar
    vertex_areas = spdiag(dual_areas)                                      # [V, V] Scalar
    dirac = inverse_face_areas * mesh.at_corners(edges) * 0.5              # [F, V] Point

    dirac_squared = ~dirac * face_areas * dirac                            # [V, V] Even
    squared, waves = (dirac_squared * Even).eigh(vertex_areas * Even, int(sizes[-1]))   # [modes] Scalar, [modes] Even[V]
    shares = [waves[low:high] for low, high in zip(sizes[:-1], sizes[1:])]
    chosen = mesh_module.stack([(share * (vertex_areas * share.reverse().scalar_product(probe)).sites.sum()).sum(axis=0)
                                for share in shares])                      # [levels] Even[V]
    vertices = chosen[:, None] * np.cos(phases)                            # [levels, phases] Even[V]
    faces = (dirac * chosen * (1 / frequencies))[:, None] * np.sin(phases) # [levels, phases] Even[F]
    return squared, vertices, faces
