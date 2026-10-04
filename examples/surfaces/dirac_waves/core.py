"""Waves of a spinor field on a surface, carried by the surface's Dirac operator, in the geometric
algebra of three-dimensional space.

The field is a quaternion on each vertex, an even multivector, and the Dirac operator of the spin
transformations takes it to an odd multivector on each face. The operator `D` is a sparse extensor: a
field of vectors with one cell for each face and each of its corners, the edge that corner faces over
minus twice the face's area. It acts by the geometric product, cell by cell and summed over each face's
corners, so `D * vertices` takes a vertex field `[V] Even` to a face field `[F] Odd`, the vector times
the quaternion. Its reverse `~D` reverses every cell and swaps faces for vertices, so it takes a face
field back to the vertices. Weighted by the areas, the two pair the same: summed over the faces, `M2 * faces.reverse().scalar_product(D * vertices)` equals, summed over the vertices,
`(~D * (M2 * faces)).reverse().scalar_product(vertices)`.

A wave alternates the two: the face field moved on by the Dirac operator of the vertex field, then the
vertex field moved back by the reverse of the face field, weighted by the areas, a leapfrog that keeps
the field's energy, the area-weighted squared size of both. Twice over the step is the operator
`Q = ~D * M2 * D` of the spin transformations, taking vertex fields to vertex fields. The product with
the open type, `Q * Even`, makes it a map `[V] Even <- [V] Even` that the eigensolver takes, so the
standing waves are its eigenfields against the vertex areas, `M0 * Even`; on the unit sphere their
frequencies are the whole numbers, each held eight times its own.

Lengths are in units of the sphere's radius and times in units of the radius over the waves' speed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.sparse import SparseExtensor
from examples.surfaces.spin_transformations.core import Mesh, as_diag, as_ga_sparse, as_scalar, context

mv = context.multivector
ga = context.algebra
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
Odd = ga.gatype.odd()


# --- math -----------------------------------------------------------------------------
def dirac(mesh: Mesh) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]:
    """The Dirac operator from vertices to faces, the face areas, and the vertex areas and their
    inverses."""
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = mesh.face_edge_orientation                                        # [F, 3]
    T10 = as_ga_sparse(I10, as_scalar(O10))                                 # [E, V] Scalar
    edges = T10 * mesh.vertices                                             # [E] Vector

    # diagonal operators: the triangle areas, and the vertex areas and their inverses
    M2 = as_diag(mesh.triangle_areas)                                       # [F, F] Scalar
    M2i = as_diag(1 / mesh.triangle_areas)                                  # [F, F] Scalar
    M0 = as_diag(mesh.vertex_areas)                                         # [V, V] Scalar
    M0i = as_diag(1 / mesh.vertex_areas)                                    # [V, V] Scalar

    # each face takes each corner by the edge it faces, over minus twice the face's area
    D = M2i * as_ga_sparse(I20, edges[I21] * O21) * -0.5                    # [F, V] Vector
    return D, M2, M0, M0i


def spread(mesh: Mesh, start: Even, interval: float, count: int) -> Iterator[tuple[Even, Odd]]:
    """A wave from the given vertex field: the vertex field at each step and the face field half a
    step after it."""
    D, M2, M0, M0i = dirac(mesh)
    vertices, faces = start, D * start * 0.0                                # [V] Even, [F] Odd
    for _ in range(count):
        # the face field moved on by the Dirac operator of the vertex field
        faces = faces + D * vertices * interval                             # [F] Odd
        yield vertices, faces
        # the vertex field moved back by the reverse of the face field, weighted by the areas
        vertices = vertices - M0i * (~D * (M2 * faces)) * interval          # [V] Even


def standing(mesh: Mesh, count: int) -> tuple[Scalar, Even]:
    """The count standing waves of least frequency: their frequencies squared and their vertex fields,
    the eigenfields of `~D * M2 * D` against the vertex areas."""
    D, M2, M0, _ = dirac(mesh)
    return ((~D * M2 * D) * Even).eigh(M0 * Even, count)                    # [count] Scalar, [count, V] Even


def oscillation(mesh: Mesh, waves: Even, frequencies: Scalar, phases: np.ndarray) -> tuple[Even, Odd]:
    """Standing waves at the given phases of their periods `[waves, phases, ...]`: the vertex field a
    cosine, and the face field, the Dirac operator of it over the frequency, a sine."""
    D, _, _, _ = dirac(mesh)
    vertices = waves[:, None] * np.cos(phases)[:, None]                     # [waves, phases, V] Even
    faces = (D * waves / frequencies[:, None])[:, None] * np.sin(phases)[:, None]   # [waves, phases, F] Odd
    return vertices, faces
