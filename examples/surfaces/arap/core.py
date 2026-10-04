"""As rigid as possible: a surface bent by moving a few of its vertices, every neighbourhood of it
kept as close to a turn of its rest shape as it can be, in the geometric algebra of three-dimensional
space.

The surface's energy is the cotangent-weighted squared mismatch, over every edge at each of its two
ends, between the edge as it is and the rest edge turned by a rotor of that end's own. A rotor keeps
lengths, so an edge's two mismatches add up to `2 * (edge**2 + rest**2 - 2 * edge | turned)`, with
`turned` the rest edge turned by the mean of its two ends' turns. The energy falls in two alternating
steps. With the vertices held, each vertex's best rotor turns its rest edges onto its
edges as they are: the rotor `R` that makes the most of `edge | (R >> rest)`, a form with the rotor
open twice, `edge | (Even >> rest)`, summed over the vertex's edges, and largest at its top
eigenvector. With the rotors held, the best vertices match every edge to the mean of its rest edge
turned by its two ends' rotors: one sparse Poisson solve with the cotangent Laplacian, its right side
`~T10 * H1 * turned`, as in the last step of a spin transformation. The handles are held to where
they are moved by a stiff penalty in the same solve.

Laplacian editing is the same solve without the turns: every rest edge matched as it is, so the
surface shears and shrinks where it is bent.

In the notation of Sorkine and Alexa the energy reads as
$E = \\sum_i \\sum_{j \\in N(i)} w_{ij} \\|(p'_i - p'_j) - R_i (p_i - p_j)\\|^2$, the rotor's form as
the trace of $R_i S_i$ with the covariance $S_i = \\sum_j w_{ij} e_{ij} {e'_{ij}}^\\top$.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.surfaces.spin_transformations.core import Mesh, as_diag, as_ga_sparse, as_scalar, context, cube

mv = context.multivector
ga = context.algebra
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
def deform(mesh: Mesh, handles: np.ndarray, poses: Iterable[Vector], stiffness: float,
           iterations: int) -> Iterator[tuple[Vector, Vector, Scalar]]:
    """For each pose of the handles, the targets of every vertex `[V] Vector` of which the handles' are
    held: the mesh as rigid as possible after the given iterations, each pose starting from the last
    one's; the mesh by Laplacian editing; and the energy at each iteration `[iterations] Scalar`."""
    # the boundary, taking each edge's tail from its head, and the mean over each edge
    I10 = mesh.edges                                                        # [E, 2]
    T10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) * [-1, 1]))        # [E, V] Scalar
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))              # [E, V] Scalar

    # diagonal operators: each edge's cotangent weight, and the handles' stiffness
    H1 = as_diag(mesh.edge_ratio)                                           # [E, E] Scalar
    P = as_diag(as_scalar(handles * stiffness))                             # [V, V] Scalar

    rest = T10 * mesh.vertices                                              # [E] Vector
    # the cotangent Laplacian, with the handles held
    held = (~T10 * H1 * T10 + P) * Vector                                   # [V] Vector <- [V] Vector

    vertices = mesh.vertices                                                # [V] Vector
    for targets in poses:
        energies = []
        for _ in range(iterations):
            edges = T10 * vertices                                          # [E] Vector
            # each vertex's best rotor: the top eigenvector of its edges' summed form
            forms = ~A10 * H1 * (edges | (Even >> rest))                    # [V] Scalar <- (Even, Even)
            rotors = forms.eigh()[1][..., -1]                               # [V] Even
            # each rest edge turned by its two ends' rotors, the turns averaged
            turned = (A10 * (rotors >> Vector))(rest)                       # [E] Vector
            mismatch = H1 * (edges.norm_squared() + rest.norm_squared() - 2 * (edges | turned))   # [E] Scalar
            energies.append(2 * mismatch.sum(axis=0) + (P * (vertices - targets).norm_squared()).sum(axis=0))
            # the vertices whose edges best match the turned ones
            vertices = held.solve(~T10 * H1 * turned + P * targets)         # [V] Vector
        # the vertices whose edges best match the rest edges as they are
        laplacian = held.solve(~T10 * H1 * rest + P * targets)              # [V] Vector
        yield vertices, laplacian, stack(energies)


# --- plumbing -------------------------------------------------------------------------
def bar(divisions: int, length: float) -> Mesh:
    """A closed bar along x, its square cross section 2 across and its length the given one, each side
    of the cube it is stretched from a grid of divisions by divisions squares."""
    box = cube(divisions)
    return box.copy(vertices=box.vertices + (box.vertices | mv.x) * mv.x * (length / 2 - 1))
