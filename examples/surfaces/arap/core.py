r"""As-rigid-as-possible surface deformation bends a triangle mesh to follow displaced handle vertices while keeping every local neighbourhood as close as possible to a rigid rotation of its rest shape.

The best rotor at a vertex comes from a quadratic form on rotors: `edges.dot(Rotor >> rest_edges)` is a form `Scalar <- (Rotor, Rotor)` satisfying `form(R, R) == edges.dot(R >> rest_edges)`, and summed over a vertex's edges like numbers its top eigenvector is the rotor, every vertex in one batched `eigh`, with no singular value decomposition and no correction for reflections. The best vertices then come from the rotation itself, `rotors >> Point`, a map averaged over each edge's two ends that does not depend on the sign the eigensolver gives each rotor. The mesh's operators are maps between fields on its vertices and edges, so the cotangent Laplacian is `~T10 * H1 * T10`, following the local-global iteration of Sorkine and Alexa (2007), https://igl.ethz.ch/projects/ARAP/arap_web.pdf.

In the matrix notation of Sorkine and Alexa the energy reads as $E = \sum_i \sum_{j \in N(i)} w_{ij} \|(\mathbf p'_i - \mathbf p'_j) - R_i (\mathbf p_i - \mathbf p_j)\|^2$, each rotation matrix $R_i$ comes from the singular value decomposition of the covariance $S_i = \sum_j w_{ij}\, \mathbf e_{ij} {\mathbf e'_{ij}}^{\mathsf T}$, and the vertices follow from one linear system with the cotangent Laplacian.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from numga.algebras import VGA3D
from numga.sparse import spdiag
from examples import instantiate
from examples.surfaces.spin_transformations.core import corner_cosines, cube

ga = VGA3D
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar = mesh_module.Mesh, mesh_module.as_scalar
mv = mesh_module.ctx.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Rotor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def deform(
    mesh: Mesh,
    rest_vertices: Point,
    handles: np.ndarray,
    poses: Iterable[Point],
    stiffness: float,
    n_iterations: int,
) -> Iterator[tuple[Point, Point, Scalar]]:
    """For each handle pose, the as-rigid-as-possible vertices after `n_iterations` alternating steps
    warm-started from the previous pose, the Laplacian-edited vertices, and the energy at each iteration."""
    # Incidence operators: oriented edge differences from tail to head, and midpoint averages across each edge.
    T10 = mesh.d[0]                                                                 # Scalar+[edges <- vertices]
    A10 = mesh.incidence[1, 0].mean(axis=0)                                         # Scalar+[edges <- vertices]

    rest_edges = T10 * rest_vertices                                                # Point-[edges]
    # The directed edge opposite each corner of each face, its corner cotangent from the inner product
    # over the spanned plane norm, and each edge's cotangent weight summed from its opposite corners.
    triangle_edges = mesh.boundary[2, 1] * rest_edges                               # [corners] Point-[faces]
    tails, heads = mesh.corners[2, 1]
    next_edges, prev_edges = triangle_edges[tails], triangle_edges[heads]           # [corners] Point-[faces], Point-[faces]
    cotangents = -(next_edges.dual_inverse() & prev_edges) / next_edges.commutator(prev_edges).norm()  # [corners] Scalar+[faces]
    cotangent_weights = (~mesh.incidence[2, 1] * (cotangents / 2)).sum(axis=0)     # Scalar+[edges]

    # Diagonal weights for the edge cotangents and the handle penalty stiffness.
    H1 = spdiag(cotangent_weights)                                                  # Scalar+[edges <- edges]
    P = spdiag(as_scalar(handles * stiffness).field())                              # Scalar+[vertices <- vertices]

    # The cotangent Laplacian on vertex positions with the handle penalty added.
    handle_laplacian = (~T10 * H1 * T10 + P) * Point                                # (Point- <- Point-)[vertices <- vertices]

    vertices = rest_vertices                                                        # Point-[vertices]
    for targets in poses:
        energies = []
        for _ in range(n_iterations):
            edges = T10 * vertices                                                  # Point-[edges]
            # Each vertex's optimal rotor is the top eigenvector of its incident edges' summed quadratic form.
            forms = ~A10 * H1 * edges.dot(Rotor >> rest_edges)                      # (Scalar+ <- (Rotor+, Rotor+))[vertices]
            _, modes = forms.batch().eigh()                                         # [vertices, modes] Rotor+
            rotors = modes[..., -1].field()                                         # Rotor+[vertices]
            # Average the endpoint rotation maps across each edge and apply the mean map to the rest edge.
            turned_edges = (A10 * (rotors >> Point))(rest_edges)                    # Point-[edges]
            # Because a rotor keeps lengths, an edge's two endpoint mismatches sum to 2 * (edge^2 + rest^2 - 2 * edge . turned).
            mismatch = H1 * (edges.norm_squared() + rest_edges.norm_squared() - 2 * edges.dot(turned_edges))  # Scalar+[edges]
            penalty = P * (vertices - targets).norm_squared()                       # Scalar+[vertices]
            energies.append(2 * mismatch.sites.sum() + penalty.sites.sum())
            # Solve for the vertex positions whose edges best match the rotated rest edges.
            vertices = handle_laplacian.solve(~T10 * H1 * turned_edges + P * targets)  # Point-[vertices]
        # Solve for the vertex positions whose edges best match the unrotated rest edges.
        laplacian_vertices = handle_laplacian.solve(~T10 * H1 * rest_edges + P * targets)  # Point-[vertices]
        yield vertices, laplacian_vertices, stack(energies)


# --- plumbing -------------------------------------------------------------------------
def bar(n_divisions: int, length: float) -> tuple[Mesh, Point]:
    """A closed square bar along x with cross-section width 2 and the given length, stretched from a subdivided cube."""
    box, corners = cube(n_divisions)
    return box, corners + (mv.x & corners) * mv.x.dual() * (length / 2 - 1)
