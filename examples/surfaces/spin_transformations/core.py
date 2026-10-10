"""Keenan Crane's spin transformations, in geometric algebra.

https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/paper.pdf

A spin transformation changes a surface's mean curvature by a prescribed amount rho while keeping
every angle: a quaternion at each vertex turns and scales the edges around it. The quaternions are
the least eigenfield of the Dirac operator less rho, and the new vertices are those whose edges
best match the turned ones. In Cl(3,0) the quaternions are the even subalgebra, the edges vectors,
and rho times the pseudoscalar a trivector; the operators are sparse linear maps between fields on
the vertices, edges and faces, coupling their elements through extensors, here multivectors.
"""

from __future__ import annotations

import numpy as np

from numga.sparse import SparseExtensor
from numga.extensor import Extensor
from examples.mesh import Mesh, as_diag, as_ga_sparse, as_scalar, at_sites, context, ga, Scalar, Vector, Bivector

Even = ga.gatype.even()


def spin_transform_deform(mesh: Mesh, rho) -> Mesh:
    """The mesh after the spin transformation that changes each face's mean curvature by rho,
    `Scalar[F]`, with every operator a sparse linear map coupling elements through multivectors, nullary
    extensors, as the paper's quaternionic matrices do; the energy and the Laplacian are formed with
    their reverses, and the couplings become maps only for the solvers."""
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = as_scalar(mesh.face_edge_orientation.T).field()                   # [3] Scalar[F]

    # the boundary, taking each edge's tail from its head, and the means over each edge and each face
    T10 = as_ga_sparse(I10, as_scalar(O10))                                 # [E, V] Scalar
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))               # [E, V] Scalar
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))               # [F, V] Scalar

    # diagonal operators: the triangle areas, the vertex areas, and each edge's cotangent weight
    M2 = as_diag(mesh.triangle_areas)                                       # [F, F] Scalar
    M2i = as_diag(1 / mesh.triangle_areas)                                  # [F, F] Scalar
    M0 = as_diag(mesh.vertex_areas)                                         # [V, V] Scalar
    H1 = as_diag(mesh.edge_ratio)                                           # [E, E] Scalar

    edges = T10 * mesh.vertices                                             # Vector[E]
    L = ~T10 * H1 * T10                                                     # [V, V] Scalar

    # each face takes each corner by the edge it faces, over minus twice the face's area: the geometric
    # derivative divided by each face's plane
    D = M2i * at_corners(mesh, at_sites(edges, I21.T) * O21) * -0.5         # [F, V] Vector
    R = as_diag(rho.dual()) * A20                                           # [F, V] Pseudoscalar
    A = D - R                                                               # [F, V] Odd
    Q = ~A * M2 * A                                                         # [V, V] Even

    # the field of least energy per unit of vertex area, divided by its area-weighted mean
    _, modes = (Q * Even).eigh(M0 * Even, 1)                                # [1] Even[V]
    mean = (M0 * modes[0]).batch().sum(axis=-1) / mesh.vertex_areas.batch().sum(axis=-1)   # [] Even
    q = modes[0] / mean                                                     # Even[V]

    # each edge turned and scaled by the field along it, integrated by Simpson's rule: the turns at its
    # two ends, averaged as maps, and twice the turn at its middle; and the vertices that match them best
    transformed_edges = ((A10 * (q << Vector))(edges) + 2 * ((A10 * q) << edges)) / 3   # Vector[E]
    b = ~T10 * H1 * transformed_edges                                       # Vector[V]
    return mesh.copy(vertices=(L * Vector).lstsq(b))


def dirac_spheres(mesh: Mesh, eigenvalue: int, count: int):
    """The energies of the count least-energy fields for a constant rho, the given eigenvalue, and
    the surfaces they spin the mesh into.

    With rho constant, the fields of zero energy are the eigenfields of the Dirac operator itself: on
    the unit sphere the spinor spherical harmonics, and the surfaces are the Dirac spheres. Their eigenvalues are the integers
    but -1, each with multiplicity its value plus one. The fields are taken as they come, at unit
    size per unit of vertex area: their mean vanishes.
    """
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = as_scalar(mesh.face_edge_orientation.T).field()                   # [3] Scalar[F]

    # the boundary, taking each edge's tail from its head, and the means over each edge and each face
    T10 = as_ga_sparse(I10, as_scalar(O10))                                 # [E, V] Scalar
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))               # [E, V] Scalar
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))               # [F, V] Scalar

    # diagonal operators: the triangle areas, the vertex areas, and each edge's cotangent weight
    M2 = as_diag(mesh.triangle_areas)                                       # [F, F] Scalar
    M2i = as_diag(1 / mesh.triangle_areas)                                  # [F, F] Scalar
    M0 = as_diag(mesh.vertex_areas)                                         # [V, V] Scalar
    H1 = as_diag(mesh.edge_ratio)                                           # [E, E] Scalar

    edges = T10 * mesh.vertices                                             # Vector[E]
    L = ~T10 * H1 * T10                                                     # [V, V] Scalar

    # each face takes each corner by the edge it faces, over minus twice the face's area
    rho = as_scalar(np.full(len(I20), float(eigenvalue))).field()           # Scalar[F]
    D = M2i * at_corners(mesh, at_sites(edges, I21.T) * O21) * -0.5         # [F, V] Vector
    R = as_diag(rho.dual()) * A20                                           # [F, V] Pseudoscalar
    A = D - R                                                               # [F, V] Odd
    Q = ~A * M2 * A                                                         # [V, V] Even

    # the fields of least energy per unit of vertex area, and the surfaces whose edges match theirs
    energies, fields = (Q * Even).eigh(M0 * Even, count)                    # [count] Scalar, [count] Even[V]
    turned = ((A10 * (fields << Vector))(edges) + 2 * ((A10 * fields) << edges)) / 3   # [count] Vector[E]
    spheres = [mesh.copy(vertices=vertices) for vertices in (L * Vector).lstsq(~T10 * H1 * turned)]
    return energies, spheres


def geometric_derivative(mesh: Mesh) -> SparseExtensor:
    """The geometric derivative of fields on the vertices, linear on each face, `[F, V] Vector`: on
    each face, each corner's coupling is the gradient of its hat function, the edge facing it turned
    a quarter turn in the face's plane, over twice the face's area. Divided by each face's plane, it
    is the Dirac operator."""
    turned = mesh.triangle_edges | mesh.face_planes                         # [3] Vector[F]
    return at_corners(mesh, turned / (2 * mesh.triangle_areas))


def mean_curvature(mesh: Mesh):
    """Pointwise signed mean curvature per face.

    The cotan-laplacian of the positions gives the *area-integrated* mean
    curvature normal `L x = 2 A_v H n` at each vertex; dividing by twice the
    vertex (barycentric) area recovers the pointwise `H`, which we then average
    onto faces (the domain `rho` lives on).
    """
    I20, I10 = mesh.faces, mesh.edges
    T10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) * [-1, 1]))
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))
    H1 = as_diag(mesh.edge_ratio)

    L = ~T10 * H1 * T10                                 # [V, V] Scalar
    # the integrated mean-curvature normal, its signed size at each vertex, the pointwise
    # curvature at each vertex, and its mean over each face
    Hn = L * mesh.vertices                              # Vector[V]
    h_integrated = Hn | mesh.vertex_normals             # Scalar[V]
    h = h_integrated / (2 * mesh.vertex_areas)          # Scalar[V]
    return A20 * h                                      # Scalar[F]


def _recenter(mesh: Mesh):
    """Remove the translation/scale gauge freedom the flow leaves undetermined."""
    v = mesh.vertices - mesh.vertices.batch().mean(axis=-1)
    return mesh.copy(vertices=v / v.norm().batch().mean(axis=-1))


def conformal_smooth(mesh: Mesh, iterations: int, rate: float):
    """Conformal curvature flow via iterated spin transforms.

    Each step prescribes `rho = -rate * (H - Hbar)`: a conformal *change* in mean
    curvature that shaves off each face's deviation from the (area-weighted) mean.
    This relaxes the surface toward constant mean curvature while every individual
    step stays conformal (angle-preserving). Run on a cube it rounds the edges and
    corners, which stay cone points: a conformal map keeps the angle deficit there.

    Note `rho` is a *change* in curvature, so `rho = 0` (rate 0) is the identity;
    feeding absolute curvature instead diverges. Yields the mesh before each step
    and after the last.
    """
    mesh = _recenter(mesh)
    yield mesh
    for _ in range(iterations):
        h = mean_curvature(mesh)
        h_mean = (h * mesh.triangle_areas).batch().sum(axis=-1) / mesh.triangle_areas.batch().sum(axis=-1)
        mesh = _recenter(spin_transform_deform(mesh, -rate * (h - h_mean)))
        yield mesh


# --- plumbing -------------------------------------------------------------------------
def at_corners(mesh: Mesh, cells: Extensor) -> SparseExtensor:
    """[F, V]: couplings from each face to the vertex at each of its corners, through `[3]` face
    fields of cells, one per corner."""
    return SparseExtensor.from_indices(cells.batch(), np.arange(len(mesh.faces)), mesh.faces.T,
                                       (len(mesh.faces), len(mesh.vertices.batch())))


def icosphere(levels: int) -> Mesh:
    """A unit sphere: an icosahedron with each face split in four, the given number of times."""
    golden = (1 + 5 ** 0.5) / 2
    coords = np.array([
        [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
        [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
        [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
    ])
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
        [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
        [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    for _ in range(levels):
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        (a, b, c), (ab, bc, ca) = faces.T, (len(coords) + inverse.reshape(3, -1))
        coords = np.concatenate([coords, (coords[unique[:, 0]] + coords[unique[:, 1]]) / 2])
        faces = np.concatenate([np.stack(f, axis=-1) for f in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return Mesh(context.multivector.vector(coords / np.linalg.norm(coords, axis=-1, keepdims=True)).field(), faces)


def cube(divisions: int) -> Mesh:
    """The surface of the cube [-1, 1]^3, each side a grid of divisions by divisions squares."""
    grid = np.linspace(-1, 1, divisions + 1)
    u, v = (axis.ravel() for axis in np.meshgrid(grid, grid, indexing="ij"))
    square = np.arange(divisions)[:, None] * (divisions + 1) + np.arange(divisions)[None, :]
    a, b, c, d = ((square + offset).ravel() for offset in (0, divisions + 1, divisions + 2, 1))
    side = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)])
    coords, faces = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            point = np.zeros((len(u), 3))
            point[:, axis], point[:, (axis + 1) % 3], point[:, (axis + 2) % 3] = sign, u * sign, v
            faces.append(side + len(coords) * len(u))
            coords.append(point)
    unique, inverse = np.unique(np.round(np.concatenate(coords), 9), axis=0, return_inverse=True)
    return Mesh(context.multivector.vector(unique).field(), inverse.reshape(-1)[np.concatenate(faces)])
