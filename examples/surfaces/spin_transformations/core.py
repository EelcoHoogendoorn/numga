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

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor

ga = Algebra("x+y+z+")
context = NumpyContext(ga)
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
Odd = ga.gatype.odd()


def as_scalar(v):
    return context.multivector.scalar(np.asarray(v)[..., None])


def as_ga_sparse(C, V):
    """The sparse linear map coupling each of the R output elements to the n input elements its row
    of C names, `[R, n]`, through the extensors V of the same shape."""
    return SparseExtensor.from_columns(C, V, int(C.max()) + 1)


as_diag = SparseExtensor.from_diagonal


def spin_transform_deform(mesh: Mesh, rho) -> Mesh:
    """The mesh after the spin transformation that changes each face's mean curvature by rho, `[F]
    Scalar`, with every operator a sparse linear map coupling elements through multivectors, nullary
    extensors, as the paper's quaternionic matrices do; the energy and the Laplacian are formed with
    their reverses, and the couplings become maps only for the solvers."""
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = mesh.face_edge_orientation                                        # [F, 3]

    # the boundary, taking each edge's tail from its head, and the means over each edge and each face
    T10 = as_ga_sparse(I10, as_scalar(O10))                                 # [E, V] Scalar
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))               # [E, V] Scalar
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))               # [F, V] Scalar

    # diagonal operators: the triangle areas, the vertex areas, and each edge's cotangent weight
    M2 = as_diag(mesh.triangle_areas)                                       # [F, F] Scalar
    M2i = as_diag(1 / mesh.triangle_areas)                                  # [F, F] Scalar
    M0 = as_diag(mesh.vertex_areas)                                         # [V, V] Scalar
    H1 = as_diag(mesh.edge_ratio)                                           # [E, E] Scalar

    edges = T10 * mesh.vertices                                             # [E] Vector
    L = ~T10 * H1 * T10                                                     # [V, V] Scalar

    # each face takes the vertex opposite each edge through that edge, oriented counter-clockwise
    D = M2i * as_ga_sparse(I20, edges[I21] * O21) * -0.5                    # [F, V] Vector
    R = as_diag(rho.dual()) * A20                                           # [F, V] Pseudoscalar
    A = D - R                                                               # [F, V] Odd
    Q = ~A * M2 * A                                                         # [V, V] Even

    # the field of least energy per unit of vertex area, divided by its area-weighted mean
    _, modes = (Q * Even).eigh(M0 * Even, 1)                                # [1, V] Even
    mean = (M0 * modes[0]).sum(axis=0) / mesh.vertex_areas.sum(axis=0)      # [] Even
    q = modes[0] / mean                                                     # [V] Even

    # each edge turned and scaled by the mean quaternion of its ends, and the vertices that match them best
    transformed_edges = (A10 * q) << edges                                  # [E] Vector
    b = ~T10 * H1 * transformed_edges                                       # [V] Vector
    return mesh.copy(vertices=(L * Vector).lstsq(b))


def spin_transform_deform_maps(mesh: Mesh, rho) -> Mesh:
    """The same transformation with every operator in the real form the paper gives for numerical
    packages throughout: each quaternion entry as a map between the fields' elements, and the energy
    and the Laplacian formed with the maps' adjoints in the metric of the reverse's scalar product."""
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = mesh.face_edge_orientation                                        # [F, 3]

    # the boundary, taking each edge's tail from its head, and the means over each edge and each face
    T10 = as_ga_sparse(I10, as_scalar(O10) * Vector)                        # [E] Vector <- [V] Vector
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2) * Even)        # [E] Even <- [V] Even
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3) * Even)        # [F] Even <- [V] Even

    # diagonal operators: the triangle areas, the vertex areas, and each edge's cotangent weight
    M2 = as_diag(mesh.triangle_areas * Odd)                                 # [F] Odd <- [F] Odd
    M2i = as_diag(1 / mesh.triangle_areas * Odd)                            # [F] Odd <- [F] Odd
    M0 = as_diag(mesh.vertex_areas * Even)                                  # [V] Even <- [V] Even
    H1 = as_diag(mesh.edge_ratio * Vector)                                  # [E] Vector <- [E] Vector

    edges = T10(mesh.vertices)                                              # [E] Vector
    L = metric_adjoint(T10)(H1(T10))                                        # [V] Vector <- [V] Vector

    # each face takes the vertex opposite each edge through that edge, oriented counter-clockwise
    D = M2i(as_ga_sparse(I20, edges[I21] * O21 * Even)) * -0.5              # [F] Odd <- [V] Even
    R = as_diag(rho.dual() * Even)(A20)                                     # [F] Odd <- [V] Even
    A = D - R                                                               # [F] Odd <- [V] Even
    Q = metric_adjoint(A)(M2(A))                                            # [V] Even <- [V] Even

    # the field of least energy per unit of vertex area, divided by its area-weighted mean
    _, modes = Q.eigh(M0, 1)                                                # [1, V] Even
    mean = M0(modes[0]).sum(axis=0) / mesh.vertex_areas.sum(axis=0)         # [] Even
    q = modes[0] / mean                                                     # [V] Even

    # each edge turned and scaled by the mean quaternion of its ends, and the vertices that match them best
    transformed_edges = A10(q) << edges                                     # [E] Vector
    b = metric_adjoint(T10)(H1(transformed_edges))                          # [V] Vector
    return mesh.copy(vertices=L.lstsq(b))


def dirac_spheres(mesh: Mesh, eigenvalue: int, count: int):
    """The energies of the count least-energy fields for a constant rho, the given eigenvalue, and
    the surfaces they spin the mesh into.

    With rho constant, the fields of zero energy are the eigenfields of the Dirac operator itself: on
    the unit sphere the spinor spherical harmonics, the states of an electron in a spherically
    symmetric potential, and the surfaces are the Dirac spheres. Their eigenvalues are the integers
    but -1, each with multiplicity its value plus one. The fields are taken as they come, at unit
    size per unit of vertex area: their mean vanishes.
    """
    # the face-vertex, face-edge and edge-vertex incidences, and the relative orientations
    I20, I21, I10 = mesh.faces, mesh.face_edges, mesh.edges                 # [F, 3], [F, 3], [E, 2]
    O10 = np.ones_like(I10) * [-1, 1]                                       # [E, 2]
    O21 = mesh.face_edge_orientation                                        # [F, 3]

    # the boundary, taking each edge's tail from its head, and the means over each edge and each face
    T10 = as_ga_sparse(I10, as_scalar(O10))                                 # [E, V] Scalar
    A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))               # [E, V] Scalar
    A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))               # [F, V] Scalar

    # diagonal operators: the triangle areas, the vertex areas, and each edge's cotangent weight
    M2 = as_diag(mesh.triangle_areas)                                       # [F, F] Scalar
    M2i = as_diag(1 / mesh.triangle_areas)                                  # [F, F] Scalar
    M0 = as_diag(mesh.vertex_areas)                                         # [V, V] Scalar
    H1 = as_diag(mesh.edge_ratio)                                           # [E, E] Scalar

    edges = T10 * mesh.vertices                                             # [E] Vector
    L = ~T10 * H1 * T10                                                     # [V, V] Scalar

    # each face takes the vertex opposite each edge through that edge, oriented counter-clockwise
    rho = as_scalar(np.full(len(I20), float(eigenvalue)))                   # [F] Scalar
    D = M2i * as_ga_sparse(I20, edges[I21] * O21) * -0.5                    # [F, V] Vector
    R = as_diag(rho.dual()) * A20                                           # [F, V] Pseudoscalar
    A = D - R                                                               # [F, V] Odd
    Q = ~A * M2 * A                                                         # [V, V] Even

    # the fields of least energy per unit of vertex area, and the surfaces whose edges match theirs
    energies, fields = (Q * Even).eigh(M0 * Even, count)                    # [count] Scalar, [count, V] Even
    spheres = [mesh.copy(vertices=(L * Vector).lstsq(~T10 * H1 * ((A10 * q) << edges))) for q in fields]
    return energies, spheres


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
    Hn = L * mesh.vertices                              # [V] Vector
    h_integrated = Hn | mesh.vertex_normals             # [V] Scalar
    h = h_integrated / (2 * mesh.vertex_areas)          # [V] Scalar
    return A20 * h                                      # [F] Scalar


def _recenter(mesh: Mesh):
    """Remove the translation/scale gauge freedom the flow leaves undetermined."""
    v = mesh.vertices - mesh.vertices.mean(axis=0)
    return mesh.copy(vertices=v / v.norm().mean(axis=0))


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
        h_mean = (h * mesh.triangle_areas).sum(axis=0) / mesh.triangle_areas.sum(axis=0)
        mesh = _recenter(spin_transform_deform(mesh, -rate * (h - h_mean)))
        yield mesh


# --- plumbing -------------------------------------------------------------------------
def metric_adjoint(operator: SparseExtensor) -> SparseExtensor:
    """The map with inputs and outputs swapped and each map cell replaced by its adjoint under the
    scalar product of one field with the reverse of another: that pairing on the cells' input,
    solved against the pairing on their output."""
    output, input = operator.cells.axes
    pairing = ga.operator.scalar_norm_squared
    cells = pairing(input).solve(pairing(output)(ga.gatype(output), operator.cells))
    return SparseExtensor(cells, operator.columns, operator.rows, operator.shape[::-1])


class Mesh:
    """A closed, oriented triangle mesh: vertices `[V] Vector` and faces `[F, 3]`, counter-clockwise
    from outside, with its edges `[E, 2]` from lower to higher vertex, the edge facing each corner of
    each face `[F, 3]`, and that edge's orientation relative to the face."""

    def __init__(self, vertices: Vector, faces: np.ndarray) -> None:
        corner = np.arange(3)
        # the ends of the edge facing each corner, counter-clockwise
        facing = np.stack([faces[:, (corner + 1) % 3], faces[:, (corner + 2) % 3]], axis=-1)   # [F, 3, 2]
        self.vertices, self.faces = vertices, faces
        self.edges, face_edges = np.unique(np.sort(facing, axis=-1).reshape(-1, 2), axis=0, return_inverse=True)
        self.face_edges = face_edges.reshape(-1, 3)
        self.face_edge_orientation = np.where(facing[..., 0] < facing[..., 1], 1.0, -1.0)

    def copy(self, vertices: Vector) -> Mesh:
        return Mesh(vertices, self.faces)

    @property
    def triangle_edges(self) -> Vector:
        """[F, 3] the edge facing each corner, counter-clockwise."""
        corner = np.arange(3)
        return self.vertices[self.faces[:, (corner + 2) % 3]] - self.vertices[self.faces[:, (corner + 1) % 3]]

    @property
    def triangle_areas(self):
        edges = self.triangle_edges
        return (edges[:, 0] ^ edges[:, 1]).norm() / 2

    @property
    def edge_ratio(self):
        """[E] dual over primal edge length: half the cotangents of the angles facing each edge."""
        edges = self.triangle_edges
        after, before = edges[:, [1, 2, 0]], edges[:, [2, 0, 1]]
        # the cotangent at each corner
        cotangents = -(after | before) / (after ^ before).norm()               # [F, 3] Scalar
        return ~as_ga_sparse(self.face_edges, cotangents / 2) * as_scalar(np.ones(len(self.faces)))

    @property
    def vertex_normals(self) -> Vector:
        edges = self.triangle_edges
        # each face's normal, twice its area long
        face_normals = (edges[:, 0] ^ edges[:, 1]).dual()                     # [F] Vector
        spread = as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces, dtype=float)))
        return (~spread * face_normals).normalized()

    @property
    def vertex_areas(self):
        """[V] a third of the area of each triangle at the vertex."""
        return ~as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces) / 3)) * self.triangle_areas

    def corner_cosines(self):
        """[F, 3] the cosine of each corner's angle, kept exactly by a conformal map."""
        edges = self.triangle_edges
        after, before = edges[:, [1, 2, 0]], edges[:, [2, 0, 1]]
        return -(after | before) / (after.norm() * before.norm())


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
    return Mesh(context.multivector.vector(coords / np.linalg.norm(coords, axis=-1, keepdims=True)), faces)


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
    return Mesh(context.multivector.vector(unique), inverse.reshape(-1)[np.concatenate(faces)])
