"""Keenan Crane's spin transformations, in geometric algebra.

https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/paper.pdf

A spin transformation changes a surface's mean curvature by a prescribed amount while keeping every
angle: a quaternion at each vertex turns and scales the edges around it. The quaternions are the field
of least energy, the area-weighted squared residual of the Dirac operator less the curvature change, and
the new vertices are those whose edges best match the turned ones. In the algebra of space the
quaternions are the even multivectors, and the vertices and edges are points, antivectors, intrinsic,
which in three dimensions are bivectors. The Dirac operator's couplings are bivectors, the curvature
change a scalar against the sense the surface turns, and the operators are sparse linear maps between
fields on the vertices, edges and faces, coupling their elements through multivectors.
"""

from __future__ import annotations


import numpy as np

from numga.sparse import SparseExtensor, spdiag
from numga.extensor import Extensor
from numga.algebras import VGA3D as ga
from examples import MODEL_DIR
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, Scalar, Point = mesh_module.Mesh, mesh_module.Scalar, mesh_module.Point
Vector, Bivector = ga.gatype.vector(), ga.gatype.bivector()
as_ga_sparse, as_scalar, context = mesh_module.as_ga_sparse, mesh_module.as_scalar, mesh_module.ctx

Even = ga.gatype.even()
# Spot the cow, from the shared models, and the two quarter turns that stand her up facing along x: her y
# axis, up in the file, turned onto z, and then x onto y.
SPOT = MODEL_DIR / "spot.obj"
UPRIGHT = ((context.multivector.x ^ context.multivector.y) * (-np.pi / 4)).exp() * ((context.multivector.y ^ context.multivector.z) * (-np.pi / 4)).exp()


# --- math -----------------------------------------------------------------------------
def cotangent_weights(mesh: Mesh, sides: Point) -> Scalar:
    """`Scalar[E]`: each edge's cotangent weight from the edge facing each corner `[3] Point[F]`, half the cotangents
    of the angles facing it; a corner's cotangent is one edge's slice read along the other, over the size of the
    plane they span."""
    after, before = sides[[1, 2, 0]], sides[[2, 0, 1]]                         # [3] Point[F] each
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar[F]
    return ~mesh.by_corner(cotangents / 2) * as_scalar(np.ones(len(mesh.faces))).field()


def sense_of_space(mesh: Mesh, vertices: Point) -> Scalar:
    """`Scalar`: the sense the counter-clockwise faces give space, plus or minus one: the faces' corners meeting
    about the origin, summed over the solid, over its size."""
    a, b, c = vertices.batch()[mesh.faces.T].field()                           # Point[F] each
    volume = (a & b & c).sites.sum()                                           # [] Scalar
    return volume / volume.norm()


def corner_cosines(mesh: Mesh, vertices: Point) -> Scalar:
    """`[3] Scalar[F]`: the cosine of each corner's angle, kept exactly by a conformal map: one edge's slice read along
    the other, over both their lengths."""
    sides = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    after, before = sides[[1, 2, 0]], sides[[2, 0, 1]]                         # [3] Point[F] each
    return -(after.dual_inverse() & before) / (after.norm() * before.norm())


def spin_transform_deform(mesh: Mesh, vertices: Point, curvature_change: Scalar) -> Point:
    """The vertices after the spin transformation that changes each face's mean curvature by
    `curvature_change`, `Scalar[F]`: the field of least energy per unit of vertex area, divided by its
    area-weighted mean, turning the edges."""
    edge_means = as_ga_sparse(mesh.edges, as_scalar(np.ones_like(mesh.edges) / 2))   # [E, V] Scalar
    face_means = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    edges = mesh.d0 * vertices                                                 # Point[E]
    sides = (edges.batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar[F]
    face_areas, vertex_areas = spdiag(areas), spdiag(~face_means * areas)      # [F, F], [V, V] Scalar
    edge_weights = spdiag(cotangent_weights(mesh, sides))                      # [E, E] Scalar
    laplacian = ~mesh.d0 * edge_weights * mesh.d0                              # [V, V] Scalar

    residual = mesh.at_corners(sides / (2 * areas) - curvature_change * sense_of_space(mesh, vertices) / 3)   # [F, V] Even
    _, (field,) = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, 1)   # Even[V]
    quaternions = field / ((vertex_areas * field).sites.sum() / areas.sites.sum())   # Even[V]
    moved = ((edge_means * (quaternions << Point))(edges) + 2 * ((edge_means * quaternions) << edges)) / 3   # Point[E]
    return (laplacian * Point).lstsq(~mesh.d0 * edge_weights * moved)


def dirac_spheres(mesh: Mesh, vertices: Point, eigenvalues: np.ndarray, count: int) -> tuple[Scalar, Point]:
    """The energies of the count least-energy fields for each constant curvature change, the eigenvalues
    `[eigenvalues]`, and the vertices of the surfaces they spin the mesh into `[eigenvalues, count] Point[V]`.

    With the change constant, the fields of zero energy are the eigenfields of the Dirac operator itself:
    on the unit sphere the spinor spherical harmonics, and the surfaces are the Dirac spheres. Their
    eigenvalues are the integers but -1, each with multiplicity its value plus one. The fields are taken
    as they come, at unit size per unit of vertex area: their mean vanishes.
    """
    edge_means = as_ga_sparse(mesh.edges, as_scalar(np.ones_like(mesh.edges) / 2))   # [E, V] Scalar
    face_means = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    edges = mesh.d0 * vertices                                                 # Point[E]
    sides = (edges.batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar[F]
    face_areas, vertex_areas = spdiag(areas), spdiag(~face_means * areas)      # [F, F], [V, V] Scalar
    edge_weights = spdiag(cotangent_weights(mesh, sides))                      # [E, E] Scalar
    laplacian = ~mesh.d0 * edge_weights * mesh.d0                              # [V, V] Scalar

    constant = as_scalar(eigenvalues[:, None]) * sense_of_space(mesh, vertices) / 3   # [eigenvalues, 1] Scalar
    residual = mesh.at_corners(sides / (2 * areas) - constant)                 # [eigenvalues] [F, V] Even
    energies, fields = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, count)   # [eigenvalues, count] Even[V]
    moved = ((edge_means * (fields << Point))(edges) + 2 * ((edge_means * fields) << edges)) / 3   # [eigenvalues, count] Point[E]
    return energies, (laplacian * Point).lstsq(~mesh.d0 * edge_weights * moved)


def geometric_derivative(mesh: Mesh, vertices: Point) -> SparseExtensor:
    """The geometric derivative of fields on the vertices, linear on each face, `[F, V] Vector`: on
    each face, each corner's coupling is the gradient of its hat function, the edge facing it paired with
    the face's unit normal, over twice the face's area."""
    sides = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar[F]
    normals = (sides[0] & sides[1]) / (2 * areas)                              # Vector[F]
    return mesh.at_corners((sides | normals) / (2 * areas))


def mean_curvature(mesh: Mesh, vertices: Point) -> Scalar:
    """Pointwise signed mean curvature per face.

    The cotan-laplacian of the positions gives the *area-integrated* mean
    curvature normal `L x = 2 A_v H n` at each vertex; dividing by twice the
    vertex (barycentric) area recovers the pointwise `H`, which we then average
    onto faces (the domain the curvature change lives on).
    """
    face_means = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    sides = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar[F]
    laplacian = ~mesh.d0 * spdiag(cotangent_weights(mesh, sides)) * mesh.d0    # [V, V] Scalar
    # each vertex's normal: its faces' normals, each twice the face's area long, summed and made unit
    normals = (~face_means * (sides[0] & sides[1])).normalized()               # Vector[V]
    # the integrated mean-curvature normal, a displacement at each vertex, met with the vertex's normal:
    # its signed size, read against the sense the faces give space
    integrated = ((laplacian * vertices) & normals) * sense_of_space(mesh, vertices)   # Scalar[V]
    return face_means * (integrated / (2 * (~face_means * areas)))             # Scalar[F]


def _recenter(vertices: Point) -> Point:
    """Remove the translation/scale gauge freedom the flow leaves undetermined."""
    v = vertices - vertices.sites.mean()
    return v / v.norm().sites.mean()


def conformal_smooth(mesh: Mesh, vertices: Point, iterations: int, rate: float):
    """Conformal curvature flow via iterated spin transforms.

    Each step prescribes `rho = -rate * (H - Hbar)`: a conformal *change* in mean
    curvature that shaves off each face's deviation from the (area-weighted) mean.
    This relaxes the surface toward constant mean curvature while every individual
    step stays conformal (angle-preserving). Run on Spot the cow it draws in the legs,
    horns and ears and rounds the body into a sphere, every square of a checkerboard on
    the cow staying square.

    Note `rho` is a *change* in curvature, so `rho = 0` (rate 0) is the identity;
    feeding absolute curvature instead diverges. Yields the vertices before each step
    and after the last.
    """
    vertices = _recenter(vertices)
    yield vertices
    for _ in range(iterations):
        h = mean_curvature(mesh, vertices)
        sides = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point[F]
        areas = sides[1].commutator(sides[0]).norm() / 2                       # Scalar[F]
        h_mean = (h * areas).sites.sum() / areas.sites.sum()
        vertices = _recenter(spin_transform_deform(mesh, vertices, -rate * (h - h_mean)))
        yield vertices


# --- plumbing -------------------------------------------------------------------------
icosphere = Mesh.icosphere


def cube(divisions: int) -> tuple[Mesh, Point]:
    """The surface of the cube [-1, 1]^3, each side a grid of divisions by divisions squares, and its vertices."""
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
    return Mesh(len(unique), inverse.reshape(-1)[np.concatenate(faces)]), context.multivector.antivector(unique).field()


def spot() -> tuple[Mesh, Point]:
    """Spot the cow, by Keenan Crane, from his model repository (CC0),
    https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/: her mesh, and her vertices upright with z up,
    centred, and scaled to reach one at her furthest."""
    mesh, vertices = Mesh.read(SPOT, 2.0)
    return mesh, UPRIGHT >> vertices