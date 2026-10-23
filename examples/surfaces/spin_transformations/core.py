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

from pathlib import Path

import numpy as np

from numga.sparse import SparseExtensor, spdiag
from numga.extensor import Extensor
from numga.algebras import VGA3D as ga
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, Scalar, Vector, Bivector, Point = (mesh_module.Mesh, mesh_module.Scalar, mesh_module.Vector,
                                         mesh_module.Bivector, mesh_module.Point)
as_ga_sparse, as_scalar, context = mesh_module.as_ga_sparse, mesh_module.as_scalar, mesh_module.ctx

Even = ga.gatype.even()
# Spot the cow, from the shared models, and the two quarter turns that stand her up facing along x: her y
# axis, up in the file, turned onto z, and then x onto y.
SPOT = Path(__file__).parents[2] / "models" / "spot.obj"
UPRIGHT = ((context.multivector.x ^ context.multivector.y) * (-np.pi / 4)).exp() * ((context.multivector.y ^ context.multivector.z) * (-np.pi / 4)).exp()


def energy(mesh: Mesh, curvature_change: Scalar) -> SparseExtensor:
    """`[V, V] Even`: the area-weighted squared residual of the equation the transformation's quaternions
    solve, `dirac * field == potential * field`, for a change in each face's mean curvature `Scalar[F]`."""
    face_means = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # [F, V] Scalar
    face_areas, inverse_face_areas = spdiag(mesh.triangle_areas), spdiag(1 / mesh.triangle_areas)   # [F, F] Scalar
    # each face takes each corner by the edge it faces, over twice the face's area
    dirac = inverse_face_areas * mesh.at_corners(mesh.triangle_edges) * 0.5    # [F, V] Point
    # the curvature change, against the sense the faces give space, on each face's mean
    potential = spdiag(curvature_change * mesh.unit_volume) * face_means       # [F, V] Scalar
    residual = dirac - potential                                               # [F, V] Even
    return ~residual * face_areas * residual                                   # [V, V] Even


def turned(mesh: Mesh, quaternions: Even) -> Point:
    """The vertices `[...] Point[V]` whose edges best match the mesh's edges turned and scaled by the
    quaternion field `[...] Even[V]` along them, integrated by Simpson's rule: the turns at the two ends,
    averaged as maps, and twice the turn at the middle; matched by the cotangent Laplacian, centred."""
    boundary = as_ga_sparse(mesh.edges, as_scalar(np.ones_like(mesh.edges) * [-1, 1]))   # [E, V] Scalar
    edge_means = as_ga_sparse(mesh.edges, as_scalar(np.ones_like(mesh.edges) / 2))     # [E, V] Scalar
    edge_weights = spdiag(mesh.edge_ratio)                                     # [E, E] Scalar
    edges = boundary * mesh.vertices                                           # Point[E]
    laplacian = ~boundary * edge_weights * boundary                            # [V, V] Scalar
    moved = ((edge_means * (quaternions << Point))(edges) + 2 * ((edge_means * quaternions) << edges)) / 3   # [...] Point[E]
    return (laplacian * Point).lstsq(~boundary * edge_weights * moved)


def spin_transform_deform(mesh: Mesh, curvature_change: Scalar) -> Mesh:
    """The mesh after the spin transformation that changes each face's mean curvature by
    `curvature_change`, `Scalar[F]`: the field of least energy per unit of vertex area, divided by its
    area-weighted mean, turning the edges."""
    vertex_areas = spdiag(mesh.vertex_areas)                                   # [V, V] Scalar
    _, modes = (energy(mesh, curvature_change) * Even).eigh(vertex_areas * Even, 1)   # [1] Even[V]
    mean = (vertex_areas * modes[0]).sites.sum() / mesh.vertex_areas.sites.sum()   # [] Even
    return mesh.copy(vertices=turned(mesh, modes[0] / mean))


def dirac_spheres(mesh: Mesh, eigenvalue: int, count: int):
    """The energies of the count least-energy fields for a constant curvature change, the given
    eigenvalue, and the surfaces they spin the mesh into.

    With the change constant, the fields of zero energy are the eigenfields of the Dirac operator itself:
    on the unit sphere the spinor spherical harmonics, and the surfaces are the Dirac spheres. Their
    eigenvalues are the integers but -1, each with multiplicity its value plus one. The fields are taken
    as they come, at unit size per unit of vertex area: their mean vanishes.
    """
    constant = as_scalar(np.full(len(mesh.faces), float(eigenvalue))).field()   # Scalar[F]
    energies, fields = (energy(mesh, constant) * Even).eigh(spdiag(mesh.vertex_areas) * Even, count)   # [count] Scalar, [count] Even[V]
    return energies, [mesh.copy(vertices=vertices) for vertices in turned(mesh, fields)]


def geometric_derivative(mesh: Mesh) -> SparseExtensor:
    """The geometric derivative of fields on the vertices, linear on each face, `[F, V] Vector`: on
    each face, each corner's coupling is the gradient of its hat function, the edge facing it paired with
    the face's unit normal, over twice the face's area."""
    gradients = mesh.triangle_edges | mesh.face_normals                        # [3] Vector[F]
    return mesh.at_corners(gradients / (2 * mesh.triangle_areas))


def mean_curvature(mesh: Mesh):
    """Pointwise signed mean curvature per face.

    The cotan-laplacian of the positions gives the *area-integrated* mean
    curvature normal `L x = 2 A_v H n` at each vertex; dividing by twice the
    vertex (barycentric) area recovers the pointwise `H`, which we then average
    onto faces (the domain the curvature change lives on).
    """
    boundary = as_ga_sparse(mesh.edges, as_scalar(np.ones_like(mesh.edges) * [-1, 1]))
    face_means = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))
    laplacian = ~boundary * spdiag(mesh.edge_ratio) * boundary                 # [V, V] Scalar
    # the integrated mean-curvature normal, a displacement at each vertex, met with the vertex's normal:
    # its signed size, read against the sense the faces give space
    curving = laplacian * mesh.vertices                                        # Point[V]
    integrated = (curving & mesh.vertex_normals) * mesh.unit_volume            # Scalar[V]
    return face_means * (integrated / (2 * mesh.vertex_areas))                 # Scalar[F]


def _recenter(mesh: Mesh):
    """Remove the translation/scale gauge freedom the flow leaves undetermined."""
    v = mesh.vertices - mesh.vertices.sites.mean()
    return mesh.copy(vertices=v / v.norm().sites.mean())


def conformal_smooth(mesh: Mesh, iterations: int, rate: float):
    """Conformal curvature flow via iterated spin transforms.

    Each step prescribes `rho = -rate * (H - Hbar)`: a conformal *change* in mean
    curvature that shaves off each face's deviation from the (area-weighted) mean.
    This relaxes the surface toward constant mean curvature while every individual
    step stays conformal (angle-preserving). Run on Spot the cow it draws in the legs,
    horns and ears and rounds the body into a sphere, every square of a checkerboard on
    the cow staying square.

    Note `rho` is a *change* in curvature, so `rho = 0` (rate 0) is the identity;
    feeding absolute curvature instead diverges. Yields the mesh before each step
    and after the last.
    """
    mesh = _recenter(mesh)
    yield mesh
    for _ in range(iterations):
        h = mean_curvature(mesh)
        h_mean = (h * mesh.triangle_areas).sites.sum() / mesh.triangle_areas.sites.sum()
        mesh = _recenter(spin_transform_deform(mesh, -rate * (h - h_mean)))
        yield mesh


# --- plumbing -------------------------------------------------------------------------
icosphere = Mesh.icosphere


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
    return Mesh(context.multivector.antivector(unique).field(), inverse.reshape(-1)[np.concatenate(faces)])


def spot() -> Mesh:
    """Spot the cow, by Keenan Crane, from his model repository (CC0),
    https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/: upright with z up, centred, and scaled to
    reach one at her furthest."""
    mesh = Mesh.read(SPOT, 2.0)
    return mesh.copy(UPRIGHT >> mesh.vertices)