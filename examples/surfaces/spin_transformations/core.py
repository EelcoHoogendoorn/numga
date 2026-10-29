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

from collections.abc import Iterator

import numpy as np

from numga.sparse import SparseExtensor, spdiag
from numga.extensor import Extensor
from numga.algebras import VGA3D as ga
from examples import MODEL_DIR
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, Scalar, Point = mesh_module.Mesh, mesh_module.Scalar, mesh_module.Point
Vector, Bivector = ga.gatype.vector(), ga.gatype.bivector()
as_scalar, context = mesh_module.as_scalar, mesh_module.ctx

Even = ga.gatype.even()
# Spot the cow, from the shared models, and the two quarter turns that stand her up facing along x: her y
# axis, up in the file, turned onto z, and then x onto y.
SPOT = MODEL_DIR / "spot.obj"
upright = (context.multivector.xy * (-np.pi / 4)).exp() * (context.multivector.yz * (-np.pi / 4)).exp()   # [] Rotor+


# --- math -----------------------------------------------------------------------------
def cotangent_weights(mesh: Mesh, sides: Point) -> Scalar:
    """Each edge's cotangent weight, `Scalar+[edges]`, is half the cotangents of the angles facing it, from the edge
    facing each corner, `[3] Point-[faces]`; a corner's cotangent is one edge's slice read along the other, over the
    size of the plane they span."""
    after, before = sides[[1, 2, 0]], sides[[2, 0, 1]]                         # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    return (~mesh.incidence[2, 1] * (cotangents / 2)).sum(axis=0)


def sense_of_space(mesh: Mesh, vertices: Point) -> Scalar:
    """The sense the counter-clockwise faces give space is plus or minus one: the faces' corners meeting about the
    origin, summed over the solid, over its size."""
    a, b, c = mesh.incidence[2, 0] * vertices                                  # Point-[faces], Point-[faces], Point-[faces]
    volume = (a & b & c).sites.sum()                                           # [] Scalar-
    return volume / volume.norm()


def corner_cosines(mesh: Mesh, vertices: Point) -> Scalar:
    """The cosine of each corner's angle, `[3] Scalar+[faces]`, which a conformal map keeps exactly, is one edge's
    slice read along the other, over both their lengths."""
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                       # [3] Point-[faces]
    after, before = sides[[1, 2, 0]], sides[[2, 0, 1]]                         # [3] Point-[faces], Point-[faces]
    return -(after.dual_inverse() & before) / (after.norm() * before.norm())


def spin_transform_deform(mesh: Mesh, vertices: Point, curvature_change: Scalar) -> Point:
    """The spin transformation that changes each face's mean curvature by `curvature_change`, `Scalar+[faces]`,
    turns the edges by the field of least energy per unit of vertex area, divided by its area-weighted mean, and
    gives the vertices whose edges match the turned ones best."""
    edge_means = mesh.incidence[1, 0].mean(axis=0)                             # Scalar+[edges <- vertices]
    face_means = mesh.incidence[2, 0].mean(axis=0)                             # Scalar+[faces <- vertices]
    edges = mesh.d[0] * vertices                                               # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                        # [3] Point-[faces]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar+[faces]
    face_areas, vertex_areas = spdiag(areas), spdiag(~face_means * areas)      # Scalar+[faces <- faces], Scalar+[vertices <- vertices]
    edge_weights = spdiag(cotangent_weights(mesh, sides))                      # Scalar+[edges <- edges]
    laplacian = ~mesh.d[0] * edge_weights * mesh.d[0]                          # Scalar+[vertices <- vertices]

    residual = (spdiag(sides / (2 * areas) - curvature_change * sense_of_space(mesh, vertices) / 3) * mesh.incidence[2, 0]).sum(axis=0)   # Even-[faces <- vertices]
    _, (field,) = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, 1)   # Even+[vertices]
    quaternions = field / ((vertex_areas * field).sites.sum() / areas.sites.sum())   # Even+[vertices]
    moved = ((edge_means * (quaternions << Point))(edges) + 2 * ((edge_means * quaternions) << edges)) / 3   # Point-[edges]
    return (laplacian * Point).lstsq(~mesh.d[0] * edge_weights * moved)


def dirac_spheres(mesh: Mesh, vertices: Point, eigenvalues: np.ndarray, count: int) -> tuple[Scalar, Point]:
    """For each constant curvature change, one per eigenvalue, the `count` fields of least energy have their
    energies and spin the mesh into surfaces with vertices `[eigenvalues, count] Point-[vertices]`.

    With the change constant, the fields of zero energy are the eigenfields of the Dirac operator itself:
    on the unit sphere the spinor spherical harmonics, and the surfaces are the Dirac spheres. Their
    eigenvalues are the integers but -1, each with multiplicity its value plus one. The fields are taken
    as they come, at unit size per unit of vertex area: their mean vanishes.
    """
    edge_means = mesh.incidence[1, 0].mean(axis=0)                             # Scalar+[edges <- vertices]
    face_means = mesh.incidence[2, 0].mean(axis=0)                             # Scalar+[faces <- vertices]
    edges = mesh.d[0] * vertices                                               # Point-[edges]
    sides = mesh.boundary[2, 1] * edges                                        # [3] Point-[faces]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar+[faces]
    face_areas, vertex_areas = spdiag(areas), spdiag(~face_means * areas)      # Scalar+[faces <- faces], Scalar+[vertices <- vertices]
    edge_weights = spdiag(cotangent_weights(mesh, sides))                      # Scalar+[edges <- edges]
    laplacian = ~mesh.d[0] * edge_weights * mesh.d[0]                          # Scalar+[vertices <- vertices]

    constant = as_scalar(eigenvalues[:, None]) * sense_of_space(mesh, vertices) / 3   # [eigenvalues, 1] Scalar-
    residual = (spdiag(sides / (2 * areas) - constant) * mesh.incidence[2, 0]).sum(axis=-1)   # [eigenvalues] Even-[faces <- vertices]
    energies, fields = ((~residual * face_areas * residual) * Even).eigh(vertex_areas * Even, count)   # [eigenvalues, count] Even+[vertices]
    moved = ((edge_means * (fields << Point))(edges) + 2 * ((edge_means * fields) << edges)) / 3   # [eigenvalues, count] Point-[edges]
    return energies, (laplacian * Point).lstsq(~mesh.d[0] * edge_weights * moved)


def geometric_derivative(mesh: Mesh, vertices: Point) -> SparseExtensor:
    """The geometric derivative of fields on the vertices, linear on each face, `Vector+[faces <- vertices]`,
    couples each face to each corner by the gradient of the corner's hat function: the edge facing it paired with
    the face's unit normal, over twice the face's area."""
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                       # [3] Point-[faces]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar+[faces]
    normals = (sides[0] & sides[1]) / (2 * areas)                              # Vector-[faces]
    return (spdiag((sides | normals) / (2 * areas)) * mesh.incidence[2, 0]).sum(axis=0)


def mean_curvature(mesh: Mesh, vertices: Point) -> Scalar:
    """Each face's signed mean curvature follows from the cotangent Laplacian of the positions, the
    area-integrated mean-curvature normal at each vertex: met with the vertex's normal and divided by twice the
    vertex's area, it is the mean curvature there, averaged onto the faces, where the curvature change lives."""
    face_means = mesh.incidence[2, 0].mean(axis=0)                             # Scalar+[faces <- vertices]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                       # [3] Point-[faces]
    areas = sides[1].commutator(sides[0]).norm() / 2                           # Scalar+[faces]
    laplacian = ~mesh.d[0] * spdiag(cotangent_weights(mesh, sides)) * mesh.d[0]   # Scalar+[vertices <- vertices]
    # Each vertex's normal: its faces' normals, each twice the face's area long, summed and made unit.
    normals = (~face_means * (sides[0] & sides[1])).normalized()               # Vector-[vertices]
    # The integrated mean-curvature normal, a displacement at each vertex, met with the vertex's normal: its
    # signed size, read against the sense the faces give space.
    integrated = ((laplacian * vertices) & normals) * sense_of_space(mesh, vertices)   # Scalar+[vertices]
    return face_means * (integrated / (2 * (~face_means * areas)))             # Scalar+[faces]


def recentred(vertices: Point) -> Point:
    """The vertices are moved to their centre and scaled to unit mean distance from it, which is what a spin
    transformation leaves free."""
    vertices = vertices - vertices.sites.mean()
    return vertices / vertices.norm().sites.mean()


def conformal_smooth(mesh: Mesh, vertices: Point, iterations: int, rate: float) -> Iterator[Point]:
    """Conformal curvature flow yields the vertices at the start and after each step, each step one spin
    transformation whose curvature change takes away a share `rate` of each face's departure from the
    area-weighted mean curvature. The surface relaxes towards constant mean curvature while every step keeps its
    angles: Spot the cow draws in her legs, horns and ears and rounds into a sphere. The change is a change, so a
    rate of zero leaves the surface as it is; asking for the absolute curvature instead diverges."""
    vertices = recentred(vertices)
    yield vertices
    for _ in range(iterations):
        curvature = mean_curvature(mesh, vertices)                             # Scalar+[faces]
        sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                   # [3] Point-[faces]
        areas = sides[1].commutator(sides[0]).norm() / 2                       # Scalar+[faces]
        mean = (curvature * areas).sites.sum() / areas.sites.sum()             # [] Scalar+
        vertices = recentred(spin_transform_deform(mesh, vertices, -rate * (curvature - mean)))
        yield vertices


# --- plumbing -------------------------------------------------------------------------
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
    return mesh, upright >> vertices