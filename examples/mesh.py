"""Oriented triangle meshes, their geometric fields, and sparse incidence operators.

Quantities on the mesh are fields over its vertices, edges or faces, `Vector[V]` or `Scalar[F]`;
a quantity per corner of each face is a `[3]` batch of face fields, one per corner."""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack
from numga.extensor import Extensor
from numga.sparse import SparseExtensor

ga = Algebra("x+y+z+")
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
PlanarVector = ga.gatype.from_blades("x y")


# --- plumbing -------------------------------------------------------------------------
def as_scalar(values: np.ndarray) -> Scalar:
    return context.multivector.scalar(np.asarray(values)[..., None])


def at_sites(field: Extensor, index: np.ndarray) -> Extensor:
    """The elements of a field at the sites an index array names; its last axis indexes the sites of
    the result and its leading axes are batch."""
    return field.batch()[..., index].field()


def as_ga_sparse(C: np.ndarray, V: Extensor) -> SparseExtensor:
    """The sparse linear map coupling each of the R output elements to the n input elements its row
    of C names, `[R, n]`, through the extensors V of the same shape."""
    return SparseExtensor.from_columns(C, V, int(C.max()) + 1)


as_diag = SparseExtensor.from_diagonal


class Mesh:
    """An oriented triangle mesh: vertices `Vector[V]` and faces `[F, 3]`, counter-clockwise
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

    @classmethod
    def disk(cls, radius: float, rings: int, sectors: int) -> Mesh:
        """A flat disk with concentric rings and a counter-clockwise triangular boundary."""
        angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        radii = np.linspace(radius / rings, radius, rings)
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x                # [sectors] Vector
        positions = directions[None, :] * radii[:, None]                  # [rings, sectors] Vector
        vertices = concatenate([mv.vector([[0, 0, 0]]), positions.reshape(-1)]).field()  # Vector[V]

        # Each sector starts with a centre triangle; successive rings contribute two triangles.
        indices = np.arange(rings * sectors).reshape(rings, sectors) + 1
        following = np.roll(indices, -1, axis=1)
        fan = np.stack([np.zeros(sectors, dtype=int), indices[0], following[0]], axis=-1)
        inner, outer = indices[:-1], indices[1:]
        inner_next, outer_next = following[:-1], following[1:]
        first = np.stack([inner, outer, outer_next], axis=-1).reshape(-1, 3)
        second = np.stack([inner, outer_next, inner_next], axis=-1).reshape(-1, 3)
        return cls(vertices, np.concatenate([fan, first, second]))

    @classmethod
    def concentric_disk(cls, boundaries: np.ndarray, divisions: int) -> Mesh:
        """Staggered circular rings with nearly equilateral cells.

        Boundaries are increasing positive interface radii, ending at the disc's rim.
        Divisions sets the approximate number of radial intervals across the disc.
        """
        spacing = boundaries[-1] / divisions
        intervals = np.diff(np.concatenate(([0], boundaries)))
        counts = np.maximum(1, np.rint(intervals / spacing).astype(int))
        starts = boundaries - intervals
        radii = np.concatenate([start + width * np.arange(1, count + 1) / count
                                for start, width, count in zip(starts, intervals, counts)])
        # Circumference determines population, so the centre has the same cell size as the rim.
        edge_length = 2 * spacing / np.sqrt(3)
        populations = np.maximum(6, np.rint(2 * np.pi * radii / edge_length).astype(int))
        ring = np.repeat(np.arange(len(radii)), populations)
        offsets = np.repeat(np.cumsum(populations) - populations, populations)
        angles = 2 * np.pi * (np.arange(populations.sum()) - offsets + (ring % 2) / 2) / populations[ring]
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x                # [nodes] Vector
        vertices = concatenate((mv.vector([[0, 0, 0]]), directions * radii[ring])).field()   # Vector[V]
        return cls._triangulate(vertices)

    @classmethod
    def triangular_disk(cls, radius: float, divisions: int) -> Mesh:
        """A circular disk of nearly equilateral triangles with six-neighbour interior vertices."""
        spacing = radius / divisions
        relaxation_steps = 20
        relaxation_rate = 0.1
        indices = np.arange(-divisions, divisions + 1)
        columns, rows = np.meshgrid(indices, indices, indexing="ij")
        layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
        inside = (layers > 0) & (layers <= divisions)
        lattice = np.stack((columns[inside], rows[inside]), axis=-1)          # [nodes, axes]
        axes = stack((mv.x, (mv.xy * (-np.pi / 6)).exp() >> mv.x))           # [axes] Vector
        positions = (axes * lattice).sum(axis=-1)                          # [nodes] Vector
        # Map complete hexagonal rings to circles, keeping their six-neighbour topology.
        positions = spacing * layers[inside] * positions.normalized()
        vertices = concatenate((mv.vector([[0, 0, 0]]), positions)).field()   # Vector[V]
        mesh = cls._triangulate(vertices)
        boundary = np.flatnonzero(layers[inside] == divisions) + 1

        # Equal edge springs even out the circular map's distortion. Rim vertices can
        # slide along the circle; the incidence adjoint sums edge forces at each vertex.
        target = (4 * mesh.triangle_areas.batch().sum() / (np.sqrt(3) * len(mesh.faces))).square_root()
        for _ in range(relaxation_steps):
            edges = mesh.edge_vectors
            force = edges * (1 - target / edges.norm())                    # Vector[E]
            positions = (mesh.vertices - relaxation_rate * (~mesh.d0 * force)).batch()   # [V] Vector
            positions = positions.at[boundary].set(radius * positions[boundary].normalized())
            mesh = cls._triangulate(positions.field())
        return mesh

    @classmethod
    def _triangulate(cls, vertices: Vector) -> Mesh:
        """Delaunay topology for vertices in the xy plane, with counter-clockwise faces."""
        from scipy.spatial import Delaunay

        coordinates = vertices.cast(PlanarVector).kernel
        return cls(vertices, Delaunay(coordinates).simplices)

    @property
    def d0(self) -> SparseExtensor:
        """[E, V] Scalar: oriented differences from edge tails to heads."""
        return as_ga_sparse(self.edges, as_scalar(np.ones_like(self.edges) * [-1, 1]))

    @property
    def d1(self) -> SparseExtensor:
        """[F, E] Scalar: oriented circulation around each face."""
        return as_ga_sparse(self.face_edges, as_scalar(self.face_edge_orientation))

    @property
    def boundary_edges(self) -> np.ndarray:
        """Indices of edges incident to only one face."""
        return np.flatnonzero(np.bincount(self.face_edges.ravel()) == 1)

    @property
    def edge_vectors(self) -> Vector:
        """Vector[E]: the displacement from each edge's tail to its head."""
        return self.d0 * self.vertices

    @property
    def edge_midpoints(self) -> Vector:
        """Vector[E]: the midpoint of each edge."""
        return as_ga_sparse(self.edges, as_scalar(np.full(self.edges.shape, 1 / 2))) * self.vertices

    @property
    def face_centers(self) -> Vector:
        """Vector[F]: the centroid of each face."""
        return as_ga_sparse(self.faces, as_scalar(np.full(self.faces.shape, 1 / 3))) * self.vertices

    @property
    def triangle_edges(self) -> Vector:
        """[3] Vector[F]: the edge facing each corner, counter-clockwise."""
        corner = np.arange(3)
        return at_sites(self.vertices, self.faces.T[(corner + 2) % 3]) - at_sites(self.vertices, self.faces.T[(corner + 1) % 3])

    @property
    def face_planes(self) -> Bivector:
        """Bivector[F]: each face's unit plane, counter-clockwise seen from outside."""
        edges = self.triangle_edges
        return (edges[0] ^ edges[1]).normalized()

    @property
    def triangle_areas(self) -> Scalar:
        """Scalar[F]."""
        edges = self.triangle_edges
        return (edges[0] ^ edges[1]).norm() / 2

    @property
    def edge_ratio(self) -> Scalar:
        """Scalar[E]: dual over primal edge length, half the cotangents of the angles facing each edge."""
        edges = self.triangle_edges
        after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
        # the cotangent at each corner
        cotangents = -(after | before) / (after ^ before).norm()               # [3] Scalar[F]
        return ~self.by_corner(cotangents / 2) * as_scalar(np.ones(len(self.faces))).field()

    @property
    def vertex_normals(self) -> Vector:
        """Vector[V]."""
        edges = self.triangle_edges
        # each face's normal, twice its area long
        face_normals = (edges[0] ^ edges[1]).dual()                           # Vector[F]
        spread = as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces, dtype=float)))
        return (~spread * face_normals).normalized()

    @property
    def vertex_areas(self) -> Scalar:
        """Scalar[V]: a third of the area of each triangle at the vertex."""
        return ~as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces) / 3)) * self.triangle_areas

    def corner_cosines(self) -> Scalar:
        """[3] Scalar[F]: the cosine of each corner's angle, kept exactly by a conformal map."""
        edges = self.triangle_edges
        after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
        return -(after | before) / (after.norm() * before.norm())

    @property
    def reconstruction(self) -> SparseExtensor:
        """[F, E] Vector <- Scalar: Whitney edge integrals evaluated at face centroids."""
        # The gradient of each corner's hat function points across its opposite edge.
        gradients = (self.triangle_edges | self.face_planes) / (2 * self.triangle_areas)   # [3] Vector[F]
        # All three hat functions equal one third at the centroid. The difference of two
        # gradients reconstructs an oriented edge integral, with the mesh edge's sign.
        coefficients = (gradients[[2, 0, 1]] - gradients[[1, 2, 0]]) / 3   # [3] Vector[F]
        orientation = as_scalar(self.face_edge_orientation.T).field()          # [3] Scalar[F]
        return self.by_corner(coefficients * orientation * Scalar)

    def by_corner(self, cells: Extensor) -> SparseExtensor:
        """[F, E]: couplings from each face to the edge facing each of its corners, through `[3]` face
        fields of cells, one per corner."""
        return SparseExtensor.from_indices(cells.batch(), np.arange(len(self.faces)), self.face_edges.T,
                                           (len(self.faces), len(self.edges)))
