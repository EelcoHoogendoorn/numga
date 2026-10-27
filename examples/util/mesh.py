"""Oriented triangle meshes: their topology, the incidence operators, and the Loop step.

A mesh is its faces and the edges between them; it holds no positions. Its constructors return the mesh with its
vertices, `Point[V]`, and every reading of those vertices is the example's own. Quantities on the mesh are fields over
its vertices, edges or faces, `Point[V]` or `Scalar[F]`; a quantity per corner of each face is a `[3]` batch of face
fields, one per corner, and a face's corners read off its vertices by indexing, `vertices.batch()[mesh.faces.T]`.
The edges are `mesh.d0 * vertices`, and the edge facing each corner is one of them, picked out and signed as the face
runs along it, `edges.batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T`.

No geometry belongs here, by design: no edge vectors, midpoints, centroids, areas, normals, angles, cotangent weights,
Whitney maps or anything else that reads the vertices, however many examples compute the same thing. The notebooks
promise to be self-contained expressions of their geometry, so each example writes those few lines itself, in view,
in the algebra; shared helpers would hide exactly the part an example exists to show. Repetition of geometry across
examples is intended and is not to be factored back into this module. What belongs here is what knows only the face
array: incidences, the maps that index by them, subdivision of the connectivity, and constructors that place
vertices to build a mesh.

The algebra is not fixed here. `ga` is supplied per instance, by `examples.instantiate("examples.util.mesh", VGA3D)`
or any other algebra. The incidence operators and the Loop step hold scalar cells and act on fields of any type; the
constructors place the vertices as points, antivectors, intrinsic like all geometry."""

from __future__ import annotations

from functools import cached_property, reduce
from operator import mul
from pathlib import Path

import numpy as np

from numga import Algebra, concatenate, stack
from numga.backend.context import Context
from numga.extensor import Extensor
from numga.sparse import SparseExtensor

# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()
# A position: an antivector, intrinsic.
Point = ga.gatype.antivector()


# --- plumbing -------------------------------------------------------------------------
def as_scalar(values: np.ndarray) -> Scalar:
    return mv.scalar(np.asarray(values)[..., None])


def as_ga_sparse(C: np.ndarray, V: Extensor) -> SparseExtensor:
    """The sparse linear map coupling each of the R output elements to the n input elements its row
    of C names, `[R, n]`, through the extensors V of the same shape."""
    return SparseExtensor.from_columns(C, V, int(C.max()) + 1)


class Mesh:
    """An oriented triangle mesh's topology: `n_vertices` vertices and faces `[F, 3]`, counter-clockwise
    from outside, with its edges `[E, 2]` from lower to higher vertex, the edge facing each corner of
    each face `[F, 3]`, and that edge's orientation relative to the face. A mesh does not change, so
    each map derived from it is computed once."""

    # A mesh that was not subdivided has no Loop steps behind it.
    steps: tuple[SparseExtensor, ...] = ()

    def __init__(self, n_vertices: int, faces: np.ndarray) -> None:
        corner = np.arange(3)
        # the ends of the edge facing each corner, counter-clockwise
        facing = np.stack([faces[:, (corner + 1) % 3], faces[:, (corner + 2) % 3]], axis=-1)   # [F, 3, 2]
        self.n_vertices, self.faces = n_vertices, faces
        self.edges, face_edges = np.unique(np.sort(facing, axis=-1).reshape(-1, 2), axis=0, return_inverse=True)
        self.face_edges = face_edges.reshape(-1, 3)
        self.face_edge_orientation = np.where(facing[..., 0] < facing[..., 1], 1.0, -1.0)

    @classmethod
    def disk(cls, radius: float, rings: int, sectors: int) -> tuple[Mesh, Point]:
        """A flat disk with concentric rings and a counter-clockwise triangular boundary."""
        angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        radii = np.linspace(radius / rings, radius, rings)
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()   # [sectors] Point-
        positions = directions[None, :] * radii[:, None]                  # [rings, sectors] Point-
        vertices = concatenate([mv.antivector([[0, 0, 0]]), positions.reshape(-1)]).field()  # Point-[V]

        # Each sector starts with a centre triangle; successive rings contribute two triangles.
        indices = np.arange(rings * sectors).reshape(rings, sectors) + 1
        following = np.roll(indices, -1, axis=1)
        fan = np.stack([np.zeros(sectors, dtype=int), indices[0], following[0]], axis=-1)
        inner, outer = indices[:-1], indices[1:]
        inner_next, outer_next = following[:-1], following[1:]
        first = np.stack([inner, outer, outer_next], axis=-1).reshape(-1, 3)
        second = np.stack([inner, outer_next, inner_next], axis=-1).reshape(-1, 3)
        return cls(len(vertices.batch()), np.concatenate([fan, first, second])), vertices

    @classmethod
    def concentric_disk(cls, boundaries: np.ndarray, divisions: int) -> tuple[Mesh, Point]:
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
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()   # [nodes] Point-
        vertices = concatenate((mv.antivector([[0, 0, 0]]), directions * radii[ring])).field()   # Point-[V]
        return cls.triangulate(vertices), vertices

    @classmethod
    def triangular_disk(cls, radius: float, divisions: int) -> tuple[Mesh, Point]:
        """A circular disk of nearly equilateral triangles with six-neighbour interior vertices."""
        spacing = radius / divisions
        relaxation_steps = 20
        relaxation_rate = 0.1
        indices = np.arange(-divisions, divisions + 1)
        columns, rows = np.meshgrid(indices, indices, indexing="ij")
        layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
        inside = (layers > 0) & (layers <= divisions)
        lattice = np.stack((columns[inside], rows[inside]), axis=-1)          # [nodes, axes]
        unit = mv.x.dual()                                   # [] Point-
        axes = stack((unit, (mv.xy * (-np.pi / 6)).exp() >> unit))        # [axes] Point-
        positions = (axes * lattice).sum(axis=-1)                          # [nodes] Point-
        # Map complete hexagonal rings to circles, keeping their six-neighbour topology.
        positions = spacing * layers[inside] * positions / positions.norm()
        vertices = concatenate((mv.antivector([[0, 0, 0]]), positions)).field()   # Point-[V]
        mesh = cls.triangulate(vertices)
        boundary = np.flatnonzero(layers[inside] == divisions) + 1

        # Equal edge springs even out the circular map's distortion. Rim vertices can
        # slide along the circle; the incidence adjoint sums edge forces at each vertex.
        a, b, c = vertices.batch()[mesh.faces.T]                           # [F] Point-
        area = ((b - a) & (c - a)).norm().sum() / 2
        target = (4 * area / (np.sqrt(3) * len(mesh.faces))).square_root()
        for _ in range(relaxation_steps):
            edges = mesh.d0 * vertices                                     # Point-[E]
            force = edges * (1 - target / edges.norm())                    # Point-[E]
            positions = (vertices - relaxation_rate * (~mesh.d0 * force)).batch()   # [V] Point-
            positions = positions.at[boundary].set(radius * positions[boundary] / positions[boundary].norm())
            vertices = positions.field()
            mesh = cls.triangulate(vertices)
        return mesh, vertices

    @classmethod
    def icosphere(cls, levels: int) -> tuple[Mesh, Point]:
        """A unit sphere: an icosahedron with each face split in four, the given number of times."""
        golden = (1 + 5 ** 0.5) / 2
        vertices = mv.antivector([
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ]).normalized().field()
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
            [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
            [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ])
        mesh = cls(len(vertices.batch()), faces)
        for _ in range(levels):
            # Each edge's midpoint, pushed out onto the sphere.
            ends = vertices.batch()[mesh.edges.T]                          # [2, E] Point-
            finer = concatenate([vertices.batch(), (ends[0] + ends[1]) / 2]).normalized().field()
            mesh, vertices = cls(len(finer.batch()), mesh.split_faces), finer
        return mesh, vertices

    @classmethod
    def read(cls, path: Path, size: float) -> tuple[Mesh, Point]:
        """A mesh from a Wavefront OBJ file, its vertices and triangles only, centred on its bounding box
        and scaled so that the box's longest side is `size`."""
        rows = [line.split() for line in path.read_text().splitlines()]
        places = np.array([[float(word) for word in row[1:4]] for row in rows if row[:1] == ["v"]])
        faces = np.array([[int(word.split("/")[0]) - 1 for word in row[1:4]] for row in rows if row[:1] == ["f"]])
        middle, extent = (places.max(axis=0) + places.min(axis=0)) / 2, np.ptp(places, axis=0).max()
        return cls(len(places), faces), mv.antivector(((places - middle) * size / extent).tolist()).field()

    @classmethod
    def torus(cls, major_radius: float, minor_radius: float, major_count: int, minor_count: int) -> tuple[Mesh, Point]:
        """A closed triangulated torus around the z axis, built by turning a circle in the xz plane around
        the xy plane."""
        major_angles = np.linspace(0.0, 2 * np.pi, major_count, endpoint=False)
        minor_angles = np.linspace(0.0, 2 * np.pi, minor_count, endpoint=False)
        around_tube = (mv.xz * (-minor_angles / 2)).exp()                       # [minor_count] Rotor+
        around_axis = (mv.xy * (-major_angles / 2)).exp()                       # [major_count] Rotor+
        unit = mv.x.dual()                                       # [] Point-
        tube = unit * major_radius + (around_tube >> (unit * minor_radius))     # [minor_count] Point-
        vertices = (around_axis[:, None] >> tube[None, :]).reshape(-1).field()  # Point-[V]

        u, v = np.meshgrid(np.arange(major_count), np.arange(minor_count), indexing="ij")
        v00 = u * minor_count + v
        v10 = ((u + 1) % major_count) * minor_count + v
        v01 = u * minor_count + (v + 1) % minor_count
        v11 = ((u + 1) % major_count) * minor_count + (v + 1) % minor_count
        faces = np.concatenate([
            np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
            np.stack([v00, v11, v01], axis=-1).reshape(-1, 3),
        ])
        return cls(major_count * minor_count, faces), vertices

    @classmethod
    def triangulate(cls, vertices: Point) -> Mesh:
        """Delaunay topology for vertices in the xy plane, with counter-clockwise faces."""
        from scipy.spatial import Delaunay

        # Positions read by pairing with the slices across the first two directions.
        coordinates = (stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array()
        return cls(len(coordinates), Delaunay(coordinates).simplices)

    @cached_property
    def edge_ends(self) -> tuple[SparseExtensor, SparseExtensor]:
        """The maps `[E, V]` reading every edge's tail and head."""
        return self.reading(self.edges[:, 0]), self.reading(self.edges[:, 1])

    @cached_property
    def face_corners(self) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor]:
        """The maps `[F, V]` reading every face's three corners, in its winding."""
        return tuple(self.reading(self.faces[:, corner]) for corner in range(3))

    def reading(self, vertices: np.ndarray) -> SparseExtensor:
        """[n, V] Scalar: the map reading the field at each of the given vertices `[n]`; its reverse adds
        values at them back onto the vertices."""
        return SparseExtensor.from_columns(vertices[:, None], as_scalar(np.ones((len(vertices), 1))), self.n_vertices)

    @cached_property
    def d0(self) -> SparseExtensor:
        """Scalar[E <- V]: oriented differences from edge tails to heads."""
        return as_ga_sparse(self.edges, as_scalar(np.ones_like(self.edges) * [-1, 1]))

    @cached_property
    def d1(self) -> SparseExtensor:
        """Scalar[F <- E]: oriented circulation around each face."""
        return as_ga_sparse(self.face_edges, as_scalar(self.face_edge_orientation))

    def loop(self, pinned: np.ndarray) -> SparseExtensor:
        """Scalar[V + E <- V]: one Loop step, from the vertices to those of the finer mesh, its vertices
        followed by one per edge, with the vertices pinned `[V]` staying where they are.

        An interior vertex of valence n keeps `1 - n * beta` of itself and takes `beta` of each neighbour,
        with `beta = (5/8 - (3/8 + cos(2 pi / n) / 4)**2) / n`. An interior edge's vertex takes 3/8 of each
        end and 1/8 of each vertex across the edge: each face beside the edge gives 1/8 of the corner the
        edge faces and 3/16 of each of the other two. The boundary follows itself alone: a boundary vertex
        keeps 3/4 of itself and takes 1/8 of each neighbour along the boundary, and a boundary edge's vertex
        is the midpoint of its ends."""
        vertex_count = len(pinned)
        shape = (vertex_count + len(self.edges), vertex_count)
        everyone = np.arange(vertex_count)
        rim = np.isin(np.arange(len(self.edges)), self.boundary_edges)          # [E]
        on_rim = np.isin(everyone, self.edges[rim])                              # [V]
        valence = np.bincount(self.edges.reshape(-1), minlength=vertex_count)    # [V]
        beta = (5 / 8 - (3 / 8 + np.cos(2 * np.pi / valence) / 4) ** 2) / valence
        own = np.where(pinned, 1.0, np.where(on_rim, 3 / 4, 1 - valence * beta))   # [V]
        keep = SparseExtensor.from_indices(as_scalar(own), everyone, everyone, shape)
        # each end's share in the other end, by the end's own rule
        along = np.where(pinned[self.edges], 0.0,
                         np.where(on_rim[self.edges], np.where(rim[:, None], 1 / 8, 0.0), beta[self.edges]))   # [E, 2]
        neighbours = SparseExtensor.from_indices(as_scalar(along), self.edges, self.edges[:, ::-1], shape)
        # each face's share in the vertex of the edge facing each corner, from each of its corners
        facing_rim = rim[self.face_edges][:, :, None]                            # [F, 3, 1]
        shares = np.where(np.eye(3, dtype=bool), np.where(facing_rim, 0.0, 1 / 8), np.where(facing_rim, 1 / 2, 3 / 16))
        split = SparseExtensor.from_indices(as_scalar(shares), vertex_count + self.face_edges[:, :, None],
                                            self.faces[:, None, :], shape)
        return keep + neighbours + split

    @cached_property
    def split_faces(self) -> np.ndarray:
        """[4 F, 3]: every face split in four at the vertices of its edges, the vertices of the finer mesh: a
        middle face through them, and a face at each of its corners."""
        middle = self.n_vertices + self.face_edges                              # [F, 3]
        corners = np.stack([self.faces, np.roll(middle, 1, axis=1), np.roll(middle, -1, axis=1)], axis=-1)   # [F, 3, 3]
        return np.concatenate([middle, corners.reshape(-1, 3)])

    def loop_subdivided(self, pinned: np.ndarray, n_steps: int) -> SubdividedMesh:
        """The mesh after `n_steps` Loop steps of this one, the vertices pinned `[V]` staying where they are.
        Each finer mesh holds the one before it and the map from its vertices, so the steps compose."""
        mesh = SubdividedMesh(self, pinned)
        for _ in range(n_steps - 1):
            mesh = SubdividedMesh(mesh, mesh.pinned)
        return mesh

    @cached_property
    def boundary_edges(self) -> np.ndarray:
        """Indices of edges incident to only one face."""
        return np.flatnonzero(np.bincount(self.face_edges.ravel()) == 1)

    @cached_property
    def interior_edges(self) -> np.ndarray:
        """Indices of edges between two faces."""
        return np.flatnonzero(np.bincount(self.face_edges.ravel(), minlength=len(self.edges)) == 2)

    @cached_property
    def edge_faces(self) -> np.ndarray:
        """`[interior edges, 2]`: the faces on the left and on the right of each interior edge, seen from outside
        along the edge from its lower vertex to its higher. The face on the left runs along the edge, the face on
        the right against it: the signs of the face-edge incidence, read the other way."""
        return self._edge_sides(np.broadcast_to(np.arange(len(self.faces))[:, None], self.faces.shape))

    @cached_property
    def edge_opposites(self) -> np.ndarray:
        """`[interior edges, 2]`: the vertex across each interior edge in the face on its left and in the face on
        its right, the corner each face's edge faces."""
        return self._edge_sides(self.faces)

    def _edge_sides(self, per_corner: np.ndarray) -> np.ndarray:
        """`[interior edges, 2]`: a value per corner of each face `[F, 3]`, gathered onto the edge the corner
        faces, on the left where the face runs along the edge and on the right where it runs against it."""
        along = self.face_edge_orientation > 0                                  # [F, 3]
        left, right = np.empty((2, len(self.edges)), dtype=per_corner.dtype)
        left[self.face_edges[along]], right[self.face_edges[~along]] = per_corner[along], per_corner[~along]
        return np.stack([left, right], axis=-1)[self.interior_edges]

    def by_corner(self, cells: Extensor) -> SparseExtensor:
        """[F, E]: couplings from each face to the edge facing each of its corners, through `[3]` face
        fields of cells, one per corner."""
        return SparseExtensor.from_indices(cells.batch(), np.arange(len(self.faces)), self.face_edges.T,
                                           (len(self.faces), len(self.edges)))

    def at_corners(self, cells: Extensor) -> SparseExtensor:
        """[F, V]: couplings from each face to the vertex at each of its corners, through `[3]` face
        fields of cells, one per corner."""
        return SparseExtensor.from_indices(cells.batch(), np.arange(len(self.faces)), self.faces.T,
                                           (len(self.faces), self.n_vertices))


class SubdividedMesh(Mesh):
    """The finer mesh of one Loop step of a coarser one, every face split in four. It holds the coarser mesh, the map
    from the coarser mesh's vertices to its own, `from_coarser`, and its pinned vertices, the coarser mesh's pinned
    vertices followed by the free vertices of its edges, so that it can be subdivided again."""

    def __init__(self, coarser: Mesh, pinned: np.ndarray) -> None:
        self.coarser = coarser
        self.from_coarser = coarser.loop(pinned)                                  # Scalar+[V + E <- V]
        self.pinned = np.concatenate([pinned, np.zeros(len(coarser.edges), dtype=bool)])   # [V + E]
        super().__init__(self.from_coarser.shape[0], coarser.split_faces)

    @property
    def steps(self) -> tuple[SparseExtensor, ...]:
        """The Loop steps from the first mesh to this one, coarsest first."""
        return self.coarser.steps + (self.from_coarser,)

    @cached_property
    def subdivision(self) -> SparseExtensor:
        """The map from the first mesh's vertices to this one's: its Loop steps composed, the finest acting last."""
        return reduce(mul, reversed(self.steps))
