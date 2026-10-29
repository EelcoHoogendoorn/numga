"""Oriented simplicial meshes: their topology, the incidence and boundary operators, and the Loop step.

A mesh is its top simplices, edges of a graph, triangles of a surface or tetrahedra of a solid, and every simplex
below them; it holds no positions. Its constructors return the mesh with its vertices, `Point[V]`, and every reading
of those vertices is the example's own. Quantities on the mesh are fields over its simplices, `Point[V]` or
`Scalar[F]`; a quantity per corner of each face is a `[3]` batch of face fields, one per corner, and a face's corners
read off its vertices by `mesh.incidence[2, 0] * vertices`. The edges are `mesh.d[0] * vertices`, and the edge facing
each corner is one of them, picked out and signed as the face runs along it, `mesh.boundary[2, 1] * edges`.

`mesh.incidence[n, m]` reads the m-simplices of each n-simplex, unsigned, a `[k]` batch of maps, one per m-simplex of
an n-simplex; `mesh.boundary[k + 1, k]` reads the facets of each (k + 1)-simplex, signed by their orientation in it;
and `mesh.d[k]`, their sum, is the exterior derivative from k-simplices to (k + 1)-simplices, so that
`mesh.d[k + 1] * mesh.d[k]` vanishes. Their batches line up: entry `i` of the corners is corner `i`, and from
triangles up entry `i` of the facets is the facet facing corner `i`, so that a quantity per corner reaches the facet
facing it. An edge's corners are its facets, its tail and its head, in that order.

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

from collections.abc import Callable
from functools import cached_property, reduce
from itertools import combinations
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


def generate_simplex_boundary(simplices: np.ndarray) -> np.ndarray:
    """The facets of simplices `[N, k + 1]`, `[N, k + 1, k]`: facet `i` is the simplex without its vertex `i`, the
    facet facing corner `i`, its vertices in the simplex's order."""
    return np.stack([np.delete(simplices, corner, axis=-1) for corner in range(simplices.shape[-1])], axis=-2)


def simplex_parity(simplices: np.ndarray) -> np.ndarray:
    """`[...]`: +1 where the vertices of each simplex `[..., k]` are in an even order, -1 where odd, counting the
    pairs out of increasing order."""
    inversions = sum((simplices[..., i] > simplices[..., j] for i, j in combinations(range(simplices.shape[-1]), 2)),
                     np.zeros(simplices.shape[:-1], dtype=int))
    return np.where(inversions % 2 == 0, 1.0, -1.0)


def lookup_simplices(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """`[...]`: the row of `table` `[n, k]`, unique and sorted, holding each sorted simplex of `query` `[..., k]`."""
    _, rows = np.unique(np.concatenate([table, query.reshape(-1, table.shape[-1])]), axis=0, return_inverse=True)
    return rows.reshape(-1)[len(table):].reshape(query.shape[:-1])


class Operators:
    """Operators by the pair of dimensions `[n, m]` they couple, each built on first use and kept."""

    def __init__(self, n_dim: int, build: Callable[[int, int], SparseExtensor]) -> None:
        self.n_dim, self.build, self.built = n_dim, build, {}

    def __getitem__(self, dimensions: tuple[int, int]) -> SparseExtensor:
        n, m = (dimension % (self.n_dim + 1) for dimension in dimensions)
        if (n, m) not in self.built:
            self.built[n, m] = self.build(n, m)
        return self.built[n, m]


class Mesh:
    """An oriented simplicial mesh's topology: `n_vertices` vertices and top simplices `[N, n_dim + 1]`, kept in
    their own order, which orients them, faces counter-clockwise from outside. Every simplex below the top is
    stored once, its vertices sorted, in `elements[k]` `[n_elements[k], k + 1]`; `boundary_indices[k]` and
    `orientation[k]` `[n_elements[k + 1], k + 2]` give the facets of each (k + 1)-simplex in `elements[k]` and their
    orientations in it. A mesh does not change, so each map derived from it is computed once."""

    # A mesh that was not subdivided has no Loop steps behind it.
    steps: tuple[SparseExtensor, ...] = ()

    def __init__(self, n_vertices: int, simplices: np.ndarray) -> None:
        simplices = np.asarray(simplices)
        self.n_vertices, self.n_dim = n_vertices, simplices.shape[-1] - 1
        elements, boundary_indices, orientation = {self.n_dim: simplices}, {}, {}
        for k in range(self.n_dim, 0, -1):
            facets = generate_simplex_boundary(elements[k])                        # [N, k + 1, k]
            # Facet i of a simplex is signed (-1)**i, times the parity of its vertices' order in the simplex.
            signs = (-1.0) ** np.arange(k + 1) * simplex_parity(facets)            # [N, k + 1]
            if k == 1:
                # An edge's facets are its ends, read in its own order, tail then head.
                elements[0] = np.arange(n_vertices)[:, None]
                boundary_indices[0], orientation[0] = facets[..., ::-1, 0], signs[..., ::-1]
            else:
                ordered = np.sort(facets, axis=-1)
                elements[k - 1] = np.unique(ordered.reshape(-1, k), axis=0)
                boundary_indices[k - 1], orientation[k - 1] = lookup_simplices(elements[k - 1], ordered), signs
        self.elements = tuple(elements[k] for k in range(self.n_dim + 1))
        self.n_elements = tuple(len(element) for element in self.elements)
        self.boundary_indices = tuple(boundary_indices[k] for k in range(self.n_dim))
        self.orientation = tuple(orientation[k] for k in range(self.n_dim))

    @cached_property
    def incidence(self) -> Operators:
        """`incidence[n, m]`, `[k] Scalar+[n-simplices <- m-simplices]`: the m-simplices of each n-simplex, unsigned,
        a batch of k selection maps, one per m-simplex of an n-simplex, for any `0 <= m < n <= n_dim`; negative
        dimensions count from the top, `incidence[-1, 0]`.

        Entry `i` of the corners, `incidence[n, 0]`, is corner `i` in the simplex's own order: an edge's tail then
        its head, a face's corners in its winding. From triangles up, entry `i` of the facets, `incidence[n, n - 1]`,
        is the facet facing corner `i`, so that a quantity per corner reaches the facet across from it. Between the
        two, as a tet's six edges in `incidence[3, 1]`, the m-simplices come in the order of their corners, (0, 1),
        (0, 2), (0, 3), (1, 2), (1, 3), (2, 3). Applied to a field, `mesh.incidence[2, 0] * vertices` is each face's
        corners, `[3] Point-[faces]`; summed over the batch, `mesh.incidence[2, 0].mean(axis=0) * vertices` is its
        centroid. Each map is built on first use."""
        return Operators(self.n_dim, lambda n, m: SparseExtensor.selection(ctx, self._local(n, m).T, self.n_elements[m]))

    @cached_property
    def boundary(self) -> Operators:
        """`boundary[k + 1, k]`, `[k + 2] Scalar+[(k + 1)-simplices <- k-simplices]`: the facets of each (k + 1)-simplex,
        the batches of `incidence[k + 1, k]`, each signed +1 or -1 by the facet's orientation in the simplex: an
        edge's tail -1 and head +1, a face's edges +1 where the face's winding runs along them, a tet's faces +1
        where they face outward. Only adjacent dimensions have a boundary; summed over the batch, it is the exterior
        derivative `d[k]`. Each map is built on first use."""
        def signed_facets(n: int, m: int) -> SparseExtensor:
            if m != n - 1:
                raise ValueError(f"only adjacent dimensions have a boundary, not {n} and {m}")
            return self._signed(n, m, self.orientation[m])
        return Operators(self.n_dim, signed_facets)

    @cached_property
    def oriented(self) -> Operators:
        """`oriented[n, m]`, `[k] Scalar+[n-simplices <- m-simplices]`: the m-simplices of each n-simplex, the batches
        of `incidence[n, m]`, each signed +1 where the m-simplex as stored, its vertices in increasing order, runs the
        way the n-simplex's own vertex order runs it, and -1 where it runs the other way. A local formula written in
        a simplex's own vertex order, such as a Whitney form on a tet's edge from its local tail to its local head,
        reaches the stored m-simplices through these signs. A facet's sign here differs from its sign in `boundary`
        by `(-1)**i` for the facet facing corner `i`. Each map is built on first use."""
        return Operators(self.n_dim, lambda n, m: self._signed(n, m, simplex_parity(self.elements[n][:, self._corners(n, m)])))

    @cached_property
    def d(self) -> tuple[SparseExtensor, ...]:
        """`d[k]`, `Scalar+[(k + 1)-simplices <- k-simplices]` for each k below the top: the exterior derivative, the
        signed sum `boundary[k + 1, k].sum(axis=0)` over the facets of each simplex. `d[0] * vertices` is the edges,
        each its head less its tail, and `d[k + 1] * d[k]` vanishes."""
        return tuple(self.boundary[k + 1, k].sum(axis=0) for k in range(self.n_dim))

    def _corners(self, n: int, m: int) -> np.ndarray:
        """`[k, m + 1]`: the local corners of each m-simplex of an n-simplex, in the order of `incidence[n, m]`."""
        if m == 0:
            return np.arange(n + 1)[:, None]
        if m == n - 1:
            return generate_simplex_boundary(np.arange(n + 1))
        return np.array(list(combinations(range(n + 1), m + 1)))

    def _local(self, n: int, m: int) -> np.ndarray:
        """`[N, k]`: the index in `elements[m]` of each m-simplex of each n-simplex, in the order of
        `incidence[n, m]`."""
        if m == 0:
            return self.elements[n]
        if m == n - 1:
            return self.boundary_indices[m]
        return lookup_simplices(self.elements[m], np.sort(self.elements[n][:, self._corners(n, m)], axis=-1))

    def _signed(self, n: int, m: int, signs: np.ndarray) -> SparseExtensor:
        """`[k] Scalar+[n-simplices <- m-simplices]`: the batches of `incidence[n, m]`, each entry signed by
        `signs` `[N, k]`."""
        local = self._local(n, m)                                                  # [N, k]
        cells = np.eye(local.shape[-1])[:, None, :] * signs[None, :, :]            # [k, N, k]
        return SparseExtensor.from_indices(as_scalar(cells), np.arange(self.n_elements[n])[:, None], local,
                                           (self.n_elements[n], self.n_elements[m]))

    @property
    def edges(self) -> np.ndarray:
        """`[E, 2]`: the edges, from tail to head."""
        return self.elements[1]

    @property
    def faces(self) -> np.ndarray:
        """`[F, 3]`: the triangles."""
        return self.elements[2]

    @property
    def tets(self) -> np.ndarray:
        """`[T, 4]`: the tetrahedra."""
        return self.elements[3]

    @property
    def face_edges(self) -> np.ndarray:
        """`[F, 3]`: the edge facing each corner of each face."""
        return self.boundary_indices[1]

    @property
    def face_edge_orientation(self) -> np.ndarray:
        """`[F, 3]`: +1 where a face runs along the edge facing a corner, -1 where against it."""
        return self.orientation[1]

    def exterior(self, k: int) -> np.ndarray:
        """Indices of the k-simplices on the boundary: those of the facets that bound a single top simplex."""
        top = self.n_dim - 1
        on_boundary = np.flatnonzero(np.bincount(self.boundary_indices[top].ravel(), minlength=self.n_elements[top]) == 1)
        for j in range(top, k, -1):
            on_boundary = np.unique(self.boundary_indices[j - 1][on_boundary])
        return on_boundary

    @classmethod
    def disk(cls, radius: float, rings: int, sectors: int) -> tuple[Mesh, Point]:
        """A flat disk with concentric rings and a counter-clockwise triangular boundary."""
        angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        radii = np.linspace(radius / rings, radius, rings)
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()   # [sectors] Point-
        positions = directions[None, :] * radii[:, None]                  # [rings, sectors] Point-
        vertices = concatenate([mv.x.dual() * np.zeros(1), positions.reshape(-1)]).field()  # Point-[V]

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
        vertices = concatenate((mv.x.dual() * np.zeros(1), directions * radii[ring])).field()   # Point-[V]
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
        # Map complete hexagonal rings to circles, keeping their six-neighbour topology. A direction's size is the root
        # of its square's size, whichever sign the square has in the algebra.
        positions = spacing * layers[inside] * positions / positions.squared().norm().square_root()
        vertices = concatenate((mv.x.dual() * np.zeros(1), positions)).field()   # Point-[V]
        mesh = cls.triangulate(vertices)
        boundary = np.flatnonzero(layers[inside] == divisions) + 1

        # Equal edge springs even out the circular map's distortion. Rim vertices can
        # slide along the circle; the incidence adjoint sums edge forces at each vertex.
        a, b, c = (mesh.incidence[2, 0] * vertices).batch()                # [F] Point-
        area = ((b - a) & (c - a)).squared().norm().square_root().sum() / 2
        target = (4 * area / (np.sqrt(3) * len(mesh.faces))).square_root()
        for _ in range(relaxation_steps):
            edges = mesh.d[0] * vertices                                   # Point-[E]
            force = edges * (1 - target / edges.squared().norm().square_root())   # Point-[E]
            positions = (vertices - relaxation_rate * (~mesh.d[0] * force)).batch()   # [V] Point-
            positions = positions.at[boundary].set(radius * positions[boundary] / positions[boundary].squared().norm().square_root())
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
            midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices       # Point-[E]
            finer = concatenate([vertices.batch(), midpoints.batch()]).normalized().field()
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

    def reading(self, vertices: np.ndarray) -> SparseExtensor:
        """[..., n, V] Scalar: the map reading the field at each of the given vertices `[..., n]`; its reverse adds
        values at them back onto the vertices."""
        return SparseExtensor.selection(ctx, vertices, self.n_vertices)

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

    def midpoint_split(self, vertices: Point, n_levels: int) -> tuple[Mesh, Point]:
        """The mesh with every face split in four at the midpoints of its edges, `n_levels` times over, and its
        vertices: the coarse vertices followed, level by level, by the midpoints, halfway along each edge, so that
        the refined mesh lies exactly on the coarse one."""
        mesh = self
        for _ in range(n_levels):
            midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices               # Point-[E]
            vertices = concatenate([vertices.batch(), midpoints.batch()]).field()  # Point-[V + E]
            mesh = Mesh(len(vertices.batch()), mesh.split_faces)
        return mesh, vertices

    def loop_subdivided(self, pinned: np.ndarray, n_steps: int) -> SubdividedMesh:
        """The mesh after `n_steps` Loop steps of this one, the vertices pinned `[V]` staying where they are.
        Each finer mesh holds the one before it and the map from its vertices, so the steps compose."""
        mesh = SubdividedMesh(self, pinned)
        for _ in range(n_steps - 1):
            mesh = SubdividedMesh(mesh, mesh.pinned)
        return mesh

    @cached_property
    def boundary_edges(self) -> np.ndarray:
        """Indices of the edges on the boundary."""
        return self.exterior(1)

    @cached_property
    def interior_edges(self) -> np.ndarray:
        """Indices of the edges off the boundary."""
        return np.setdiff1d(np.arange(len(self.edges)), self.boundary_edges)

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

    @cached_property
    def vertex_faces(self) -> np.ndarray:
        """`[V]`: the first face at each vertex, of those with the vertex as a corner."""
        faces = np.full(self.n_vertices, len(self.faces))
        np.minimum.at(faces, self.faces.reshape(-1), np.repeat(np.arange(len(self.faces)), 3))
        return faces

    @cached_property
    def spanning_tree(self) -> tuple[np.ndarray, np.ndarray]:
        """Every vertex but the first `[V - 1]`, and the edge joining each to its parent `[V - 1]`, in a
        breadth-first tree from the first vertex."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order
        graph = coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])), shape=(self.n_vertices,) * 2)
        order, parents = breadth_first_order(graph.tocsr(), 0, directed=False)
        children = order[1:]
        low, high = np.minimum(children, parents[children]), np.maximum(children, parents[children])
        return children, np.searchsorted(self.edges[:, 0] * self.n_vertices + self.edges[:, 1], low * self.n_vertices + high)

    def circulation(self, loops: list[np.ndarray]) -> SparseExtensor:
        """`Scalar[F + L <- E]`: the circulation of edge values around each face, counter-clockwise, the signed
        face-edge incidence, then around each given closed loop of vertices `[steps]`, in its own order."""
        rows, columns, signs = [np.repeat(np.arange(len(self.faces)), 3)], [self.face_edges.reshape(-1)], [self.face_edge_orientation.reshape(-1)]
        keys = self.edges[:, 0] * self.n_vertices + self.edges[:, 1]           # [E], sorted
        for row, loop in enumerate(loops, start=len(self.faces)):
            steps = np.stack([loop, np.roll(loop, -1)], axis=-1)                 # [steps, 2]
            low, high = steps.min(axis=-1), steps.max(axis=-1)
            rows.append(np.full(len(loop), row))
            columns.append(np.searchsorted(keys, low * self.n_vertices + high))
            signs.append(np.where(steps[:, 0] < steps[:, 1], 1.0, -1.0))
        shape = (len(self.faces) + len(loops), len(self.edges))
        return SparseExtensor.from_indices(as_scalar(np.concatenate(signs)), np.concatenate(rows), np.concatenate(columns), shape) * Scalar

    def _edge_sides(self, per_corner: np.ndarray) -> np.ndarray:
        """`[interior edges, 2]`: a value per corner of each face `[F, 3]`, gathered onto the edge the corner
        faces, on the left where the face runs along the edge and on the right where it runs against it."""
        along = self.face_edge_orientation > 0                                  # [F, 3]
        left, right = np.empty((2, len(self.edges)), dtype=per_corner.dtype)
        left[self.face_edges[along]], right[self.face_edges[~along]] = per_corner[along], per_corner[~along]
        return np.stack([left, right], axis=-1)[self.interior_edges]


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
