"""Oriented triangle meshes, their geometric fields, and sparse incidence operators.

Quantities on the mesh are fields over its vertices, edges or faces, `Point[V]` or `Scalar[F]`;
a quantity per corner of each face is a `[3]` batch of face fields, one per corner.

The algebra is not fixed here. `ga` is supplied per instance, by `examples.instantiate("examples.util.mesh", VGA3D)`
or any other algebra. The incidence operators and the Loop step hold scalar cells and act on fields of
any type. The vertices are points, antivectors, intrinsic like all geometry, and so are the edges between
them; the geometric quantities pair them with the meet `&` and the inner product `|`: two edges meet in
their face's normal, twice its area long, and three corners meet in the sense the face turns about the
origin. That is the metric of points of the algebra of space; projective points carry theirs in joins,
the line through two points as long as their distance, and a projective surface reads it there."""

from __future__ import annotations

from functools import cached_property
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
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
# A position: an antivector, intrinsic.
Point = ga.gatype.antivector()


# --- plumbing -------------------------------------------------------------------------
def as_scalar(values: np.ndarray) -> Scalar:
    return mv.scalar(np.asarray(values)[..., None])


def length(directions: Point) -> Scalar:
    """The length of each direction: its slice read along it, in any signature where the direction is
    spacelike."""
    return (directions.dual_inverse() & directions).square_root()


def as_ga_sparse(C: np.ndarray, V: Extensor) -> SparseExtensor:
    """The sparse linear map coupling each of the R output elements to the n input elements its row
    of C names, `[R, n]`, through the extensors V of the same shape."""
    return SparseExtensor.from_columns(C, V, int(C.max()) + 1)



class Mesh:
    """An oriented triangle mesh: vertices `Point[V]` and faces `[F, 3]`, counter-clockwise
    from outside, with its edges `[E, 2]` from lower to higher vertex, the edge facing each corner of
    each face `[F, 3]`, and that edge's orientation relative to the face. A mesh does not change, so
    each quantity derived from it is computed once."""

    def __init__(self, vertices: Point, faces: np.ndarray) -> None:
        corner = np.arange(3)
        # the ends of the edge facing each corner, counter-clockwise
        facing = np.stack([faces[:, (corner + 1) % 3], faces[:, (corner + 2) % 3]], axis=-1)   # [F, 3, 2]
        self.vertices, self.faces = vertices, faces
        self.edges, face_edges = np.unique(np.sort(facing, axis=-1).reshape(-1, 2), axis=0, return_inverse=True)
        self.face_edges = face_edges.reshape(-1, 3)
        self.face_edge_orientation = np.where(facing[..., 0] < facing[..., 1], 1.0, -1.0)

    def copy(self, vertices: Point) -> Mesh:
        return type(self)(vertices, self.faces)

    @classmethod
    def disk(cls, radius: float, rings: int, sectors: int) -> Mesh:
        """A flat disk with concentric rings and a counter-clockwise triangular boundary."""
        angles = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
        radii = np.linspace(radius / rings, radius, rings)
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()   # [sectors] Point
        positions = directions[None, :] * radii[:, None]                  # [rings, sectors] Point
        vertices = concatenate([mv.antivector([[0, 0, 0]]), positions.reshape(-1)]).field()  # Point[V]

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
        directions = (mv.xy * (-angles / 2)).exp() >> mv.x.dual()   # [nodes] Point
        vertices = concatenate((mv.antivector([[0, 0, 0]]), directions * radii[ring])).field()   # Point[V]
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
        unit = mv.x.dual()                                   # [] Point
        axes = stack((unit, (mv.xy * (-np.pi / 6)).exp() >> unit))        # [axes] Point
        positions = (axes * lattice).sum(axis=-1)                          # [nodes] Point
        # Map complete hexagonal rings to circles, keeping their six-neighbour topology.
        positions = spacing * layers[inside] * positions / length(positions)
        vertices = concatenate((mv.antivector([[0, 0, 0]]), positions)).field()   # Point[V]
        mesh = cls._triangulate(vertices)
        boundary = np.flatnonzero(layers[inside] == divisions) + 1

        # Equal edge springs even out the circular map's distortion. Rim vertices can
        # slide along the circle; the incidence adjoint sums edge forces at each vertex.
        target = (4 * mesh.triangle_areas.batch().sum() / (np.sqrt(3) * len(mesh.faces))).square_root()
        for _ in range(relaxation_steps):
            edges = mesh.edge_displacements
            force = edges * (1 - target / length(edges))                   # Point[E]
            positions = (mesh.vertices - relaxation_rate * (~mesh.d0 * force)).batch()   # [V] Point
            positions = positions.at[boundary].set(radius * positions[boundary] / length(positions[boundary]))
            mesh = cls._triangulate(positions.field())
        return mesh

    @classmethod
    def icosphere(cls, levels: int) -> Mesh:
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
        mesh = cls(vertices, faces)
        for _ in range(levels):
            finer = concatenate([mesh.vertices.batch(), mesh.edge_midpoints.batch()]).normalized().field()
            mesh = cls(finer, mesh.split_faces)
        return mesh

    @classmethod
    def read(cls, path: Path, size: float) -> Mesh:
        """A mesh from a Wavefront OBJ file, its vertices and triangles only, centred on its bounding box
        and scaled so that the box's longest side is `size`."""
        rows = [line.split() for line in path.read_text().splitlines()]
        places = np.array([[float(word) for word in row[1:4]] for row in rows if row[:1] == ["v"]])
        faces = np.array([[int(word.split("/")[0]) - 1 for word in row[1:4]] for row in rows if row[:1] == ["f"]])
        middle, extent = (places.max(axis=0) + places.min(axis=0)) / 2, np.ptp(places, axis=0).max()
        return cls(mv.antivector(((places - middle) * size / extent).tolist()).field(), faces)

    @classmethod
    def torus(cls, major_radius: float, minor_radius: float, major_count: int, minor_count: int) -> Mesh:
        """A closed triangulated torus around the z axis, built by turning a circle in the xz plane around
        the xy plane."""
        major_angles = np.linspace(0.0, 2 * np.pi, major_count, endpoint=False)
        minor_angles = np.linspace(0.0, 2 * np.pi, minor_count, endpoint=False)
        around_tube = (mv.xz * (-minor_angles / 2)).exp()                       # [minor_count] Rotor
        around_axis = (mv.xy * (-major_angles / 2)).exp()                       # [major_count] Rotor
        unit = mv.x.dual()                                       # [] Point
        tube = unit * major_radius + (around_tube >> (unit * minor_radius))     # [minor_count] Point
        vertices = (around_axis[:, None] >> tube[None, :]).reshape(-1).field()  # Point[V]

        u, v = np.meshgrid(np.arange(major_count), np.arange(minor_count), indexing="ij")
        v00 = u * minor_count + v
        v10 = ((u + 1) % major_count) * minor_count + v
        v01 = u * minor_count + (v + 1) % minor_count
        v11 = ((u + 1) % major_count) * minor_count + (v + 1) % minor_count
        faces = np.concatenate([
            np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
            np.stack([v00, v11, v01], axis=-1).reshape(-1, 3),
        ])
        return cls(vertices, faces)

    @classmethod
    def _triangulate(cls, vertices: Point) -> Mesh:
        """Delaunay topology for vertices in the xy plane, with counter-clockwise faces."""
        from scipy.spatial import Delaunay

        # Positions read by pairing with the slices across the first two directions.
        coordinates = (stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array()
        return cls(vertices, Delaunay(coordinates).simplices)

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
        return SparseExtensor.from_columns(vertices[:, None], as_scalar(np.ones((len(vertices), 1))),
                                           len(self.vertices.batch()))

    @cached_property
    def d0(self) -> SparseExtensor:
        """[E, V] Scalar: oriented differences from edge tails to heads."""
        return as_ga_sparse(self.edges, as_scalar(np.ones_like(self.edges) * [-1, 1]))

    @cached_property
    def d1(self) -> SparseExtensor:
        """[F, E] Scalar: oriented circulation around each face."""
        return as_ga_sparse(self.face_edges, as_scalar(self.face_edge_orientation))

    def loop(self, pinned: np.ndarray) -> SparseExtensor:
        """[V + E, V] Scalar: one Loop step, from the vertices to those of the finer mesh, its vertices
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
        middle = self.vertices.batch().shape[-1] + self.face_edges              # [F, 3]
        corners = np.stack([self.faces, np.roll(middle, 1, axis=1), np.roll(middle, -1, axis=1)], axis=-1)   # [F, 3, 3]
        return np.concatenate([middle, corners.reshape(-1, 3)])

    def split_pins(self, pinned: np.ndarray) -> np.ndarray:
        """[V + E]: the pinned vertices of the finer mesh, the same vertices; the vertices of the edges are
        free."""
        return np.concatenate([pinned, np.zeros(len(self.edges), dtype=bool)])

    def subdivide_loop(self, pinned: np.ndarray) -> Mesh:
        """The finer mesh of one Loop step with the vertices pinned `[V]` staying where they are: the step
        applied to the vertices, on the split faces."""
        return Mesh(self.loop(pinned) * self.vertices, self.split_faces)

    @cached_property
    def boundary_edges(self) -> np.ndarray:
        """Indices of edges incident to only one face."""
        return np.flatnonzero(np.bincount(self.face_edges.ravel()) == 1)

    @cached_property
    def edge_displacements(self) -> Point:
        """Point[E]: the displacement from each edge's tail to its head, a difference of points."""
        return self.d0 * self.vertices

    @cached_property
    def edge_midpoints(self) -> Point:
        """Point[E]: the midpoint of each edge."""
        return as_ga_sparse(self.edges, as_scalar(np.full(self.edges.shape, 1 / 2))) * self.vertices

    @cached_property
    def face_centers(self) -> Point:
        """Point[F]: the centroid of each face."""
        return as_ga_sparse(self.faces, as_scalar(np.full(self.faces.shape, 1 / 3))) * self.vertices

    @cached_property
    def corners(self) -> Point:
        """[3] Point[F]: each face's vertex at each of its corners."""
        return SparseExtensor.selection(ctx, self.faces.T, len(self.vertices.batch())) * self.vertices

    @cached_property
    def triangle_edges(self) -> Point:
        """[3] Point[F]: the edge facing each corner, counter-clockwise, from the corner after it to the
        corner after that."""
        corners = self.corners
        return corners[[2, 0, 1]] - corners[[1, 2, 0]]

    @cached_property
    def face_planes(self) -> Bivector:
        """Bivector[F]: each face's unit plane, counter-clockwise seen from outside."""
        edges = self.triangle_edges
        return edges[1].commutator(edges[0]).normalized()

    @cached_property
    def face_normals(self) -> Vector:
        """Vector[F]: each face's unit normal, outwards: where two of its edges meet, over twice its area."""
        edges = self.triangle_edges
        return (edges[0] & edges[1]) / (2 * self.triangle_areas)

    @cached_property
    def unit_volume(self) -> Scalar:
        """Scalar: the sense its counter-clockwise faces give space, plus or minus one: the faces' corners
        meeting about the origin, summed over the solid, over its size."""
        corners = self.corners
        volume = (corners[0] & corners[1] & corners[2]).sites.sum()
        return volume / volume.norm()

    @cached_property
    def triangle_areas(self) -> Scalar:
        """Scalar[F]: half the size of the face's own plane, the commutator of two of its edges."""
        edges = self.triangle_edges
        return edges[1].commutator(edges[0]).norm() / 2

    @cached_property
    def edge_ratio(self) -> Scalar:
        """Scalar[E]: dual over primal edge length, half the cotangents of the angles facing each edge."""
        return self.weighted_edge_ratio(as_scalar(np.ones(len(self.faces))).field())

    def weighted_edge_ratio(self, face_weights: Scalar) -> Scalar:
        """Scalar[E]: half the cotangents of the angles facing each edge, each face's share weighted by
        `face_weights` `Scalar[F]`."""
        edges = self.triangle_edges
        after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
        # the cotangent at each corner: one edge's slice read along the other, over the size of the plane
        # they span; head to tail, the corner's own edges point the other way
        cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar[F]
        return ~self.by_corner(cotangents / 2) * face_weights

    @cached_property
    def vertex_normals(self) -> Vector:
        """Vector[V]."""
        edges = self.triangle_edges
        # each face's normal, twice its area long
        face_normals = edges[0] & edges[1]                                     # Vector[F]
        spread = as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces, dtype=float)))
        return (~spread * face_normals).normalized()

    @cached_property
    def vertex_areas(self) -> Scalar:
        """Scalar[V]: a third of the area of each triangle at the vertex."""
        return ~as_ga_sparse(self.faces, as_scalar(np.ones_like(self.faces) / 3)) * self.triangle_areas

    def corner_cosines(self) -> Scalar:
        """[3] Scalar[F]: the cosine of each corner's angle, kept exactly by a conformal map."""
        edges = self.triangle_edges
        after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]
        return -(after.dual_inverse() & before) / (length(after) * length(before))

    @cached_property
    def reconstruction(self) -> SparseExtensor:
        """[F, E] Vector <- Scalar: Whitney edge integrals evaluated at face centroids."""
        # The gradient of each corner's hat function points across its opposite edge.
        gradients = (self.triangle_edges | self.face_normals.inverse()) / (2 * self.triangle_areas)   # [3] Vector[F]
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

    def at_corners(self, cells: Extensor) -> SparseExtensor:
        """[F, V]: couplings from each face to the vertex at each of its corners, through `[3]` face
        fields of cells, one per corner."""
        return SparseExtensor.from_indices(cells.batch(), np.arange(len(self.faces)), self.faces.T,
                                           (len(self.faces), len(self.vertices.batch())))
