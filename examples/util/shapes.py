"""This module makes meshes from geometry. Each function gives the topology of a shape together with the coordinates
of its vertices `[V, d]`, which an example places in its own algebra in one visible line,
`vertices = mv.vector(coordinates).dual()`, the points dual to the planes with those coordinates. A topology is
combinatorial only, so every mesh made from coordinates is made here.

Points given by their coordinates are meshed by their Delaunay simplices, triangles in the plane and tetrahedra in
space, or by the facets of their convex hull, which tile its boundary, a sphere through points on it; a closed surface
given by its simplices is wound to face out, and a mesh of triangles is split at the midpoints of its edges. The stock
shapes are made the same way: bounded domains, a box, disks, a ball and a solid torus, and closed surfaces, a sphere, a
torus, a regular polytope's surface and a surface read from a file.

A name says the dimension: a disk, a sphere and a torus are surfaces of triangles, a ball and a solid torus solids of
tetrahedra, and a solid's border is the surface of that name; a box fills as many dimensions as it has axes, a
rectangle of triangles in the plane and a cuboid of tetrahedra in space; a polytope is the boundary of its convex hull,
triangles in space and tetrahedra in four dimensions. Each shape is built here in an algebra of its own, the vector
algebra of the dimension it lives in, its generators unnamed: its turns are rotors in the planes its basis planes span,
a polytope's vertices the orbit of a point under reflections in its mirrors, and the coordinates are read off at the
end by pairing with its basis planes. A domain's simplices are wound to positive volume, a surface's triangles to face
out.
"""

from __future__ import annotations

from functools import cache, reduce
from pathlib import Path

import numpy as np

from numga import Algebra, NumpyContext, concatenate
from numga.extensor import Extensor
from examples.util.topology import Topology, coherent, turned


# --- plumbing -------------------------------------------------------------------------
@cache
def algebra(n_dim: int):
    """The vector algebra of a dimension, its generators unnamed: its multivectors."""
    return NumpyContext(Algebra.from_pqr(n_dim, 0, 0)).multivector


def planes(n_dim: int) -> Extensor:
    """`[n_dim]`: the basis planes of a dimension's vector algebra, each reading one coordinate of a point."""
    return algebra(n_dim).vector(np.eye(n_dim))


def directions(n_dim: int) -> Extensor:
    """`[n_dim]`: the unit directions the basis planes read, one to each, antivectors."""
    return planes(n_dim).dual()


def placed(coordinates: np.ndarray) -> Extensor:
    """Points `[...]` at coordinates `[..., n_dim]` in the vector algebra of their dimension: the duals of the planes
    with those coordinates."""
    return algebra(coordinates.shape[-1]).vector(coordinates).dual()


def read(points: Extensor, readers: Extensor) -> np.ndarray:
    """`[..., n_dim]`: points' coordinates, read by pairing them with the planes `readers` `[n_dim]` of their vector
    algebra."""
    return np.moveaxis((readers.reshape(readers.shape + (1,) * points.ndim) & points).to_array(), 0, -1)


def outward(simplices: np.ndarray, coordinates: np.ndarray) -> Topology:
    """The topology of top simplices `[N, n + 1]` filling the space of their vertices at the coordinates `[V, n]`, each
    wound to positive volume: its edges from its first corner a right-handed frame, counter-clockwise in the plane."""
    edges = coordinates[simplices[:, 1:]] - coordinates[simplices[:, :1]]       # [N, n, n]
    return Topology(len(coordinates), np.where((np.linalg.det(edges) > 0)[:, None], simplices, turned(simplices)))


# --- meshing --------------------------------------------------------------------------
def delaunay(coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """The Delaunay mesh of points at the coordinates `[V, n]`, in any dimension: their convex hull filled with the
    simplices whose circumspheres have none of the points inside, triangles in the plane and tetrahedra in space, each
    wound to positive volume. Its topology and the coordinates."""
    from scipy.spatial import Delaunay
    return outward(Delaunay(coordinates).simplices, coordinates), coordinates


def closed(simplices: np.ndarray, coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """A closed surface of the top simplices `[N, k + 1]` on vertices at the coordinates `[V, k + 1]`, whatever sense
    the simplices come in, wound coherently and to face out, so that the volume it encloses is positive. Its topology
    and the coordinates."""
    wound = coherent(simplices)
    # The enclosed volume: the cone each simplex spans from the origin, its corners' determinant, summed.
    volume = np.linalg.det(coordinates[wound]).sum()
    return Topology(len(coordinates), np.where(volume < 0, turned(wound), wound)), coordinates


def convex_boundary(coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """The boundary of the convex hull of points in convex position at the coordinates `[V, n]`, tiled by its facets
    as simplices, triangles in space and tetrahedra in four dimensions, wound to face out: of points on a sphere, the
    sphere tiled by the simplices between them. Its topology and the coordinates."""
    from scipy.spatial import ConvexHull
    return closed(ConvexHull(coordinates).simplices, coordinates)


def midpoint_split(topology: Topology, coordinates: np.ndarray, n_levels: int) -> tuple[Topology, np.ndarray]:
    """A mesh of triangles on vertices at the coordinates `[V, n]` with every triangle split in four at the midpoints
    of its edges, `n_levels` times over, so that the finer mesh lies exactly on the coarse one. Its topology and
    coordinates, the coarse vertices followed, level by level, by the midpoints, halfway along each edge."""
    points = placed(coordinates).field()                                        # Point-[V]
    for _ in range(n_levels):
        midpoints = topology.incidence[1, 0].mean(axis=0) * points             # Point-[E]
        points = concatenate([points.batch(), midpoints.batch()]).field()      # Point-[V + E]
        topology = Topology(len(points.batch()), topology.split_faces)
    return topology, read(points.batch(), planes(coordinates.shape[-1]))


# --- domains --------------------------------------------------------------------------
def box(axes: list[np.ndarray]) -> tuple[Topology, np.ndarray]:
    """The box a grid of coordinates along each axis `[axes][steps]` spans, filled with simplices in as many dimensions
    as it has axes: a rectangle of triangles in the plane, a cuboid of tetrahedra in space. Each cell of the grid is
    cut along the staircase from its lowest corner to its highest, as `Topology.product` cuts a prism, and each
    simplex wound to positive volume. Its topology and coordinates `[V, n]`, the first axis running fastest."""
    grid = reduce(Topology.product, [Topology.chains([len(steps)]) for steps in axes])
    coordinates = np.stack(np.meshgrid(*axes[::-1], indexing="ij")[::-1], axis=-1).reshape(-1, len(axes))
    return outward(grid.elements[-1], coordinates), coordinates


def disk(radius: float, n_rings: int, n_sectors: int) -> tuple[Topology, np.ndarray]:
    """A flat disk of triangles, `n_rings` concentric rings of `n_sectors` vertices about its centre, wound from x
    toward y: its topology and the coordinates of its vertices `[V, 2]`."""
    across, up = planes(2)
    turns = ((across ^ up) * (-np.linspace(0, 2 * np.pi, n_sectors, endpoint=False) / 2)).exp()   # [sectors] Rotor+
    spokes = turns >> directions(2)[0]                                          # [sectors] Point-
    rings = spokes[None, :] * np.linspace(radius / n_rings, radius, n_rings)[:, None]   # [rings, sectors] Point-
    points = concatenate([spokes[:1] * 0, rings.reshape(-1)])                  # [V] Point-
    return Topology.rings(n_rings, n_sectors), read(points, planes(2))


def hexagonal_disk(radius: float, divisions: int) -> tuple[Topology, np.ndarray]:
    """A circular disk of nearly equilateral triangles, its first vertex at the centre: a triangular lattice in
    `divisions` hexagonal rings about the centre, each ring drawn out onto the circle of its radius so that every
    interior vertex keeps six neighbours, and the Delaunay mesh of its points. Its topology and coordinates `[V, 2]`."""
    across, up = planes(2)
    spacing = radius / divisions
    indices = np.arange(-divisions, divisions + 1)
    columns, rows = np.meshgrid(indices, indices, indexing="ij")
    layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
    inside = (layers > 0) & (layers <= divisions)
    lattice = np.stack((columns[inside], rows[inside]), axis=-1)               # [nodes, lattice axes]
    # The lattice's two axes, the first direction and that turned a sixth of a turn toward the second.
    unit = directions(2)[0]                                                      # [] Point-
    lattice_axes = concatenate([unit[None], (((across ^ up) * (-np.pi / 6)).exp() >> unit)[None]])
    positions = (lattice_axes * lattice).sum(axis=-1)                           # [nodes] Point-
    # Map complete hexagonal rings to circles, keeping their six-neighbour topology.
    positions = spacing * layers[inside] * positions / positions.antinorm()
    return delaunay(read(concatenate((unit[None] * 0, positions)), planes(2)))


def concentric_disk(boundaries: np.ndarray, divisions: int) -> tuple[Topology, np.ndarray]:
    """A disk of triangles in staggered circular rings with nearly equilateral cells: the boundaries are increasing
    interface radii, ending at the disk's rim, and `divisions` sets the approximate number of radial intervals across
    it. Its topology and coordinates `[V, 2]`."""
    across, up = planes(2)
    spacing = boundaries[-1] / divisions
    intervals = np.diff(np.concatenate(([0], boundaries)))
    counts = np.maximum(1, np.rint(intervals / spacing).astype(int))
    # Each interval cut evenly into its count of ring radii.
    of_interval = np.repeat(np.arange(len(counts)), counts)                       # [rings]
    steps = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + 1
    radii = boundaries[of_interval] - intervals[of_interval] + intervals[of_interval] * steps / counts[of_interval]
    # Circumference determines population, so the centre has the same cell size as the rim.
    edge_length = 2 * spacing / np.sqrt(3)
    populations = np.maximum(6, np.rint(2 * np.pi * radii / edge_length).astype(int))
    ring = np.repeat(np.arange(len(radii)), populations)
    offsets = np.repeat(np.cumsum(populations) - populations, populations)
    angles = 2 * np.pi * (np.arange(populations.sum()) - offsets + (ring % 2) / 2) / populations[ring]
    spokes = ((across ^ up) * (-angles / 2)).exp() >> directions(2)[0]         # [nodes] Point-
    return delaunay(read(concatenate((spokes[:1] * 0, spokes * radii[ring])), planes(2)))


def triangular_disk(radius: float, divisions: int) -> tuple[Topology, np.ndarray]:
    """A circular disk of nearly equilateral triangles with six-neighbour interior vertices: the hexagonal disk with
    its triangles evened out. Its topology and coordinates `[V, 2]`."""
    relaxation_steps = 20
    relaxation_rate = 0.1
    topology, coordinates = hexagonal_disk(radius, divisions)
    vertices = placed(coordinates).field()                                     # Point-[V]
    boundary = topology.exterior[0]

    # Equal edge springs even out the circular map's distortion. Rim vertices can slide along the circle; the
    # incidence's reverse sums edge forces at each vertex.
    sides = (topology.boundary[2, 1] * (topology.d[0] * vertices)).batch()     # [corners, F] Point-
    area = (sides[0] & sides[1]).antinorm().sum() / 2
    target = (4 * area / (np.sqrt(3) * topology.n_elements[2])).square_root()
    for _ in range(relaxation_steps):
        edges = topology.d[0] * vertices                                         # Point-[E]
        force = edges * (1 - target / edges.antinorm())                          # Point-[E]
        positions = (vertices - relaxation_rate * (~topology.d[0] * force)).batch()   # [V] Point-
        positions = positions.at[boundary].set(radius * positions[boundary] / positions[boundary].antinorm())
        vertices = positions.field()
        topology, _ = delaunay(read(positions, planes(2)))
    return topology, read(vertices.batch(), planes(2))


def ball(levels: int, n_layers: int) -> tuple[Topology, np.ndarray]:
    """The unit ball, a solid of tetrahedra: the sphere of the given level in `n_layers` shells, the innermost coned to
    the centre, each next one swept from the one inside it. Its border is that sphere. Its topology and coordinates
    `[V, 3]`, the centre first, then shell after shell."""
    shell, points = sphere(levels)
    radii = np.arange(1, n_layers + 1) / n_layers                               # [layers]
    coordinates = np.concatenate([np.zeros((1, 3)), (radii[:, None, None] * points[None]).reshape(-1, 3)])
    # The innermost shell coned to the centre, and the shells swept outward one after another, after the centre.
    cone = shell.cone().elements[-1]                                             # the apex is the shell's next vertex
    cone = np.where(cone == shell.n_vertices, -1, cone) + 1
    layers = shell.product(Topology.chains([n_layers])).elements[-1] + 1
    return outward(np.concatenate([cone, layers]), coordinates), coordinates


def solid_torus(major_radius: float, minor_radius: float, n_major: int, divisions: int) -> tuple[Topology, np.ndarray]:
    """A solid torus around z, a solid of tetrahedra: a disk of nearly equilateral triangles in the plane of x and z,
    about x at the major radius, swept around the plane of x and y in `n_major` steps, closing on itself. Its border is
    a torus. Its topology and coordinates `[V, 3]`, the disk's copy after copy."""
    across, up, _ = planes(3)
    section, flat = triangular_disk(minor_radius, divisions)
    # The disk's coordinates across x and z, about x at the major radius.
    disk_points = directions(3)[0] * major_radius + placed(np.stack([flat[:, 0], np.zeros(len(flat)), flat[:, 1]], axis=-1))
    around_axis = ((across ^ up) * (-np.linspace(0.0, 2 * np.pi, n_major, endpoint=False) / 2)).exp()   # [major] Rotor+
    coordinates = read((around_axis[:, None] >> disk_points[None, :]).reshape(-1), planes(3))
    loop = Topology.loops([n_major])
    return outward(section.product(loop).elements[-1], coordinates), coordinates


# --- closed surfaces ------------------------------------------------------------------
def sphere(levels: int) -> tuple[Topology, np.ndarray]:
    """The unit sphere, a surface of triangles: an icosahedron with each face split in four the given number of times,
    each edge's midpoint pushed out onto the sphere. Its topology and coordinates `[V, 3]`."""
    topology, coordinates = icosahedron()
    coordinates = coordinates / np.linalg.norm(coordinates, axis=-1, keepdims=True)
    for _ in range(levels):
        topology, finer = midpoint_split(topology, coordinates, 1)
        points = placed(finer)                                                  # [V + E] Point-
        coordinates = read(points / points.antinorm(), planes(3))
    return topology, coordinates


def torus(major_radius: float, minor_radius: float, n_major: int, n_minor: int) -> tuple[Topology, np.ndarray]:
    """A closed torus around z, a surface of triangles: a circle in the plane of x and z about x at the major radius,
    turned around the plane of x and y. Its topology and coordinates `[V, 3]`."""
    across, up, out = planes(3)
    around_tube = ((across ^ out) * (-np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False) / 2)).exp()   # [minor] Rotor+
    around_axis = ((across ^ up) * (-np.linspace(0.0, 2 * np.pi, n_major, endpoint=False) / 2)).exp()   # [major] Rotor+
    unit = directions(3)[0]                                                      # [] Point-
    tube = unit * major_radius + (around_tube >> (unit * minor_radius))         # [minor] Point-
    return Topology.torus(n_major, n_minor), read((around_axis[:, None] >> tube[None, :]).reshape(-1), planes(3))


def surface(path: Path, size: float) -> tuple[Topology, np.ndarray]:
    """A closed surface of triangles from a Wavefront OBJ file, its vertices and triangles only, centred on its bounding
    box and scaled so that the box's longest side is `size`, wound to face out whatever winding the file gives. Its
    topology and coordinates `[V, 3]`."""
    rows = [line.split() for line in path.read_text().splitlines()]
    places = np.array([row[1:4] for row in rows if row[:1] == ["v"]], dtype=float)
    faces = np.array([[word.split("/")[0] for word in row[1:4]] for row in rows if row[:1] == ["f"]], dtype=int) - 1
    middle, extent = (places.max(axis=0) + places.min(axis=0)) / 2, np.ptp(places, axis=0).max()
    return closed(faces, (places - middle) * size / extent)


# --- regular polytopes ----------------------------------------------------------------
def polytope(schlafli: tuple[int, ...]) -> tuple[Topology, np.ndarray]:
    """The regular polytope of a Schläfli symbol, `(3, 5)` the icosahedron, `(3, 3, 5)` the 600-cell, as the boundary
    of its convex hull: triangles in space, tetrahedra in four dimensions, wound to face out. Its vertices are the
    orbit of one point under the reflections in the mirrors of its symmetry group: mirrors whose normals meet at the
    angles the symbol names, pi over each entry between neighbours and square otherwise, a Cholesky factor of their
    cosines; the point on every mirror but the first, at unit distance. Every other mirror is square to the rest of its
    kind and factored first, so that they are basis planes: the icosahedron's vertices lie at `(1, 0, golden)`, its
    cyclic turns and their mirror images. Its topology and coordinates `[V, n]`."""
    from scipy.spatial import cKDTree
    n_dim = len(schlafli) + 1
    orders = np.full((n_dim, n_dim), 2)
    orders[np.arange(n_dim - 1), np.arange(1, n_dim)] = orders[np.arange(1, n_dim), np.arange(n_dim - 1)] = schlafli
    cosines = -np.cos(np.pi / orders)
    np.fill_diagonal(cosines, 1.0)
    order = np.r_[0:n_dim:2, 1:n_dim:2]
    normals = np.empty((n_dim, n_dim))                                          # [mirrors, n_dim]
    normals[order] = np.linalg.cholesky(cosines[order][:, order])
    mirrors = (planes(n_dim) * normals).sum(axis=-1)                            # [mirrors] Vector+
    seed = np.linalg.solve(normals, np.eye(n_dim)[0])
    # Reflect the points found so far in every mirror, round after round, until no new point appears. Distinct vertices
    # lie far apart on the unit sphere, so a point within a millionth of a known one is that one.
    found = (seed / np.linalg.norm(seed))[None]                                 # [points, n_dim]
    frontier = found
    while len(frontier):
        reflected = read(mirrors[:, None] >> placed(frontier)[None], planes(n_dim)).reshape(-1, n_dim)
        fresh = reflected[np.isinf(cKDTree(found).query(reflected, distance_upper_bound=1e-6)[0])]
        # The fresh points that come out of several reflections at once, kept once.
        repeats = cKDTree(fresh).query_pairs(1e-6, output_type="ndarray")[:, 1]
        frontier = np.delete(fresh, repeats, axis=0)
        found = np.concatenate([found, frontier])
    return convex_boundary(found)


def tetrahedron() -> tuple[Topology, np.ndarray]:
    return polytope((3, 3))


def cube() -> tuple[Topology, np.ndarray]:
    return polytope((4, 3))


def octahedron() -> tuple[Topology, np.ndarray]:
    return polytope((3, 4))


def dodecahedron() -> tuple[Topology, np.ndarray]:
    return polytope((5, 3))


def icosahedron() -> tuple[Topology, np.ndarray]:
    return polytope((3, 5))


def simplex_4() -> tuple[Topology, np.ndarray]:
    return polytope((3, 3, 3))


def tesseract() -> tuple[Topology, np.ndarray]:
    return polytope((4, 3, 3))


def cell_16() -> tuple[Topology, np.ndarray]:
    return polytope((3, 3, 4))


def cell_24() -> tuple[Topology, np.ndarray]:
    return polytope((3, 4, 3))


def cell_120() -> tuple[Topology, np.ndarray]:
    return polytope((5, 3, 3))


def cell_600() -> tuple[Topology, np.ndarray]:
    return polytope((3, 3, 5))
