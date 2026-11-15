"""Simplicial domain and surface constructors pair a combinatorial `Topology` with vertex
coordinates of shape `[V, d]`.

Because `Topology` stores integer incidences alone, coordinate-driven geometry (Delaunay
simplicial complexes, convex-hull boundaries, coherent outward winding, and edge-midpoint
subdivision) lives here, and each example embeds the returned `[V, d]` coordinates into its own
algebra by dualizing vectors built from those coordinates.

Stock shapes distinguish filled domains from closed bounding surfaces by name. `disk`,
`hexagonal_disk`, `concentric_disk`, and `triangular_disk` triangulate planar disks; `ball` and
`solid_torus` fill three-dimensional solids with tetrahedra whose borders match `sphere` and
`torus`; `box` subdivides an axis-aligned product grid into simplices in `len(axes)` dimensions;
`flat_torus` wraps an equilateral triangular lattice across periodic edge steps; `surface` reads and
normalizes Wavefront OBJ triangle surfaces; and `polytope` (together with its regular 3D and 4D
specializations) constructs convex boundaries from Schläfli mirror cosines.

Internally, each shape is built in the unnamed Euclidean vector algebra of its ambient dimension,
sweeping rings and tubes with rotors, reflecting seed points across symmetry mirrors, and reading
out `[V, d]` coordinates by regressive pairing against the basis planes. Filled domains wind every
top simplex to positive volume, while closed surfaces wind coherently outward. Companion notes on
simplicial operators and plane-based point representations are in `docs/fields.md` and
`docs/orientation.md`.
"""

from __future__ import annotations

from functools import cache, reduce
from pathlib import Path

import numpy as np

from numga import Algebra, NumpyContext, concatenate
from numga.extensor import Extensor
from numga.topology import Topology, orient_coherently, reverse_orientation


# --- plumbing -------------------------------------------------------------------------
@cache
def algebra(n_dim: int):
    """Cached multivector namespace for the unnamed Euclidean algebra of dimension `n_dim`."""
    return NumpyContext(Algebra.from_pqr(n_dim, 0, 0)).multivector


def planes(n_dim: int) -> Extensor:
    """Basis planes `[n_dim]` of the `n_dim`-dimensional Euclidean algebra each read one point
    coordinate by regressive pairing."""
    return algebra(n_dim).vector(np.eye(n_dim))


def directions(n_dim: int) -> Extensor:
    """Dualizing `planes(n_dim)` gives the `[n_dim]` unit intrinsic antivector directions."""
    return planes(n_dim).dual()


def placed(coordinates: np.ndarray) -> Extensor:
    """Dualizing vectors with `coordinates` of shape `[..., n_dim]` places intrinsic points `[...]`
    in the Euclidean algebra of dimension `n_dim`."""
    return algebra(coordinates.shape[-1]).vector(coordinates).dual()


def read(points: Extensor, readers: Extensor) -> np.ndarray:
    """Regressive pairing against `readers` (`[n_dim]` planes) reads `points` (`[...]`) back into
    a coordinate array of shape `[..., n_dim]`."""
    return np.moveaxis((readers.reshape(readers.shape + (1,) * points.ndim) & points).to_array(), 0, -1)


def outward(simplices: np.ndarray, coordinates: np.ndarray) -> Topology:
    """Full-dimensional `simplices` (`[n_simplices, d + 1]`) on `coordinates` (`[V, d]`) flip when
    needed so every simplex has positive edge-frame determinant (counter-clockwise in 2D)."""
    edges = coordinates[simplices[:, 1:]] - coordinates[simplices[:, :1]]
    return Topology(len(coordinates), np.where((np.linalg.det(edges) > 0)[:, None], simplices, reverse_orientation(simplices)))


# --- meshing --------------------------------------------------------------------------
def delaunay(coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """Delaunay triangulation or tetrahedralization of `coordinates` (`[V, d]`) winds every top
    simplex to positive volume and pairs the resulting `Topology` with `coordinates`."""
    from scipy.spatial import Delaunay
    return outward(Delaunay(coordinates).simplices, coordinates), coordinates


def closed(simplices: np.ndarray, coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """Coherent reorientation of a closed hypersurface `simplices` (`[n_simplices, k + 1]`) on
    `coordinates` (`[V, k + 1]`) winds every facet outward so the enclosed signed volume is
    positive, returning `(topology, coordinates)`."""
    wound = orient_coherently(simplices)
    # Sum the signed cone determinants from the origin to test whether the coherent orientation faces inward or outward.
    volume = np.linalg.det(coordinates[wound]).sum()
    return Topology(len(coordinates), np.where(volume < 0, reverse_orientation(wound), wound)), coordinates


def convex_boundary(coordinates: np.ndarray) -> tuple[Topology, np.ndarray]:
    """Convex-hull facets of `coordinates` (`[V, d]`) in convex position form a coherently
    outward-wound boundary `Topology` (triangles in 3D, tetrahedra in 4D) paired with
    `coordinates`."""
    from scipy.spatial import ConvexHull
    return closed(ConvexHull(coordinates).simplices, coordinates)


def midpoint_split(topology: Topology, coordinates: np.ndarray, n_levels: int) -> tuple[Topology, np.ndarray]:
    """Each of `n_levels` refinements splits every triangle of `topology` into four sub-triangles at
    its edge midpoints, appending each level's midpoint coordinates to `coordinates` (`[V, d]`)."""
    points = placed(coordinates).field()                                        # Point-[V]
    for _ in range(n_levels):
        V, E, F = topology.n_elements
        midpoints = topology.incidence[1, 0].mean(axis=0) * points             # Point-[E]
        points = concatenate([points.batch(), midpoints.batch()]).field()      # Point-[V + E]
        topology = Topology(V + E, topology.split_faces)
    return topology, read(points.batch(), planes(coordinates.shape[-1]))


# --- domains --------------------------------------------------------------------------
def box(axes: list[np.ndarray]) -> tuple[Topology, np.ndarray]:
    """Product-grid coordinates along `axes` subdivide each hyperrectangle into positively oriented
    simplices via `Topology.product`, returning `(topology, coordinates)` of shape `[V, len(axes)]`
    with `axes[0]` varying fastest."""
    grid = reduce(Topology.product, [Topology.chains([len(steps)]) for steps in axes])
    coordinates = np.stack(np.meshgrid(*axes[::-1], indexing="ij")[::-1], axis=-1).reshape(-1, len(axes))
    return outward(grid.elements[-1], coordinates), coordinates


def disk(radius: float, n_rings: int, n_sectors: int) -> tuple[Topology, np.ndarray]:
    """Concentric rings (`n_rings` of `n_sectors` vertices each around a central vertex) triangulate
    a flat disk of `radius` wound from `x` toward `y`, returning `(topology, coordinates)` with
    `coordinates` of shape `[V, 2]`."""
    across, up = planes(2)
    turns = ((across ^ up) * (-np.linspace(0, 2 * np.pi, n_sectors, endpoint=False) / 2)).exp()   # [n_sectors] Rotor+
    spokes = turns >> directions(2)[0]                                          # [n_sectors] Point-
    rings = spokes[None, :] * np.linspace(radius / n_rings, radius, n_rings)[:, None]   # [n_rings, n_sectors] Point-
    points = concatenate([spokes[:1] * 0, rings.reshape(-1)])                  # [...] Point-
    return Topology.rings(n_rings, n_sectors), read(points, planes(2))


def hexagonal_disk(radius: float, divisions: int) -> tuple[Topology, np.ndarray]:
    """Projecting `divisions` hexagonal lattice rings around the origin onto concentric circles up
    to `radius` and Delaunay-triangulating them gives an interior six-neighbour disk triangulation
    and `[V, 2]` vertex coordinates."""
    across, up = planes(2)
    spacing = radius / divisions
    indices = np.arange(-divisions, divisions + 1)
    columns, rows = np.meshgrid(indices, indices, indexing="ij")
    layers = np.maximum.reduce((abs(columns), abs(rows), abs(columns + rows)))
    inside = (layers > 0) & (layers <= divisions)
    lattice = np.stack((columns[inside], rows[inside]), axis=-1)
    # Generate the two triangular-lattice basis directions separated by a sixth of a turn.
    unit = directions(2)[0]                                                      # [] Point-
    lattice_axes = concatenate([unit[None], (((across ^ up) * (-np.pi / 6)).exp() >> unit)[None]])
    positions = (lattice_axes * lattice).sum(axis=-1)                           # [...] Point-
    # Project each hexagonal ring onto its concentric circle while preserving six-neighbour connectivity.
    positions = spacing * layers[inside] * positions / positions.antinorm()
    return delaunay(read(concatenate((unit[None] * 0, positions)), planes(2)))


def concentric_disk(boundaries: np.ndarray, divisions: int) -> tuple[Topology, np.ndarray]:
    """Staggered circular rings subdivided across increasing radial `boundaries` (with roughly
    `divisions` radial steps across the disk) Delaunay-triangulate into nearly equilateral triangles
    with `[V, 2]` vertex coordinates."""
    across, up = planes(2)
    spacing = boundaries[-1] / divisions
    intervals = np.diff(np.concatenate(([0], boundaries)))
    counts = np.maximum(1, np.rint(intervals / spacing).astype(int))
    # Subdivide each radial interval evenly into concentric ring radii.
    of_interval = np.repeat(np.arange(len(counts)), counts)
    steps = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts) + 1
    radii = boundaries[of_interval] - intervals[of_interval] + intervals[of_interval] * steps / counts[of_interval]
    # Scale each ring's vertex count with circumference so triangle edge lengths stay uniform from centre to rim.
    edge_length = 2 * spacing / np.sqrt(3)
    populations = np.maximum(6, np.rint(2 * np.pi * radii / edge_length).astype(int))
    ring = np.repeat(np.arange(len(radii)), populations)
    offsets = np.repeat(np.cumsum(populations) - populations, populations)
    angles = 2 * np.pi * (np.arange(populations.sum()) - offsets + (ring % 2) / 2) / populations[ring]
    spokes = ((across ^ up) * (-angles / 2)).exp() >> directions(2)[0]         # [...] Point-
    return delaunay(read(concatenate((spokes[:1] * 0, spokes * radii[ring])), planes(2)))


def triangular_disk(radius: float, divisions: int) -> tuple[Topology, np.ndarray]:
    """Spring relaxation on `hexagonal_disk(radius, divisions)` evens out radial distortion while
    sliding rim vertices along the circle of `radius`, returning `(topology, coordinates)` with
    `coordinates` of shape `[V, 2]`."""
    relaxation_steps = 20
    relaxation_rate = 0.1
    topology, coordinates = hexagonal_disk(radius, divisions)
    V, E, F = topology.n_elements
    vertices = placed(coordinates).field()                                     # Point-[V]
    boundary = topology.exterior[0]

    # Equal-rest-length edge springs relax interior distortion while rim vertices project back onto the circle.
    sides = (topology.boundary[2, 1] * (topology.d[0] * vertices)).batch()     # [3, F] Point-
    area = (sides[0] & sides[1]).antinorm().sum() / 2
    target = (4 * area / (np.sqrt(3) * F)).square_root()
    for _ in range(relaxation_steps):
        edges = topology.d[0] * vertices                                         # Point-[E]
        force = edges * (1 - target / edges.antinorm())                          # Point-[E]
        positions = (vertices - relaxation_rate * (topology.d[0].adjugate() * force)).batch()  # [V] Point-
        positions = positions.at[boundary].set(radius * positions[boundary] / positions[boundary].antinorm())
        vertices = positions.field()
        topology, _ = delaunay(read(positions, planes(2)))
    return topology, read(vertices.batch(), planes(2))


def ball(levels: int, n_layers: int) -> tuple[Topology, np.ndarray]:
    """Coning the innermost of `n_layers` concentric copies of `sphere(levels)` to the origin and
    sweeping prism shells outward fills the unit ball with tetrahedra, returning `(topology,
    coordinates)` of shape `[V, 3]` with the origin at index `0`."""
    shell, points = sphere(levels)
    V_shell, E_shell, F_shell = shell.n_elements
    radii = np.arange(1, n_layers + 1) / n_layers                               # [n_layers]
    coordinates = np.concatenate([np.zeros((1, 3)), (radii[:, None, None] * points[None]).reshape(-1, 3)])
    # Cone the innermost shell to vertex 0 at the centre, then sweep consecutive spherical shells outward.
    cone = shell.cone().elements[-1]
    cone = np.where(cone == V_shell, -1, cone) + 1
    layers = shell.product(Topology.chains([n_layers])).elements[-1] + 1
    return outward(np.concatenate([cone, layers]), coordinates), coordinates


def solid_torus(major_radius: float, minor_radius: float, n_major: int, divisions: int) -> tuple[Topology, np.ndarray]:
    """Sweeping a `triangular_disk(minor_radius, divisions)` cross-section centred at `major_radius`
    in the `xz`-plane around the `z`-axis in `n_major` steps fills a solid torus with tetrahedra
    and `[V, 3]` vertex coordinates."""
    across, up, _ = planes(3)
    section, flat = triangular_disk(minor_radius, divisions)
    # Place the 2D disk cross-section in the xz-plane offset by `major_radius` along x.
    disk_points = directions(3)[0] * major_radius + placed(np.stack([flat[:, 0], np.zeros(len(flat)), flat[:, 1]], axis=-1))
    around_axis = ((across ^ up) * (-np.linspace(0.0, 2 * np.pi, n_major, endpoint=False) / 2)).exp()   # [n_major] Rotor+
    coordinates = read((around_axis[:, None] >> disk_points[None, :]).reshape(-1), planes(3))
    loop = Topology.loops([n_major])
    return outward(section.product(loop).elements[-1], coordinates), coordinates


# --- closed surfaces ------------------------------------------------------------------
def sphere(levels: int) -> tuple[Topology, np.ndarray]:
    """Subdividing an icosahedron `levels` times at edge midpoints and projecting new vertices onto
    the unit sphere produces an outward-wound triangle `Topology` and `[V, 3]` coordinates."""
    topology, coordinates = icosahedron()
    coordinates = coordinates / np.linalg.norm(coordinates, axis=-1, keepdims=True)
    for _ in range(levels):
        topology, finer = midpoint_split(topology, coordinates, 1)
        points = placed(finer)                                                  # [...] Point-
        coordinates = read(points / points.antinorm(), planes(3))
    return topology, coordinates


def torus(major_radius: float, minor_radius: float, n_major: int, n_minor: int) -> tuple[Topology, np.ndarray]:
    """Revolving an `n_minor`-gon of `minor_radius` centred at `major_radius` in the `xz`-plane
    around the `z`-axis in `n_major` steps produces a closed triangle `Topology` and `[V, 3]`
    coordinates."""
    across, up, out = planes(3)
    around_tube = ((across ^ out) * (-np.linspace(0.0, 2 * np.pi, n_minor, endpoint=False) / 2)).exp()   # [n_minor] Rotor+
    around_axis = ((across ^ up) * (-np.linspace(0.0, 2 * np.pi, n_major, endpoint=False) / 2)).exp()   # [n_major] Rotor+
    unit = directions(3)[0]                                                      # [] Point-
    tube = unit * major_radius + (around_tube >> (unit * minor_radius))         # [n_minor] Point-
    return Topology.torus(n_major, n_minor), read((around_axis[:, None] >> tube[None, :]).reshape(-1), planes(3))


def flat_torus(n_rings: int, n_sites: int) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapping a unit-edge equilateral triangular lattice of `n_rings` by `n_sites` vertices into a
    periodic torus yields `(topology, coordinates, steps, periods)` with rhombus vertex coordinates
    `[V, 2]`, unwrapped unit edge displacements `[E, 2]`, and lattice period vectors `[2, 2]`."""
    across, up = planes(2)
    along = directions(2)[0]                                                     # [] Point-
    around = ((across ^ up) * (-np.pi / 3)).exp() >> along                       # [] Point-
    topology = Topology.torus(n_rings, n_sites)
    V, E, F = topology.n_elements
    rings, sites = np.divmod(np.arange(V), n_sites)                              # [V], [V]
    # Take the minimum-image step (-1, 0, or +1) along each lattice period so seam-crossing edges keep unit length.
    tails, heads = topology.elements[1].T
    ring_steps = (rings[heads] - rings[tails] + 1) % n_rings - 1                # [E]
    site_steps = (sites[heads] - sites[tails] + 1) % n_sites - 1                # [E]
    coordinates = read(along * rings + around * sites, planes(2))               # [V, 2]
    steps = read(along * ring_steps + around * site_steps, planes(2))           # [E, 2]
    periods = concatenate((along[None] * n_rings, around[None] * n_sites))       # [2] Point-
    return topology, coordinates, steps, read(periods, planes(2))


def surface(path: Path, size: float) -> tuple[Topology, np.ndarray]:
    """Vertex (`v`) and triangle (`f`) records from a Wavefront OBJ file at `path` centre on their
    bounding box, scale its longest axis to `size`, and wind coherently outward into `(topology,
    coordinates)` with `coordinates` of shape `[V, 3]`."""
    rows = [line.split() for line in path.read_text().splitlines()]
    places = np.array([row[1:4] for row in rows if row[:1] == ["v"]], dtype=float)
    faces = np.array([[word.split("/")[0] for word in row[1:4]] for row in rows if row[:1] == ["f"]], dtype=int) - 1
    middle, extent = (places.max(axis=0) + places.min(axis=0)) / 2, np.ptp(places, axis=0).max()
    return closed(faces, (places - middle) * size / extent)


# --- regular polytopes ----------------------------------------------------------------
def polytope(schlafli: tuple[int, ...]) -> tuple[Topology, np.ndarray]:
    """Reflecting a unit-sphere seed point across the Coxeter mirrors defined by `schlafli` (such as
    `(3, 5)` for the icosahedron or `(3, 3, 5)` for the 600-cell) generates the regular polytope's
    outward-wound convex boundary `Topology` and `[V, d]` vertex coordinates.

    Adjacent mirrors meet at dihedral angles `pi / schlafli[i]` and non-adjacent mirrors are
    orthogonal; factoring alternating mutually orthogonal mirrors first via Cholesky aligns them
    with coordinate planes before closing the vertex orbit under reflection.
    """
    from scipy.spatial import cKDTree
    n_dim = len(schlafli) + 1
    orders = np.full((n_dim, n_dim), 2)
    orders[np.arange(n_dim - 1), np.arange(1, n_dim)] = orders[np.arange(1, n_dim), np.arange(n_dim - 1)] = schlafli
    cosines = -np.cos(np.pi / orders)
    np.fill_diagonal(cosines, 1.0)
    order = np.r_[0:n_dim:2, 1:n_dim:2]
    normals = np.empty((n_dim, n_dim))                                          # [n_dim, n_dim]
    normals[order] = np.linalg.cholesky(cosines[order][:, order])
    mirrors = (planes(n_dim) * normals).sum(axis=-1)                            # [n_dim] Vector+
    seed = np.linalg.solve(normals, np.eye(n_dim)[0])
    # Reflect frontier points across every mirror until no point farther than 1e-6 from known vertices appears.
    found = (seed / np.linalg.norm(seed))[None]                                 # [..., n_dim]
    frontier = found
    while len(frontier):
        reflected = read(mirrors[:, None] >> placed(frontier)[None], planes(n_dim)).reshape(-1, n_dim)
        fresh = reflected[np.isinf(cKDTree(found).query(reflected, distance_upper_bound=1e-6)[0])]
        # Deduplicate newly discovered vertices reached simultaneously by distinct mirrors in the same round.
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
