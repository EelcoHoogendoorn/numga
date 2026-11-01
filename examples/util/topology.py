"""The topology of oriented simplicial meshes: their simplices, the incidence and boundary operators between them,
their border, and the Loop step. A mesh is a topology with the geometry of its vertices; this module holds the topology
alone.

A topology is its top simplices, edges of a graph, triangles of a surface or tetrahedra of a solid, and every simplex
below them; it holds no positions. Its vertices, `Point[V]`, are the example's own, placed in its own algebra, and so
is every reading of them. Quantities on a mesh are fields over the topology's simplices, `Point[V]` or `Scalar[F]`; a
quantity per corner of each face is a `[3]` batch of face fields, one per corner, and a face's corners read off its
vertices by `topology.incidence[2, 0] * vertices`. The edges are `topology.d[0] * vertices`, and the edge facing each
corner is one of them, picked out and signed as the face runs along it, `topology.boundary[2, 1] * edges`.

`topology.incidence[n, m]` reads the m-simplices of each n-simplex, unsigned, a `[k]` batch of maps, one per m-simplex
of an n-simplex; `topology.boundary[k + 1, k]` reads the facets of each (k + 1)-simplex, signed by their orientation in
it; and `topology.d[k]`, their sum, is the exterior derivative from k-simplices to (k + 1)-simplices, so that
`topology.d[k + 1] * topology.d[k]` vanishes. Their batches line up: entry `i` of the corners is corner `i`, and from
triangles up entry `i` of the facets is the facet facing corner `i`, so that a quantity per corner reaches the facet
facing it. An edge's corners are its facets, its tail and its head, in that order. `topology.border` is the facets
bounding a single top simplex, a topology one dimension lower, wound outward.

No geometry belongs here, by design: no edge vectors, midpoints, centroids, areas, normals, angles, cotangent weights,
Whitney maps or anything else that reads the vertices, however many examples compute the same thing. The notebooks
promise to be self-contained expressions of their geometry, so each example writes those few lines itself, in view,
in the algebra; shared helpers would hide exactly the part an example exists to show. Repetition of geometry across
examples is intended and is not to be factored back into this module. What belongs here is what knows only the
simplex arrays: incidences, the maps that index by them, and subdivision of the connectivity. Placing vertices is not
the topology's: `examples.util.shapes` gives stock shapes, a sphere, a torus, disks, for an example to place.

A topology belongs to no algebra. Its maps hold plain numbers, which become scalars of whatever algebra they meet, as an
untyped array does beside an extensor: `topology.incidence[2, 0] * vertices` reads the corners of the faces of points
in the plane, in space or in spacetime alike, each in the points' own type."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, cached_property, reduce
from itertools import combinations
from operator import mul

import numpy as np

from numga import concatenate, stack
from numga.extensor import Extensor
from numga.sparse import SparseExtensor


# --- plumbing -------------------------------------------------------------------------
def selection(index: np.ndarray, size: int) -> SparseExtensor:
    """`[..., output <- size]`: the map of plain numbers that reads, at each of its output sites, the input site of
    `size` an index names, `index[..., output]`; leading axes of the index give one map each, all of one pattern, each
    map's numbers one where it reads and zero elsewhere."""
    leading, count = index.shape[:-1], index.shape[-1]
    cases = int(np.prod(leading, dtype=int))
    reads = np.broadcast_to(np.eye(cases)[:, None, :], (cases, count, cases)).reshape(leading + (count, cases))
    return SparseExtensor.from_indices(reads, np.arange(count)[:, None], index.reshape(cases, count).T, (count, size))


def generate_simplex_boundary(simplices: np.ndarray) -> np.ndarray:
    """The facets of simplices `[N, k + 1]`, `[N, k + 1, k]`: facet `i` is the simplex without its vertex `i`, the
    facet facing corner `i`, its vertices in the simplex's order."""
    # Row i of the table is the corners but corner i: column j is corner j, shifted past i once j reaches it.
    kept = np.arange(simplices.shape[-1] - 1)                                   # [k]
    table = kept[None, :] + (kept[None, :] >= np.arange(simplices.shape[-1])[:, None])   # [k + 1, k]
    return simplices[..., table]


def simplex_parity(simplices: np.ndarray) -> np.ndarray:
    """`[...]`: +1 where the vertices of each simplex `[..., k]` are in an even order, -1 where odd, counting the
    pairs out of increasing order."""
    inversions = np.triu(simplices[..., :, None] > simplices[..., None, :], 1).sum(axis=(-2, -1))   # [...]
    return np.where(inversions % 2 == 0, 1.0, -1.0)


def turned(simplices: np.ndarray) -> np.ndarray:
    """`[N, k + 1]`: the simplices run in the other sense, their first two vertices swapped; a lone vertex has no sense
    to turn."""
    order = np.arange(simplices.shape[-1])
    order[:2] = order[:2][::-1]
    return simplices[:, order]


def coherent(simplices: np.ndarray) -> np.ndarray:
    """`[N, k + 1]`: the top simplices of a manifold, some turned, so that the two simplices sharing a facet run it in
    opposite senses, as the simplices of an oriented manifold do; each connected piece keeps the sense of its first
    simplex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    topology = Topology(int(simplices.max()) + 1, simplices)
    n_simplices = len(simplices)
    neighbours, entries = topology.neighbours                                       # [N, k + 1], [N, k + 1]
    signs = topology.orientation[topology.n_dim - 1]                               # [N, k + 1]
    owners = np.broadcast_to(np.arange(n_simplices)[:, None], neighbours.shape)    # [N, k + 1]
    shared = neighbours != owners                                                  # [N, k + 1]
    # Each simplex has two senses, two sheets of copies: simplex s and s + N. Two simplices running their shared facet
    # in opposite senses agree and join on the same sheets; running it in the same sense, they join across. On an
    # orientable piece the sheets fall apart into two components, and a simplex is turned where it lies off the
    # component of its piece's first simplex.
    across = n_simplices * (signs == signs[neighbours, entries])[shared]       # [shared]
    tails, heads = owners[shared], neighbours[shared] + across                  # [shared], [shared]
    links = (np.concatenate([tails, tails + n_simplices]), np.concatenate([heads, (heads + n_simplices) % (2 * n_simplices)]))
    sheets = coo_matrix((np.ones(len(links[0])), links), shape=(2 * n_simplices,) * 2)
    _, components = connected_components(sheets, directed=False)                # [2 N]
    pieces = np.minimum(components[:n_simplices], components[n_simplices:])     # [N]
    _, firsts, of_piece = np.unique(pieces, return_index=True, return_inverse=True)
    flips = components[:n_simplices] != components[firsts[of_piece]]            # [N]
    return np.where(flips[:, None], turned(simplices), simplices)


def lookup_simplices(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """`[...]`: the row of `table` `[n, k]`, unique and sorted, holding each sorted simplex of `query` `[..., k]`."""
    _, rows = np.unique(np.concatenate([table, query.reshape(-1, table.shape[-1])]), axis=0, return_inverse=True)
    return rows.reshape(-1)[len(table):].reshape(query.shape[:-1])


def local_corners(n: int, m: int) -> np.ndarray:
    """`[m + 1, k]`: the local corners of each m-simplex of an n-simplex, in the order of `incidence[n, m]`: its
    corners themselves, the facets facing them, or between the two, the m-simplices in the order of their corners."""
    if m >= n:
        raise ValueError(f"sub-simplex dimension m={m} must be strictly less than simplex dimension n={n}")
    if m == 0:
        return np.arange(n + 1)[None, :]
    if m == n - 1:
        return generate_simplex_boundary(np.arange(n + 1)).T
    return np.array(list(combinations(range(n + 1), m + 1))).T


class ByDimensions:
    """What a topology holds for each pair of dimensions `[n, m]`, its n-simplices against its m-simplices: the maps between
    them, `topology.incidence[n, m]`, `topology.boundary[n, m]` and `topology.oriented[n, m]`, each a sparse map from fields on
    the m-simplices to fields on the n-simplices, `[n-simplices <- m-simplices]`, a `[k]` batch of maps, one per
    m-simplex of an n-simplex, with plain numbers for cells; or the local corners, `topology.corners[n, m]`. Negative
    dimensions count from the top, so `[-1, 0]` couples the top simplices to the vertices whatever the topology's
    dimension. Each entry is built by `build(n, m)` the first time it is asked for and kept, since a topology does
    not change."""

    def __init__(self, n_dim: int, build: Callable[[int, int], SparseExtensor | np.ndarray]) -> None:
        self.n_dim, self.build = n_dim, cache(build)

    def __getitem__(self, dimensions: tuple[int, int]) -> SparseExtensor | np.ndarray:
        return self.build(*(dimension % (self.n_dim + 1) for dimension in dimensions))


class Topology:
    """An oriented simplicial topology: `n_vertices` vertices and top simplices `[N, n_dim + 1]`, kept in
    their own order, which orients them, faces counter-clockwise from outside. Every simplex below the top is
    stored once, its vertices sorted, in `elements[k]` `[n_elements[k], k + 1]`; `boundary_indices[k]` and
    `orientation[k]` `[n_elements[k + 1], k + 2]` give the facets of each (k + 1)-simplex in `elements[k]` and their
    orientations in it. A topology does not change, so each map derived from it is computed once."""

    # A topology that was not subdivided has no Loop steps behind it.
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
    def incidence(self) -> ByDimensions:
        """`incidence[n, m]`, `[k] numbers[n-simplices <- m-simplices]`: the m-simplices of each n-simplex, unsigned,
        a batch of k selection maps, one per m-simplex of an n-simplex, for any `0 <= m < n <= n_dim`; negative
        dimensions count from the top, `incidence[-1, 0]`.

        Entry `i` of the corners, `incidence[n, 0]`, is corner `i` in the simplex's own order: an edge's tail then
        its head, a face's corners in its winding. From triangles up, entry `i` of the facets, `incidence[n, n - 1]`,
        is the facet facing corner `i`, so that a quantity per corner reaches the facet across from it. Between the
        two, as a tet's six edges in `incidence[3, 1]`, the m-simplices come in the order of their corners, (0, 1),
        (0, 2), (0, 3), (1, 2), (1, 3), (2, 3). Applied to a field, `topology.incidence[2, 0] * vertices` is each face's
        corners, `[3] Point-[faces]`; summed over the batch, `topology.incidence[2, 0].mean(axis=0) * vertices` is its
        centroid. Each map is built on first use."""
        return ByDimensions(self.n_dim, lambda n, m: selection(self._local(n, m).T, self.n_elements[m]))

    @cached_property
    def boundary(self) -> ByDimensions:
        """`boundary[k + 1, k]`, `[k + 2] numbers[(k + 1)-simplices <- k-simplices]`: the facets of each (k + 1)-simplex,
        the batches of `incidence[k + 1, k]`, each signed +1 or -1 by the facet's orientation in the simplex: an
        edge's tail -1 and head +1, a face's edges +1 where the face's winding runs along them, a tet's faces +1
        where they face outward. Only adjacent dimensions have a boundary; summed over the batch, it is the exterior
        derivative `d[k]`. Each map is built on first use."""
        def signed_facets(n: int, m: int) -> SparseExtensor:
            if m != n - 1:
                raise ValueError(f"only adjacent dimensions have a boundary, not {n} and {m}")
            return self._signed(n, m, self.orientation[m])
        return ByDimensions(self.n_dim, signed_facets)

    @cached_property
    def oriented(self) -> ByDimensions:
        """`oriented[n, m]`, `[k] numbers[n-simplices <- m-simplices]`: the m-simplices of each n-simplex, the batches
        of `incidence[n, m]`, each signed +1 where the m-simplex as stored, its vertices in increasing order, runs the
        way the n-simplex's own vertex order runs it, and -1 where it runs the other way. A local formula written in
        a simplex's own vertex order, such as a Whitney form on a tet's edge from its local tail to its local head,
        reaches the stored m-simplices through these signs. A facet's sign here differs from its sign in `boundary`
        by `(-1)**i` for the facet facing corner `i`. Each map is built on first use."""
        return ByDimensions(self.n_dim, lambda n, m: self._signed(n, m, simplex_parity(self.elements[n][:, self.corners[n, m].T])))

    @cached_property
    def d(self) -> tuple[SparseExtensor, ...]:
        """`d[k]`, `numbers[(k + 1)-simplices <- k-simplices]` for each k below the top: the exterior derivative, the
        signed sum `boundary[k + 1, k].sum(axis=0)` over the facets of each simplex. `d[0] * vertices` is the edges,
        each its head less its tail, and `d[k + 1] * d[k]` vanishes."""
        return tuple(self.boundary[k + 1, k].sum(axis=0) for k in range(self.n_dim))

    @cached_property
    def corners(self) -> ByDimensions:
        """`corners[n, m]`, `[m + 1, k]`: the local corners of each m-simplex of an n-simplex, in the order of
        `incidence[n, m]`. Unpacks directly along axis 0: on edges, `tails, heads = topology.corners[n, 1]`, and on a tet's
        faces, `first, second, third = topology.corners[3, 2]`."""
        return ByDimensions(self.n_dim, local_corners)

    def _local(self, n: int, m: int) -> np.ndarray:
        """`[N, k]`: the index in `elements[m]` of each m-simplex of each n-simplex, in the order of
        `incidence[n, m]`."""
        if m == 0:
            return self.elements[n]
        if m == n - 1:
            return self.boundary_indices[m]
        return lookup_simplices(self.elements[m], np.sort(self.elements[n][:, self.corners[n, m].T], axis=-1))

    def _signed(self, n: int, m: int, signs: np.ndarray) -> SparseExtensor:
        """`[k] numbers[n-simplices <- m-simplices]`: the batches of `incidence[n, m]`, each entry signed by
        `signs` `[N, k]`."""
        local = self._local(n, m)                                                  # [N, k]
        cells = np.eye(local.shape[-1])[:, None, :] * signs[None, :, :]            # [k, N, k]
        return SparseExtensor.from_indices(cells, np.arange(self.n_elements[n])[:, None], local,
                                           (self.n_elements[n], self.n_elements[m]))

    @cached_property
    def exterior(self) -> tuple[np.ndarray, ...]:
        """`exterior[k]`: indices of the k-simplices on the border, for each dimension: the facets that bound a single top
        simplex, and every simplex of theirs; no top simplex lies on it."""
        top = self.n_dim - 1
        on_border = [np.flatnonzero(np.bincount(self.boundary_indices[top].ravel(), minlength=self.n_elements[top]) == 1)]
        for k in range(top, 0, -1):
            on_border.insert(0, np.unique(self.boundary_indices[k - 1][on_border[0]]))
        return (*on_border, np.zeros(0, dtype=int))

    @cached_property
    def interior(self) -> tuple[np.ndarray, ...]:
        """`interior[k]`: indices of the k-simplices off the border, for each dimension: all but those of `exterior[k]`."""
        return tuple(np.setdiff1d(np.arange(count), on_border) for count, on_border in zip(self.n_elements, self.exterior))

    def subset(self, chosen: np.ndarray, k: int = -1) -> Topology:
        """The topology of the chosen k-simplices, the top ones unless asked otherwise, a boolean mask over them or their
        indices, on the same vertices, so that every field on this topology's vertices reads on it unchanged: the
        interior edges of a surface as a truss, `topology.subset(topology.interior[1], 1)`, or a region of its faces."""
        return Topology(self.n_vertices, self.elements[k][chosen])

    def cone(self) -> Topology:
        """This topology joined to one new vertex, its apex, numbered after the others: each top simplex with the apex
        before its corners, one dimension up. A sphere's cone is a ball, a polygon's a fan."""
        apex = np.full((len(self.elements[-1]), 1), self.n_vertices)
        return Topology(self.n_vertices + 1, np.concatenate([apex, self.elements[-1]], axis=-1))

    def product(self, path: Topology) -> Topology:
        """This topology swept along a path, a topology of edges: a copy of the vertices at each of the path's vertices,
        numbered copy after copy, and each top simplex times each edge of the path cut into simplices one dimension up,
        the staircase its sorted vertices climb from the edge's tail to its head, so that neighbouring prisms share
        their sides. Each step is wound as the product of the simplex's own winding with the edge's, so that the
        product of two coherently wound topologies is coherently wound. A disk swept around a closed loop is a solid
        torus, a sphere along a path a thick shell, a loop along a path a cylinder."""
        simplices = np.sort(self.elements[-1], axis=-1)                         # [N, k + 1]
        width = simplices.shape[-1]
        # Step j of the staircase keeps corners 0 to j on the tail's copy and takes corners j to k on the head's.
        places, steps = np.arange(width + 1), np.arange(width)[:, None]
        on_head = places > steps                                                # [steps, k + 2]
        corners = places - on_head                                              # [steps, k + 2]
        tails, heads = path.elements[1].T                                       # [E], [E]
        copies = np.where(on_head, heads[:, None, None], tails[:, None, None])  # [E, steps, k + 2]
        cells = simplices[:, None, corners] + self.n_vertices * copies[None]   # [N, E, steps, k + 2]
        # The prism's boundary-compatible winding: step j signed (-1)**j, times the simplex's own winding against its
        # sorted order.
        signs = simplex_parity(self.elements[-1])[:, None, None] * (-1.0) ** np.arange(width)   # [N, 1, steps]
        wound = np.broadcast_to(signs, cells.shape[:-1]).reshape(-1) > 0      # [N E steps]
        cells = cells.reshape(-1, width + 1)
        return Topology(self.n_vertices * path.n_vertices, np.where(wound[:, None], cells, turned(cells)))

    @cached_property
    def border(self) -> Topology:
        """The facets that bound a single top simplex, as a topology one dimension lower on the same vertices, each
        wound outward, as the simplex it bounds runs it: a solid's border is its closed surface, facing out, and a
        disk's its rim. Empty where the topology is closed, so that a border's own border is always empty."""
        top = self.n_dim - 1
        facets, signs = self.boundary_indices[top], self.orientation[top]        # [N, n_dim + 1], [N, n_dim + 1]
        lone = np.bincount(facets.ravel(), minlength=self.n_elements[top])[facets] == 1   # [N, n_dim + 1]
        # A stored facet's vertices are sorted: it runs outward where its simplex runs it that way, and is turned where
        # the simplex runs it the other way.
        sides = self.elements[top][facets[lone]]                                 # [border, n_dim]
        return Topology(self.n_vertices, np.where((signs[lone] > 0)[:, None], sides, turned(sides)))

    @cached_property
    def is_closed(self) -> bool:
        """Whether the border is empty: every facet bounds two top simplices, as on a sphere or a torus."""
        return len(self.border.elements[-1]) == 0

    @cached_property
    def is_oriented(self) -> bool:
        """Whether every facet two top simplices share is run in opposite senses by them, as on an oriented manifold."""
        neighbours, entries = self.neighbours                                    # [N, n_dim + 1], [N, n_dim + 1]
        signs = self.orientation[self.n_dim - 1]                                 # [N, n_dim + 1]
        shared = neighbours != np.arange(len(neighbours))[:, None]               # [N, n_dim + 1]
        return bool(np.all(signs[shared] != signs[neighbours, entries][shared]))

    @cached_property
    def is_manifold(self) -> bool:
        """Whether every facet bounds at most two top simplices, and the top simplices around every vertex hang
        together through the facets they share at it: no facet where three meet, and no vertex where two pieces only
        touch."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        top = self.n_dim - 1
        simplices, width = self.elements[-1], self.n_dim + 1                     # [N, n_dim + 1]
        thin = np.bincount(self.boundary_indices[top].ravel(), minlength=self.n_elements[top]).max(initial=0) <= 2
        # Each corner of each top simplex is a node; two simplices sharing a facet join their corners at each of its
        # vertices, so that the corners at a vertex fall into one piece where its star hangs together.
        neighbours, entries = self.neighbours                                    # [N, n_dim + 1], [N, n_dim + 1]
        owners = np.broadcast_to(np.arange(len(simplices))[:, None], neighbours.shape)
        shared = neighbours != owners                                            # [N, n_dim + 1]
        on_facet = local_corners(self.n_dim, top).T[np.nonzero(shared)[1]]       # [pairs, n_dim] corners of the owner
        owner, other = owners[shared], neighbours[shared]                        # [pairs], [pairs]
        at = simplices[owner[:, None], on_facet]                                 # [pairs, n_dim] the facet's vertices
        matching = (simplices[other][:, None, :] == at[:, :, None]).argmax(axis=-1)   # [pairs, n_dim] corners of the other
        links = ((owner[:, None] * width + on_facet).ravel(), (other[:, None] * width + matching).ravel())
        graph = coo_matrix((np.ones(len(links[0])), links), shape=(simplices.size,) * 2)
        _, pieces = connected_components(graph, directed=False)                  # [N (n_dim + 1)]
        stars = np.unique(np.stack([simplices.ravel(), pieces], axis=-1), axis=0)[:, 0]   # a vertex per piece of its star
        return bool(thin and np.all(np.bincount(stars) <= 1))

    @cached_property
    def n_components(self) -> int:
        """How many connected pieces the top simplices fall into, joined through the facets they share."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        neighbours = self.neighbours[0]                                          # [N, n_dim + 1]
        owners = np.broadcast_to(np.arange(len(neighbours))[:, None], neighbours.shape)
        graph = coo_matrix((np.ones(neighbours.size), (owners.ravel(), neighbours.ravel())), shape=(len(neighbours),) * 2)
        return int(connected_components(graph, directed=False)[0])

    @cached_property
    def euler_characteristic(self) -> int:
        """The alternating sum of the counts of the simplices, the vertices of the top simplices less the edges, plus the
        faces and so on: 2 for a sphere's surface, 0 for a torus's or a closed loop, 1 for a disk or a ball."""
        counts = np.array([len(np.unique(self.elements[-1])), *self.n_elements[1:]])
        return int(np.sum((-1) ** np.arange(self.n_dim + 1) * counts))

    @classmethod
    def closed(cls, simplices: np.ndarray, coordinates: np.ndarray) -> Topology:
        """A closed topology of the top simplices `[N, k + 1]` on vertices at the coordinates `[V, k + 1]`, whatever sense
        the simplices come in: wound coherently, and counter-clockwise seen from outside, so that the volume they
        enclose is positive. The winding is settled on the coordinates, at construction, before they become points."""
        wound = coherent(simplices)
        # The enclosed volume: the cone each simplex spans from the origin, its corners' determinant, summed.
        volume = np.linalg.det(coordinates[wound]).sum()
        return cls(len(coordinates), np.where(volume < 0, turned(wound), wound))

    @classmethod
    def rings(cls, n_rings: int, n_sectors: int) -> Topology:
        """A disk of rings: a centre vertex, then `n_rings` rings of `n_sectors` vertices each, numbered ring by ring;
        a fan of triangles about the centre, and two triangles to each cell between neighbouring rings, all wound
        alike, from each ring's vertex toward the next."""
        indices = np.arange(n_rings * n_sectors).reshape(n_rings, n_sectors) + 1
        following = np.roll(indices, -1, axis=1)
        fan = np.stack([np.zeros(n_sectors, dtype=int), indices[0], following[0]], axis=-1)
        inner, outer = indices[:-1], indices[1:]
        inner_next, outer_next = following[:-1], following[1:]
        first = np.stack([inner, outer, outer_next], axis=-1).reshape(-1, 3)
        second = np.stack([inner, outer_next, inner_next], axis=-1).reshape(-1, 3)
        return cls(1 + n_rings * n_sectors, np.concatenate([fan, first, second]))

    @classmethod
    def torus(cls, n_major: int, n_minor: int) -> Topology:
        """A closed torus of `n_major` rings of `n_minor` vertices each, numbered ring by ring, two triangles to each
        cell, wound alike."""
        u, v = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
        v00 = u * n_minor + v
        v10 = ((u + 1) % n_major) * n_minor + v
        v01 = u * n_minor + (v + 1) % n_minor
        v11 = ((u + 1) % n_major) * n_minor + (v + 1) % n_minor
        faces = np.concatenate([np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
                                np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)])
        return cls(n_major * n_minor, faces)

    @classmethod
    def delaunay(cls, coordinates: np.ndarray) -> Topology:
        """Delaunay topology of points in a plane, given by their coordinates `[V, 2]`, with counter-clockwise
        faces."""
        from scipy.spatial import Delaunay
        return cls(len(coordinates), Delaunay(coordinates).simplices)

    @classmethod
    def chains(cls, lengths: list[int]) -> Topology:
        """Chains of edges through runs of vertices numbered one run after another, each run `lengths` long: a set of
        polylines as a topology of edges, each edge from a vertex to the next in its run."""
        lengths = np.asarray(lengths, dtype=int)
        n_vertices = int(lengths.sum())
        # An edge from each vertex to the next, but where the next starts a run of its own.
        starts = np.zeros(n_vertices + 1, dtype=bool)
        starts[np.cumsum(lengths) - lengths] = True
        tails = np.flatnonzero(~starts[1:n_vertices])                             # [edges]
        return cls(n_vertices, np.stack([tails, tails + 1], axis=-1))

    @classmethod
    def loops(cls, lengths: list[int]) -> Topology:
        """Closed loops of edges through runs of vertices numbered one run after another, each run `lengths` long: a set
        of polygons as a topology of edges, each edge from a vertex to the next in its run and the last back to the
        first."""
        lengths = np.asarray(lengths, dtype=int)
        n_vertices = int(lengths.sum())
        tails = np.arange(n_vertices)                                            # [edges]
        firsts, sizes = np.repeat(np.cumsum(lengths) - lengths, lengths), np.repeat(lengths, lengths)   # [edges], [edges]
        return cls(n_vertices, np.stack([tails, firsts + (tails - firsts + 1) % sizes], axis=-1))

    @classmethod
    def grid(cls, n_rows: int, n_columns: int) -> Topology:
        """Two triangles to each cell of a rectangular grid of vertices numbered row by row, wound alike."""
        vertices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
        cells = np.stack((vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1]), axis=-1)
        return cls(n_rows * n_columns, cells[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))

    @classmethod
    def triangulate(cls, vertices: Extensor) -> Topology:
        """Delaunay topology for vertices `[V]` in the plane of x and y of their own algebra, with counter-clockwise
        faces."""
        # Positions read by pairing with the planes across x and up y, in the vertices' own algebra.
        mv = vertices.context.multivector
        return cls.delaunay((stack((mv.x, mv.y)) & vertices.batch()[..., None]).to_array())

    def reading(self, indices: np.ndarray, k: int = 0) -> SparseExtensor:
        """`numbers[..., n <- k-simplices]`: the map reading a field on the k-simplices, the vertices unless asked
        otherwise, at each of the given ones `[..., n]`; its reverse adds values at them back onto the simplices. On
        the interior edges, `topology.reading(topology.interior[1], 1)`."""
        return selection(indices, self.n_elements[k])

    def loop(self, pinned: np.ndarray) -> SparseExtensor:
        """numbers[V + E <- V]: one Loop step, from the vertices to those of the finer topology, its vertices
        followed by one per edge, with the vertices pinned `[V]` staying where they are.

        An interior vertex of valence n keeps `1 - n * beta` of itself and takes `beta` of each neighbour,
        with `beta = (5/8 - (3/8 + cos(2 pi / n) / 4)**2) / n`. An interior edge's vertex takes 3/8 of each
        end and 1/8 of each vertex across the edge: each face beside the edge gives 1/8 of the corner the
        edge faces and 3/16 of each of the other two. The boundary follows itself alone: a boundary vertex
        keeps 3/4 of itself and takes 1/8 of each neighbour along the boundary, and a boundary edge's vertex
        is the midpoint of its ends."""
        vertex_count = len(pinned)
        edges, faces, face_edges = self.elements[1], self.elements[2], self.boundary_indices[1]   # [E, 2], [F, 3], [F, 3]
        shape = (vertex_count + len(edges), vertex_count)
        everyone = np.arange(vertex_count)
        rim = np.isin(np.arange(len(edges)), self.exterior[1])                   # [E]
        on_rim = np.isin(everyone, edges[rim])                                   # [V]
        valence = np.bincount(edges.reshape(-1), minlength=vertex_count)         # [V]
        beta = (5 / 8 - (3 / 8 + np.cos(2 * np.pi / valence) / 4) ** 2) / valence
        own = np.where(pinned, 1.0, np.where(on_rim, 3 / 4, 1 - valence * beta)) # [V]
        keep = SparseExtensor.from_indices(own, everyone, everyone, shape)
        # each end's share in the other end, by the end's own rule
        along = np.where(pinned[edges], 0.0,
                         np.where(on_rim[edges], np.where(rim[:, None], 1 / 8, 0.0), beta[edges]))   # [E, 2]
        neighbours = SparseExtensor.from_indices(along, edges, edges[:, ::-1], shape)
        # each face's share in the vertex of the edge facing each corner, from each of its corners
        facing_rim = rim[face_edges][:, :, None]                                 # [F, 3, 1]
        shares = np.where(np.eye(3, dtype=bool), np.where(facing_rim, 0.0, 1 / 8), np.where(facing_rim, 1 / 2, 3 / 16))
        split = SparseExtensor.from_indices(shares, vertex_count + face_edges[:, :, None], faces[:, None, :], shape)
        return keep + neighbours + split

    @cached_property
    def split_faces(self) -> np.ndarray:
        """[4 F, 3]: every face split in four at the vertices of its edges, the vertices of the finer topology: a
        middle face through them, and a face at each of its corners."""
        middle = self.n_vertices + self.boundary_indices[1]                     # [F, 3]
        corners = np.stack([self.elements[2], np.roll(middle, 1, axis=1), np.roll(middle, -1, axis=1)], axis=-1)   # [F, 3, 3]
        return np.concatenate([middle, corners.reshape(-1, 3)])

    def midpoint_split(self, vertices: Extensor, n_levels: int) -> tuple[Topology, Extensor]:
        """The topology with every face split in four at the midpoints of its edges, `n_levels` times over, and its
        vertices: the coarse vertices followed, level by level, by the midpoints, halfway along each edge, so that
        the refined mesh lies exactly on the coarse one."""
        topology = self
        for _ in range(n_levels):
            midpoints = topology.incidence[1, 0].mean(axis=0) * vertices   # Point-[E]
            vertices = concatenate([vertices.batch(), midpoints.batch()]).field()  # Point-[V + E]
            topology = Topology(len(vertices.batch()), topology.split_faces)
        return topology, vertices

    def loop_subdivided(self, pinned: np.ndarray, n_steps: int) -> SubdividedTopology:
        """The topology after `n_steps` Loop steps of this one, the vertices pinned `[V]` staying where they are.
        Each finer topology holds the one before it and the map from its vertices, so the steps compose."""
        topology = SubdividedTopology(self, pinned)
        for _ in range(n_steps - 1):
            topology = SubdividedTopology(topology, topology.pinned)
        return topology

    @cached_property
    def dual(self) -> DualTopology:
        """The dual topology: a vertex for each top simplex, an edge across each interior facet, and above them a cell
        around each simplex of lower dimension. `topology.dual.incidence[1, 0] * centres` reads a field on the top
        simplices at both sides of each interior facet."""
        return DualTopology(self)

    @cached_property
    def facet_opposites(self) -> np.ndarray:
        """`[interior facets, 2]`: the vertex facing each interior facet in each of the two top simplices its
        `dual` edge joins: on a surface, the corners across each interior edge, which with the edge's ends
        make the hinge it is."""
        # The corner facing each facet of a simplex: of all its corners, the one the facet lacks.
        corners = local_corners(self.n_dim, self.n_dim - 1)                      # [n_dim, n_dim + 1]
        facing = self.n_dim * (self.n_dim + 1) // 2 - corners.sum(axis=0)        # [n_dim + 1]
        return self._facet_sides(self.elements[-1][:, facing])

    @cached_property
    def neighbours(self) -> tuple[np.ndarray, np.ndarray]:
        """For each top simplex and each of its facets, the simplex on the other side of that facet, and the corner of
        that simplex which faces the same facet: two index arrays `[N, n_dim + 1]`, the facet named by the corner
        facing it, as everywhere in the topology. A walk leaving a simplex through a facet reads from them the simplex it
        enters and the facet it enters through. A facet on the boundary has no other side, and names the simplex
        itself."""
        width = self.n_dim + 1
        facets = self.boundary_indices[self.n_dim - 1].reshape(-1)               # [N (n_dim + 1)]
        order = np.argsort(facets, kind="stable")
        shared = facets[order][1:] == facets[order][:-1]
        first, second = order[:-1][shared], order[1:][shared]
        across = np.arange(len(facets))
        across[first], across[second] = second, first
        return (across // width).reshape(-1, width), (across % width).reshape(-1, width)

    @cached_property
    def vertex_simplices(self) -> np.ndarray:
        """`[V]`: the first top simplex at each vertex, of those with the vertex as a corner."""
        simplices = self.elements[-1]
        first = np.full(self.n_vertices, len(simplices))
        np.minimum.at(first, simplices.reshape(-1), np.repeat(np.arange(len(simplices)), simplices.shape[-1]))
        return first

    @cached_property
    def spanning_tree(self) -> tuple[np.ndarray, np.ndarray]:
        """Every vertex but the first `[V - 1]`, and the edge joining each to its parent `[V - 1]`, in a
        breadth-first tree from the first vertex."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order
        edges = self.elements[1]                                                 # [E, 2]
        graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(self.n_vertices,) * 2)
        order, parents = breadth_first_order(graph.tocsr(), 0, directed=False)
        children = order[1:]
        low, high = np.minimum(children, parents[children]), np.maximum(children, parents[children])
        return children, np.searchsorted(edges[:, 0] * self.n_vertices + edges[:, 1], low * self.n_vertices + high)

    def circulation(self, loops: list[np.ndarray]) -> SparseExtensor:
        """`numbers[F + L <- E]`: the circulation of edge values around each face, counter-clockwise, the signed
        face-edge incidence, then around each given closed loop of vertices `[steps]`, in its own order."""
        lengths = np.array(list(map(len, loops)), dtype=int)                      # [loops]
        tails = np.concatenate(loops + [np.zeros(0, dtype=int)])                  # [steps]
        # Each step's head: the next vertex of its loop, the first again after the last.
        firsts = np.repeat(np.cumsum(lengths) - lengths, lengths)                 # [steps]
        heads = tails[firsts + (np.arange(len(tails)) - firsts + 1) % np.repeat(lengths, lengths)]
        edges, n_faces = self.elements[1], self.n_elements[2]                   # [E, 2]
        keys = edges[:, 0] * self.n_vertices + edges[:, 1]                       # [E], sorted
        low, high = np.minimum(tails, heads), np.maximum(tails, heads)
        rows = np.concatenate([np.repeat(np.arange(n_faces), 3), n_faces + np.repeat(np.arange(len(loops)), lengths)])
        columns = np.concatenate([self.boundary_indices[1].reshape(-1), np.searchsorted(keys, low * self.n_vertices + high)])
        signs = np.concatenate([self.orientation[1].reshape(-1), np.where(tails < heads, 1.0, -1.0)])
        return SparseExtensor.from_indices(signs, rows, columns, (n_faces + len(loops), len(edges)))

    def _facet_sides(self, per_facet: np.ndarray) -> np.ndarray:
        """`[interior facets, 2]`: a value per facet of each top simplex `[N, n_dim + 1]`, in the order of
        `boundary_indices[-1]`, gathered onto the facet, first where the simplex runs it along its stored order and then
        where it runs it against."""
        top = self.n_dim - 1
        facets, along = self.boundary_indices[top], self.orientation[top] > 0   # [N, n_dim + 1], [N, n_dim + 1]
        first, second = np.empty((2, self.n_elements[top]), dtype=per_facet.dtype)
        first[facets[along]], second[facets[~along]] = per_facet[along], per_facet[~along]
        return np.stack([first, second], axis=-1)[self.interior[top]]


class DualTopology:
    """The dual of a simplicial topology: a k-cell for each (n_dim - k)-simplex, so that its counts run the other way
    and its derivatives are the primal's transposed, `dual.d[k] = ~topology.d[n_dim - 1 - k]`. A dual vertex stands for
    each top simplex, a dual edge crosses each facet, and so on up to a dual top cell around each vertex, each numbered
    as the simplex it stands for. Its derivatives follow each other to nothing, as the primal's do; a dual edge across
    a border facet has one end, the simplex the facet bounds, and the dual is closed only where the primal is.

    The dual holds operators alone: no positions, and no simplices, since its cells above the edges are not simplices.
    Where its vertices lie is geometry, the example's to say."""

    def __init__(self, primal: Topology) -> None:
        self.primal, self.n_dim = primal, primal.n_dim
        self.n_elements = primal.n_elements[::-1]

    @cached_property
    def d(self) -> tuple[SparseExtensor, ...]:
        """`d[k]`, `numbers[dual (k + 1)-cells <- dual k-cells]`: the dual exterior derivative, the primal's transposed,
        `~topology.d[n_dim - 1 - k]`."""
        return tuple(~derivative for derivative in self.primal.d[::-1])

    @cached_property
    def incidence(self) -> ByDimensions:
        """`incidence[1, 0]`, `[2] numbers[dual edges <- dual vertices]`: the two ends of each dual edge, its tail and
        its head, where `d[0]` reads it -1 and +1, so that their difference is `d[0]`: the simplices that run the facet
        against and along its stored order. A dual edge across a border facet has its head or its tail alone. The
        dual's cells above its edges have no fixed count of faces, so the edges' ends are all its incidence holds."""
        def ends(n: int, m: int) -> SparseExtensor:
            if (n, m) != (1, 0):
                raise ValueError(f"the dual holds the ends of its edges alone, not its {n}-cells' {m}-cells")
            facets, signs = self.primal.boundary_indices[-1], self.primal.orientation[-1]   # [N, n_dim + 1], [N, n_dim + 1]
            owners = np.broadcast_to(np.arange(len(facets))[:, None], facets.shape)        # [N, n_dim + 1]
            # Each top simplex is the tail of the dual edges across the facets it runs against, the head of the others.
            cells = np.stack([signs < 0, signs > 0]).astype(float).reshape(2, -1)      # [2, N (n_dim + 1)]
            return SparseExtensor.from_indices(cells, facets.ravel(), owners.ravel(), (self.n_elements[1], self.n_elements[0]))
        return ByDimensions(self.n_dim, ends)

    def reading(self, indices: np.ndarray, k: int = 0) -> SparseExtensor:
        """`numbers[..., n <- dual k-cells]`: the map reading a field on the dual k-cells at each of the given ones
        `[..., n]`; on the dual edges across the interior facets, `topology.dual.reading(topology.interior[-2], 1)`."""
        return selection(indices, self.n_elements[k])


class SubdividedTopology(Topology):
    """The finer topology of one Loop step of a coarser one, every face split in four. It holds the coarser topology, the
    map from the coarser topology's vertices to its own, `from_coarser`, and its pinned vertices, the coarser one's pinned
    vertices followed by the free vertices of its edges, so that it can be subdivided again."""

    def __init__(self, coarser: Topology, pinned: np.ndarray) -> None:
        self.coarser = coarser
        self.from_coarser = coarser.loop(pinned)                                  # numbers[V + E <- V]
        self.pinned = np.concatenate([pinned, np.zeros(coarser.n_elements[1], dtype=bool)])   # [V + E]
        super().__init__(self.from_coarser.shape[0], coarser.split_faces)

    @property
    def steps(self) -> tuple[SparseExtensor, ...]:
        """The Loop steps from the first topology to this one, coarsest first."""
        return self.coarser.steps + (self.from_coarser,)

    @cached_property
    def subdivision(self) -> SparseExtensor:
        """The map from the first topology's vertices to this one's: its Loop steps composed, the finest acting last."""
        return reduce(mul, reversed(self.steps))
