"""An oriented simplicial complex is determined by its vertex count and top simplices (edges,
triangles, tetrahedra, or higher-dimensional simplices). Top simplices keep the caller's vertex
order to fix their orientation, while every lower simplex is deduplicated in sorted vertex order
with induced `+1` and `-1` facet signs. Because `Topology` stores integer connectivity alone and
belongs to no geometric algebra, its `Sparse` operators act linearly on fields in any algebra while
preserving their `GAType`.

`incidence` selects the sub-simplices of each simplex, `boundary` weights facets by their induced
orientation, summing over facets gives the discrete exterior derivative `d` (`d[k + 1] * d[k] == 0`),
and `reading` restricts or scatters fields on chosen subsets via `.adjugate()`. Negative dimension
indices count down from `n_dim` so `[-1, 0]` couples top simplices to vertices in any dimension,
and above dimension one facet `i` stands opposite corner `i` so corner values and opposite-facet
simplex gradients align along the local batch axis.

`dual` reverses the simplex grading by codimension so top simplices become dual vertices, facets
become dual edges, codimension-two ridges or hinges become dual faces, and primal vertices become
dual cells. Because dual elements above edges are general polytopes with variable face counts rather
than simplices, `DualTopology` carries coboundary operators directly as the adjugates of the
complementary primal coboundaries (`dual.d[k] == primal.d[n_dim - 1 - k].adjugate()`).

Topological queries and constructions operate purely on simplex incidences: `border`, `exterior`,
and `interior` separate facets belonging to a single top simplex from shared interior facets,
`orient_coherently` aligns connected orientable components across a two-sheeted orientation cover,
`cone` and `product` extend complexes in dimension, and `loop_subdivided` composes coarse-to-fine
Loop refinement maps on triangulated surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache, cached_property, reduce
from itertools import combinations
from operator import mul

import numpy as np

from numga.sparse.numeric import Sparse


# --- plumbing -------------------------------------------------------------------------


def generate_simplex_boundary(simplices: np.ndarray) -> np.ndarray:
    """Facet `i` omits corner `i`, giving the facet opposite corner `i` in the simplex's vertex order."""
    n_corners = simplices.shape[-1]
    # Row i keeps every corner except corner i by shifting indices at or above i up by one.
    kept = np.arange(n_corners - 1)                                              # [n_corners - 1]
    table = kept[None, :] + (kept[None, :] >= np.arange(n_corners)[:, None])     # [n_corners, n_corners - 1]
    return simplices[..., table]


def simplex_parity(simplices: np.ndarray) -> np.ndarray:
    """Even vertex-order permutations relative to increasing order carry `+1.0` and odd ones `-1.0`."""
    inversions = np.triu(simplices[..., :, None] > simplices[..., None, :], 1).sum(axis=(-2, -1))   # [...]
    return np.where(inversions % 2 == 0, 1.0, -1.0)


def reverse_orientation(simplices: np.ndarray) -> np.ndarray:
    """Swapping the first two corners reverses orientation on every simplex above dimension zero."""
    order = np.arange(simplices.shape[-1])
    order[:2] = order[:2][::-1]
    return simplices[:, order]


def orient_coherently(simplices: np.ndarray) -> np.ndarray:
    """Reorienting connected components of an orientable manifold gives shared facets opposite
    induced orientations across their two incident top simplices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    topology = Topology(int(simplices.max()) + 1, simplices)
    N = topology.n_elements
    n_dim = topology.n_dim
    neighbours, entries = topology.neighbours                                    # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
    signs = topology.orientation[n_dim - 1]                                      # [N[-1], n_dim + 1]
    owners = np.broadcast_to(np.arange(N[-1])[:, None], neighbours.shape)        # [N[-1], n_dim + 1]
    shared = neighbours != owners                                                # [N[-1], n_dim + 1]
    # Build a two-sheeted orientation cover (s and s + N[-1]): neighbours inducing opposite facet
    # orientations connect on the same sheet, whereas matching facet orientations cross sheets.
    across = N[-1] * (signs == signs[neighbours, entries])[shared]
    tails, heads = owners[shared], neighbours[shared] + across
    links = (np.concatenate([tails, tails + N[-1]]), np.concatenate([heads, (heads + N[-1]) % (2 * N[-1])]))
    sheets = coo_matrix((np.ones(len(links[0])), links), shape=(2 * N[-1],) * 2)
    _, components = connected_components(sheets, directed=False)                 # [2 * N[-1]]
    pieces = np.minimum(components[:N[-1]], components[N[-1]:])                  # [N[-1]]
    _, firsts, of_piece = np.unique(pieces, return_index=True, return_inverse=True)
    flips = components[:N[-1]] != components[firsts[of_piece]]                   # [N[-1]]
    return np.where(flips[:, None], reverse_orientation(simplices), simplices)


def lookup_simplices(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    _, rows = np.unique(np.concatenate([table, query.reshape(-1, table.shape[-1])]), axis=0, return_inverse=True)
    return rows.reshape(-1)[len(table):].reshape(query.shape[:-1])


def local_complements(simplex_dim: int, sub_dim: int) -> np.ndarray:
    # Sort each column stably by membership so corners absent from the sub-simplex come first.
    on = (np.arange(simplex_dim + 1)[:, None, None] == local_corners(simplex_dim, sub_dim)[None]).any(axis=1)
    return np.argsort(on, axis=0, kind="stable")[:simplex_dim - sub_dim]


def local_corners(simplex_dim: int, sub_dim: int) -> np.ndarray:
    if sub_dim >= simplex_dim:
        raise ValueError(
            f"sub-simplex dimension m={sub_dim} must be strictly less than simplex dimension n={simplex_dim}"
        )
    if sub_dim == 0:
        return np.arange(simplex_dim + 1)[None, :]
    if sub_dim == simplex_dim - 1:
        return generate_simplex_boundary(np.arange(simplex_dim + 1)).T
    return np.array(list(combinations(range(simplex_dim + 1), sub_dim + 1))).T


class ByDimensions:
    """Negative dimension indices count down from `n_dim`, so `[-1, 0]` couples top simplices to
    vertices in any dimension."""

    def __init__(self, n_dim: int, build: Callable) -> None:
        self.n_dim, self.build = n_dim, cache(build)

    def __getitem__(self, dimensions: tuple[int, int]) -> Sparse | np.ndarray:
        return self.build(*(dimension % (self.n_dim + 1) for dimension in dimensions))


class Topology:
    """Top simplices keep the caller's vertex order to fix their orientation, while lower simplices
    are deduplicated in sorted vertex order with induced `+1` and `-1` facet signs."""

    steps = ()

    def __init__(self, n_vertices: int, simplices: np.ndarray) -> None:
        simplices = np.asarray(simplices)
        self.n_vertices, self.n_dim = n_vertices, simplices.shape[-1] - 1
        elements, boundary_indices, orientation = {self.n_dim: simplices}, {}, {}
        for k in range(self.n_dim, 0, -1):
            facets = generate_simplex_boundary(elements[k])
            # Facet i carries boundary sign (-1)**i times the parity of its vertex order.
            signs = (-1.0) ** np.arange(k + 1) * simplex_parity(facets)
            if k == 1:
                # Order an edge's two boundary vertices as tail (-1) then head (+1).
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
        """Unsigned incidence operators read the `m`-simplices of each `n`-simplex, so
        `incidence[2, 0] * vertices` gathers the three corner values onto every triangle. Corners
        follow the simplex's vertex order at `m == 0`, facets stand opposite corner `i` at
        `m == n - 1`, and intermediate dimensions follow lexicographic corner combinations."""
        N = self.n_elements
        return ByDimensions(self.n_dim, lambda n, m: Sparse.selection(self._local(n, m).T, N[m]))

    @cached_property
    def boundary(self) -> ByDimensions:
        """Signed facet operators weight each facet of a `(k + 1)`-simplex by its induced boundary
        orientation: `-1` at an edge's tail and `+1` at its head, `+1` along a triangle's winding
        and `-1` against it, and `+1` on outward-oriented tetrahedron faces."""
        def signed_facets(n: int, m: int) -> Sparse:
            if m != n - 1:
                raise ValueError(f"only adjacent dimensions have a boundary, not {n} and {m}")
            return self._signed(n, m, self.orientation[m])
        return ByDimensions(self.n_dim, signed_facets)

    @cached_property
    def oriented(self) -> ByDimensions:
        """Sub-simplex incidence operators carry `+1` where the local sub-simplex agrees in
        orientation with its stored global simplex, and `-1` where opposite."""
        return ByDimensions(self.n_dim, lambda n, m: self._signed(n, m, simplex_parity(self.elements[n][:, self.corners[n, m].T])))

    @cached_property
    def d(self) -> tuple[Sparse, ...]:
        """The discrete exterior derivative (coboundary) evaluates a `k`-simplex quantity around the
        oriented boundary of each `(k + 1)`-simplex: differences from tail to head along edges at
        `d[0]`, circulation around triangles at `d[1]`, and net outward flux from tetrahedra at
        `d[2]`. Because every shared ridge cancels around the boundary of a boundary,
        `d[k + 1] * d[k] == 0`."""
        return tuple(self.boundary[k + 1, k].sum(axis=0) for k in range(self.n_dim))

    @cached_property
    def corners(self) -> ByDimensions:
        """Each column gives the local corner numbers of one `m`-simplex inside an `n`-simplex,
        ordered to match `incidence[n, m]` (`tails, heads = topology.corners[n, 1]`)."""
        return ByDimensions(self.n_dim, local_corners)

    @cached_property
    def incidence_indices(self) -> ByDimensions:
        """Row `s` lists the global indices in `elements[m]` of the `m`-simplices belonging to
        `n`-simplex `s`, in the order of `incidence[n, m]`."""
        return ByDimensions(self.n_dim, self._local)

    @cached_property
    def complements(self) -> ByDimensions:
        """Each column lists the local corners of an `n`-simplex that lie off the corresponding
        `m`-simplex in `incidence[n, m]`."""
        return ByDimensions(self.n_dim, local_complements)

    @cached_property
    def on(self) -> ByDimensions:
        """The maps in `on[n, m]` read the vertices of each sub-simplex in increasing global index
        order, matching its stored orientation in `elements[m]`."""
        N = self.n_elements

        def stored(n: int, m: int) -> Sparse:
            corners = np.sort(self.elements[n][:, local_corners(n, m)], axis=1)
            return Sparse.selection(corners.transpose(1, 2, 0), N[0])
        return ByDimensions(self.n_dim, stored)

    @cached_property
    def off(self) -> ByDimensions:
        """The maps in `off[n, m]` read the vertices of each `n`-simplex that lie off each of its
        `m`-simplices (at `m == 0`, the vertices of the facet opposite each corner)."""
        N = self.n_elements
        return ByDimensions(self.n_dim, lambda n, m: Sparse.selection(
            self.elements[n][:, local_complements(n, m)].transpose(1, 2, 0), N[0]))

    def _local(self, simplex_dim: int, sub_dim: int) -> np.ndarray:
        if sub_dim == 0:
            return self.elements[simplex_dim]
        if sub_dim == simplex_dim - 1:
            return self.boundary_indices[sub_dim]
        return lookup_simplices(
            self.elements[sub_dim],
            np.sort(self.elements[simplex_dim][:, self.corners[simplex_dim, sub_dim].T], axis=-1),
        )

    def _signed(self, simplex_dim: int, sub_dim: int, signs: np.ndarray) -> Sparse:
        N = self.n_elements
        local = self._local(simplex_dim, sub_dim)
        entries = np.eye(local.shape[-1])[:, None, :] * signs[None, :, :]
        return Sparse.from_indices(entries, np.arange(N[simplex_dim])[:, None], local, (N[simplex_dim], N[sub_dim]))

    @cached_property
    def exterior(self) -> tuple[np.ndarray, ...]:
        """Border facets are those belonging to a single top simplex, and lower border simplices are
        all simplices in their closure."""
        N = self.n_elements
        top = self.n_dim - 1
        on_border = [np.flatnonzero(np.bincount(self.boundary_indices[top].ravel(), minlength=N[-2]) == 1)]
        for k in range(top, 0, -1):
            on_border.insert(0, np.unique(self.boundary_indices[k - 1][on_border[0]]))
        return (*on_border, np.zeros(0, dtype=int))

    @cached_property
    def interior(self) -> tuple[np.ndarray, ...]:
        """Interior simplices at each dimension are those strictly off the topological border."""
        return tuple(np.setdiff1d(np.arange(count), on_border) for count, on_border in zip(self.n_elements, self.exterior))

    def subset(self, selection: np.ndarray, dimension: int) -> Topology:
        """Selected simplices at `dimension` (by index or boolean mask) form a sub-complex that
        preserves the parent vertex numbering."""
        N = self.n_elements
        return Topology(N[0], self.elements[dimension][selection])

    def cone(self) -> Topology:
        """Joining every top simplex to a single added apex vertex forms an `(n_dim + 1)`-dimensional
        cone over this complex."""
        N = self.n_elements
        apex = np.full((N[-1], 1), N[0])                                             # [N[-1], 1]
        return Topology(N[0] + 1, np.concatenate([apex, self.elements[-1]], axis=-1))

    def product(self, path: Topology) -> Topology:
        """Extruding along a one-dimensional path subdivides each prism by the sorted-vertex
        staircase so adjacent prisms share triangulated side facets and coherent orientations
        compose."""
        N = self.n_elements
        V_path, E_path = path.n_elements
        n_corners = self.n_dim + 1
        simplices = np.sort(self.elements[-1], axis=-1)                              # [N[-1], n_dim + 1]
        # Staircase step j keeps corners 0..j on the tail copy and advances corners j..n_dim to the head copy.
        places, steps = np.arange(n_corners + 1), np.arange(n_corners)[:, None]
        on_head = places > steps                                                     # [n_dim + 1, n_dim + 2]
        corners = places - on_head                                                   # [n_dim + 1, n_dim + 2]
        tails, heads = path.elements[1].T                                            # [E_path], [E_path]
        copies = np.where(on_head, heads[:, None, None], tails[:, None, None])       # [E_path, n_dim + 1, n_dim + 2]
        extruded = simplices[:, None, corners] + N[0] * copies[None]                 # [N[-1], E_path, n_dim + 1, n_dim + 2]
        # Alternate signs (-1)**j across staircase steps times each base simplex's parity relative to sorted order.
        signs = simplex_parity(self.elements[-1])[:, None, None] * (-1.0) ** np.arange(n_corners)   # [N[-1], 1, n_dim + 1]
        wound = np.broadcast_to(signs, extruded.shape[:-1]).reshape(-1) > 0          # [N[-1] * E_path * (n_dim + 1)]
        extruded = extruded.reshape(-1, n_corners + 1)                               # [N[-1] * E_path * (n_dim + 1), n_dim + 2]
        return Topology(N[0] * V_path, np.where(wound[:, None], extruded, reverse_orientation(extruded)))

    @cached_property
    def border(self) -> Topology:
        """Facets incident to a single top simplex form an outward-oriented `(n_dim - 1)`-dimensional
        topology over the same vertex set."""
        N = self.n_elements
        top = self.n_dim - 1
        facets, signs = self.boundary_indices[top], self.orientation[top]            # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
        lone = np.bincount(facets.ravel(), minlength=N[-2])[facets] == 1             # [N[-1], n_dim + 1]
        # Stored facet vertices are sorted; flip any border facet whose parent simplex induces -1 on it.
        sides = self.elements[top][facets[lone]]
        return Topology(N[0], np.where((signs[lone] > 0)[:, None], sides, reverse_orientation(sides)))

    @cached_property
    def is_closed(self) -> bool:
        """A complex is closed when its topological border is empty (such as a loop, sphere, or torus)."""
        return self.border.n_elements[-1] == 0

    @cached_property
    def is_oriented(self) -> bool:
        """A complex is coherently oriented when every shared interior facet receives opposite
        boundary orientations from its two incident top simplices."""
        N = self.n_elements
        neighbours, entries = self.neighbours                                        # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
        signs = self.orientation[self.n_dim - 1]                                     # [N[-1], n_dim + 1]
        shared = neighbours != np.arange(N[-1])[:, None]                             # [N[-1], n_dim + 1]
        return bool(np.all(signs[shared] != signs[neighbours, entries][shared]))

    @cached_property
    def is_manifold(self) -> bool:
        """Every facet bounds at most two top simplices and every vertex has a single
        facet-connected star of top simplices."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        N = self.n_elements
        top = self.n_dim - 1
        simplices, width = self.elements[-1], self.n_dim + 1                         # [N[-1], n_dim + 1]
        thin = np.bincount(self.boundary_indices[top].ravel(), minlength=N[-2]).max(initial=0) <= 2
        # Treat each corner of each top simplex as a graph node and link matching corners across shared facets;
        # the complex is manifold at vertices iff every vertex's corners lie in at most one connected component.
        neighbours, entries = self.neighbours                                        # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
        owners = np.broadcast_to(np.arange(N[-1])[:, None], neighbours.shape)        # [N[-1], n_dim + 1]
        shared = neighbours != owners                                                # [N[-1], n_dim + 1]
        on_facet = local_corners(self.n_dim, top).T[np.nonzero(shared)[1]]
        owner, other = owners[shared], neighbours[shared]
        at = simplices[owner[:, None], on_facet]
        matching = (simplices[other][:, None, :] == at[:, :, None]).argmax(axis=-1)
        links = ((owner[:, None] * width + on_facet).ravel(), (other[:, None] * width + matching).ravel())
        graph = coo_matrix((np.ones(len(links[0])), links), shape=(simplices.size,) * 2)
        _, pieces = connected_components(graph, directed=False)                      # [N[-1] * (n_dim + 1)]
        stars = np.unique(np.stack([simplices.ravel(), pieces], axis=-1), axis=0)[:, 0]
        return bool(thin and np.all(np.bincount(stars) <= 1))

    @cached_property
    def n_components(self) -> int:
        """Connected components are counted across shared facets between top simplices, so simplices
        meeting only at a vertex or ridge lie in separate components."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        N = self.n_elements
        neighbours = self.neighbours[0]                                              # [N[-1], n_dim + 1]
        owners = np.broadcast_to(np.arange(N[-1])[:, None], neighbours.shape)        # [N[-1], n_dim + 1]
        graph = coo_matrix((np.ones(neighbours.size), (owners.ravel(), neighbours.ravel())), shape=(N[-1],) * 2)
        return int(connected_components(graph, directed=False)[0])

    @cached_property
    def euler_characteristic(self) -> int:
        """The alternating sum of active simplex counts (`V - E + F - ...`, ignoring isolated
        vertices) is a topological invariant equal to `2` on a sphere, `0` on a loop or torus, and
        `1` on a disc or ball."""
        counts = np.array([len(np.unique(self.elements[-1])), *self.n_elements[1:]])
        return int(np.sum((-1) ** np.arange(self.n_dim + 1) * counts))

    @classmethod
    def rings(cls, n_rings: int, n_sectors: int) -> Topology:
        """Vertex `0` at the centre surrounded by `n_rings` concentric rings triangulates a
        topological disc."""
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
        """A doubly periodic `n_major`-by-`n_minor` grid triangulates into a closed, coherently
        oriented torus."""
        u, v = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
        v00 = u * n_minor + v
        v10 = ((u + 1) % n_major) * n_minor + v
        v01 = u * n_minor + (v + 1) % n_minor
        v11 = ((u + 1) % n_major) * n_minor + (v + 1) % n_minor
        faces = np.concatenate([np.stack([v00, v10, v11], axis=-1).reshape(-1, 3),
                                np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)])
        return cls(n_major * n_minor, faces)

    @classmethod
    def chains(cls, lengths: list[int]) -> Topology:
        """Consecutive runs of vertices form disjoint open polygonal chains."""
        lengths = np.asarray(lengths, dtype=int)
        V = int(lengths.sum())
        # Connect each vertex to the next except across run boundaries.
        starts = np.zeros(V + 1, dtype=bool)
        starts[np.cumsum(lengths) - lengths] = True
        tails = np.flatnonzero(~starts[1:V])
        return cls(V, np.stack([tails, tails + 1], axis=-1))

    @classmethod
    def loops(cls, lengths: list[int]) -> Topology:
        """Consecutive runs of vertices form disjoint closed polygonal loops."""
        return cls._loops(tuple(lengths))

    @classmethod
    @cache
    def _loops(cls, lengths: tuple[int, ...]) -> Topology:
        lengths = np.asarray(lengths, dtype=int)
        V = int(lengths.sum())
        tails = np.arange(V)                                                         # [V]
        firsts, sizes = np.repeat(np.cumsum(lengths) - lengths, lengths), np.repeat(lengths, lengths)   # [V], [V]
        return cls(V, np.stack([tails, firsts + (tails - firsts + 1) % sizes], axis=-1))

    @classmethod
    def grid(cls, n_rows: int, n_columns: int) -> Topology:
        """A rectangular `n_rows`-by-`n_columns` vertex grid triangulates into a coherently oriented
        planar sheet."""
        vertices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)          # [n_rows, n_columns]
        quads = np.stack((vertices[:-1, :-1], vertices[:-1, 1:], vertices[1:, 1:], vertices[1:, :-1]), axis=-1)
        return cls(n_rows * n_columns, quads[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))

    def reading(self, indices: np.ndarray, dimension: int) -> Sparse:
        """Selecting a field on simplices of `dimension` at `indices` pairs with `.adjugate()`,
        which scatters values at those sites back onto the full complex."""
        N = self.n_elements
        return Sparse.selection(indices, N[dimension])

    def loop(self, is_pinned: np.ndarray) -> Sparse:
        """One Loop subdivision step refines coarse vertices followed by coarse edge midpoints,
        holding vertices marked by `is_pinned` fixed and keeping border stencils strictly on the
        border."""
        V, E, F = self.n_elements
        edges, faces, face_edges = self.elements[1], self.elements[2], self.boundary_indices[1]   # [E, 2], [F, 3], [F, 3]
        shape = (V + E, V)
        everyone = np.arange(V)                                                      # [V]
        rim = np.isin(np.arange(E), self.exterior[1])                                # [E]
        on_rim = np.isin(everyone, edges[rim])                                       # [V]
        valence = np.bincount(edges.reshape(-1), minlength=V)                        # [V]
        beta = (5 / 8 - (3 / 8 + np.cos(2 * np.pi / valence) / 4) ** 2) / valence    # [V]
        own = np.where(is_pinned, 1.0, np.where(on_rim, 3 / 4, 1 - valence * beta))  # [V]
        keep = Sparse.from_indices(own, everyone, everyone, shape)
        # Off-diagonal coarse-vertex neighbour weights along each edge.
        along = np.where(is_pinned[edges], 0.0,
                         np.where(on_rim[edges], np.where(rim[:, None], 1 / 8, 0.0), beta[edges]))   # [E, 2]
        neighbours = Sparse.from_indices(along, edges, edges[:, ::-1], shape)
        # Contributions from each triangle's three corners to its three midpoint edge vertices.
        facing_rim = rim[face_edges][:, :, None]                                     # [F, 3, 1]
        shares = np.where(np.eye(3, dtype=bool), np.where(facing_rim, 0.0, 1 / 8), np.where(facing_rim, 1 / 2, 3 / 16))
        split = Sparse.from_indices(shares, V + face_edges[:, :, None], faces[:, None, :], shape)
        return keep + neighbours + split

    @cached_property
    def split_faces(self) -> np.ndarray:
        """Edge-midpoint refinement splits every triangle into four coherently wound sub-triangles
        (the central midpoint triangle followed by the three corner triangles)."""
        V, E, F = self.n_elements
        middle = V + self.boundary_indices[1]                                        # [F, 3]
        corners = np.stack([self.elements[2], np.roll(middle, 1, axis=1), np.roll(middle, -1, axis=1)], axis=-1)   # [F, 3, 3]
        return np.concatenate([middle, corners.reshape(-1, 3)])                      # [4 * F, 3]

    def loop_subdivided(self, is_pinned: np.ndarray, n_steps: int) -> SubdividedTopology:
        """Repeated Loop subdivision produces a finer surface topology carrying the composed
        coarse-to-fine vertex refinement operator."""
        topology = SubdividedTopology(self, is_pinned)
        for _ in range(n_steps - 1):
            topology = SubdividedTopology(topology, topology.pinned)
        return topology

    @cached_property
    def dual(self) -> DualTopology:
        """The combinatorial dual complex places a dual vertex on each top simplex, a dual edge
        across each facet, a dual face around each codimension-two ridge, and a dual cell around
        each primal vertex."""
        return DualTopology(self)

    @cached_property
    def facet_opposites(self) -> np.ndarray:
        """Interior facets carry the opposite corner vertex from the `+1`-oriented incident simplex
        first and the `-1`-oriented simplex second."""
        # Sum of 0..n_dim minus the n_dim corners on a facet leaves the single corner opposite that facet.
        corners = local_corners(self.n_dim, self.n_dim - 1)                          # [n_dim, n_dim + 1]
        facing = self.n_dim * (self.n_dim + 1) // 2 - corners.sum(axis=0)            # [n_dim + 1]
        return self._facet_sides(self.elements[-1][:, facing])

    @cached_property
    def neighbours(self) -> tuple[np.ndarray, np.ndarray]:
        """Across local facet `i` of top simplex `s`, the two tables give the adjacent top simplex
        and its local corner opposite the shared facet, pointing back to `(s, i)` on the border."""
        N = self.n_elements
        width = self.n_dim + 1
        facets = self.boundary_indices[self.n_dim - 1].reshape(-1)                   # [N[-1] * (n_dim + 1)]
        order = np.argsort(facets, kind="stable")
        shared = facets[order][1:] == facets[order][:-1]
        first, second = order[:-1][shared], order[1:][shared]
        across = np.arange(len(facets))
        across[first], across[second] = second, first
        return (across // width).reshape(-1, width), (across % width).reshape(-1, width)

    @cached_property
    def vertex_simplices(self) -> np.ndarray:
        """Each vertex maps to its lowest-numbered incident top simplex (or `n_elements[-1]` when
        isolated)."""
        N = self.n_elements
        simplices = self.elements[-1]                                                # [N[-1], n_dim + 1]
        first = np.full(N[0], N[-1])                                                 # [N[0]]
        np.minimum.at(first, simplices.reshape(-1), np.repeat(np.arange(N[-1]), simplices.shape[-1]))
        return first

    @cached_property
    def spanning_tree(self) -> tuple[np.ndarray, np.ndarray]:
        """A breadth-first spanning tree rooted at vertex `0` pairs each remaining vertex with the
        edge from its parent, giving a loop-free path for integrating edge differences into vertex
        potentials."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import breadth_first_order
        N = self.n_elements
        edges = self.elements[1]                                                     # [N[1], 2]
        graph = coo_matrix((np.ones(N[1]), (edges[:, 0], edges[:, 1])), shape=(N[0],) * 2)
        order, parents = breadth_first_order(graph.tocsr(), 0, directed=False)
        children = order[1:]
        low, high = np.minimum(children, parents[children]), np.maximum(children, parents[children])
        return children, np.searchsorted(edges[:, 0] * N[0] + edges[:, 1], low * N[0] + high)

    def circulation(self, loops: list[np.ndarray]) -> Sparse:
        """Edge values sum around every triangle in its winding order, followed by each closed
        vertex sequence in `loops`."""
        V, E, F = self.n_elements
        n_loops = len(loops)
        lengths = np.array(list(map(len, loops)), dtype=int)                         # [n_loops]
        tails = np.concatenate(loops + [np.zeros(0, dtype=int)])
        # Advance each loop vertex cyclically to the next vertex in its own loop.
        firsts = np.repeat(np.cumsum(lengths) - lengths, lengths)
        heads = tails[firsts + (np.arange(len(tails)) - firsts + 1) % np.repeat(lengths, lengths)]
        edges = self.elements[1]                                                     # [E, 2]
        keys = edges[:, 0] * V + edges[:, 1]                                         # [E]
        low, high = np.minimum(tails, heads), np.maximum(tails, heads)
        rows = np.concatenate([np.repeat(np.arange(F), 3), F + np.repeat(np.arange(n_loops), lengths)])
        columns = np.concatenate([self.boundary_indices[1].reshape(-1), np.searchsorted(keys, low * V + high)])
        signs = np.concatenate([self.orientation[1].reshape(-1), np.where(tails < heads, 1.0, -1.0)])
        return Sparse.from_indices(signs, rows, columns, (F + n_loops, E))

    def _facet_sides(self, per_facet: np.ndarray) -> np.ndarray:
        N = self.n_elements
        top = self.n_dim - 1
        facets, along = self.boundary_indices[top], self.orientation[top] > 0        # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
        first, second = np.empty((2, N[-2]), dtype=per_facet.dtype)
        first[facets[along]], second[facets[~along]] = per_facet[along], per_facet[~along]
        return np.stack([first, second], axis=-1)[self.interior[top]]


class DualTopology:
    """Dual grade `k` corresponds to primal codimension `k`, so dual coboundaries are the adjugates
    of the complementary primal coboundaries (`dual.d[k] == primal.d[n_dim - 1 - k].adjugate()`).
    Dual elements above edges are general polytopes rather than simplices, so `DualTopology` carries
    operators without vertex tables."""

    def __init__(self, primal: Topology) -> None:
        self.primal, self.n_dim = primal, primal.n_dim
        self.n_elements = primal.n_elements[::-1]

    @cached_property
    def d(self) -> tuple[Sparse, ...]:
        """The dual discrete exterior derivative sums dual `k`-quantities around the boundary of
        each dual `(k + 1)`-element (`dual.d[k] == primal.d[n_dim - 1 - k].adjugate()`), so
        `dual.d[0]` takes differences across dual edges and `dual.d[-1]` accumulates net flux out of
        each primal vertex's dual cell."""
        return tuple(derivative.adjugate() for derivative in self.primal.d[::-1])

    @cached_property
    def incidence(self) -> ByDimensions:
        """At `[1, 0]`, each dual edge selects its tail and head dual vertices (leaving one end
        empty on border facets); higher dual polytopes have variable face counts and use `d`
        instead."""
        N = self.primal.n_elements

        def ends(n: int, m: int) -> Sparse:
            if (n, m) != (1, 0):
                raise ValueError(f"the dual holds the ends of its edges alone, not its {n}-cells' {m}-cells")
            facets, signs = self.primal.boundary_indices[-1], self.primal.orientation[-1]   # [N[-1], n_dim + 1], [N[-1], n_dim + 1]
            owners = np.broadcast_to(np.arange(N[-1])[:, None], facets.shape)               # [N[-1], n_dim + 1]
            # A primal top simplex is the tail of dual edges where its facet sign is -1 and the head where +1.
            entries = np.stack([signs < 0, signs > 0]).astype(float).reshape(2, -1)         # [2, N[-1] * (n_dim + 1)]
            return Sparse.from_indices(entries, facets.ravel(), owners.ravel(), (self.n_elements[1], self.n_elements[0]))
        return ByDimensions(self.n_dim, ends)

    def reading(self, indices: np.ndarray, dimension: int) -> Sparse:
        """Selecting a field on dual elements of `dimension` at `indices` pairs with `.adjugate()`
        to scatter values back onto the dual complex."""
        N = self.n_elements
        return Sparse.selection(indices, N[dimension])


class SubdividedTopology(Topology):
    """A Loop-refined surface topology retains the coarse-to-fine operator chain so multi-step
    subdivision maps compose."""

    def __init__(self, coarser: Topology, is_pinned: np.ndarray) -> None:
        V, E, F = coarser.n_elements
        self.coarser = coarser
        self.from_coarser = coarser.loop(is_pinned)                                  # [V + E, V]
        self.pinned = np.concatenate([is_pinned, np.zeros(E, dtype=bool)])           # [V + E]
        super().__init__(V + E, coarser.split_faces)

    @property
    def steps(self) -> tuple[Sparse, ...]:
        """Each entry refines vertex fields across one Loop subdivision level from the coarsest
        topology to `self`."""
        return self.coarser.steps + (self.from_coarser,)

    @cached_property
    def subdivision(self) -> Sparse:
        """Composing `.steps` maps vertex fields on the coarsest topology directly to this refined
        surface."""
        return reduce(mul, reversed(self.steps))
