# Simplicial topologies

This document introduces numga's topology module, which tracks how vertices connect into edges, triangles, tetrahedra and higher-dimensional simplices.

Both the concept and its implementation are independent of geometric algebra. A topology stores only which vertices belong to each simplex, with no coordinates or metric, and its operators run on plain NumPy arrays on their own. It is included in numga because placing geometric objects on a topology lets that connectivity drive calculus on fields ([Fields and field maps](fields.md), [Discrete differential geometric algebra](../articles/ddga.md)) and GPU rendering ([WebGPU shaders and rendering](rendering.md)).

```python
import numpy as np

from numga.topology import Topology

# A 3-by-3 grid of vertices triangulated into eight coherently oriented triangles.
sheet = Topology.grid(3, 3)
V, E, F = sheet.n_elements
print((V, E, F), sheet.elements[2][0].tolist())
```

```text
(9, 16, 8) [0, 1, 4]
```

`sheet.elements[k]` holds the vertex table of each `k`-simplex (`k = 0, ..., sheet.n_dim`). Top simplices (`sheet.elements[-1]`) retain the caller's vertex order to fix their orientation, while every lower simplex is deduplicated in increasing vertex order with induced `+1` and `-1` facet signs. Negative dimension indices count down from `n_dim`, so `[-1, 0]` addresses top simplices against vertices in any dimension.

## Sub-simplex incidences

On each simplex, plain-number `Sparse` maps select either its sub-simplices (`incidence[n, m]`) or the vertices that lie opposite each sub-simplex (`off[n, m]`). Calling a `Sparse` map on a NumPy array gathers values along its columns and sums into its rows, while multiplying two `Sparse` maps (`S * T`) composes them. Above dimension one, facet `i` omits corner `i`, so corner values gathered by `incidence[-1, 0]` align column for column with the opposite-facet vertices gathered by `off[-1, 0]`:

```python
labels = np.arange(float(V))
corners = sheet.incidence[-1, 0](labels)                                 # [3, F]
opposite = sheet.off[-1, 0](labels)                                      # [2, 3, F]
print(corners[:, 0].astype(int).tolist())
print(opposite[:, :, 0].astype(int).tolist())
```

```text
[0, 1, 4]
[[1, 0, 0], [4, 4, 1]]
```

On triangle `0` (`[0, 1, 4]`), corner `0` stands opposite edge `[1, 4]`, corner `1` opposite `[0, 4]`, and corner `4` opposite `[0, 1]`. Two companion maps track global simplex storage. `on[n, m]` reads each sub-simplex's vertices in increasing global index order, and `oriented[n, m]` weights each incidence by `+1` or `-1` according to whether the local sub-simplex agrees with its stored orientation in `elements[m]`.

## Boundaries, exterior derivatives and subsets

Each `(k + 1)`-simplex induces `+1` and `-1` orientations on its facets (`-1` at an edge's tail and `+1` at its head, `+1` along a triangle's winding and `-1` against it, and `+1` on outward tetrahedron faces). Those signed facet incidences live on `boundary[k + 1, k]`, and summing across the local facets gives the discrete exterior derivative `d[k] = boundary[k + 1, k].sum(axis=0)`, which satisfies `d[k + 1] * d[k] == 0` because every shared ridge cancels around the boundary of a boundary:

```python
d0, d1 = sheet.d
edge_jumps = d0(labels)                                                  # [E]
print(np.allclose((d1 * d0).cells, 0), float(np.abs(d1(edge_jumps)).max()))
```

```text
True 0.0
```

Simplices on the topological border and those strictly inside are listed in `exterior[k]` and `interior[k]`. Passing an index array to `reading(indices, k)` builds a `Sparse` restriction map to that subset, and its `.adjugate()` scatters subset values back onto the full complex:

```python
free = sheet.reading(sheet.interior[0], 0)                               # [V_free <- V]
print(free(labels).astype(int).tolist())
print(free.adjugate()(np.array([99.0])).astype(int).tolist())
```

```text
[4]
[0, 0, 0, 0, 99, 0, 0, 0, 0]
```

## The combinatorial dual

Every simplicial complex has a combinatorial dual (`sheet.dual`) that reverses simplex grading by codimension. Each top simplex becomes a dual vertex, each facet becomes a dual edge (`dual.incidence[1, 0]`), each codimension-two ridge becomes a dual face, and each primal vertex becomes a dual cell. Because dual elements above edges are general polytopes with variable face counts rather than simplices, `DualTopology` carries coboundary operators directly as the adjugates of the complementary primal coboundaries (`dual.d[k] == sheet.d[n_dim - 1 - k].adjugate()`):

```python
dual = sheet.dual
print(dual.n_elements)
# `dual.d[0]` takes differences across shared edges; `dual.d[1]` sums edge fluxes out of each vertex's dual cell.
print(float(dual.d[1](np.ones(E)).sum()))
```

```text
(8, 16, 9)
0.0
```

Every edge enters its tail's dual cell with `-1` and its head's dual cell with `+1`, so the net flux summed across all dual cells vanishes identically.

## Constructions, queries and subdivision

Stock constructors build standard combinatorics without fixing coordinates: `Topology.grid(rows, cols)`, `Topology.rings(n_rings, n_sectors)` for a disc around vertex `0`, `Topology.torus(n_major, n_minor)`, and `Topology.chains(lengths)` and `Topology.loops(lengths)` for open and closed polygonal curves. Existing complexes combine and refine through topological operations:

| Operation | Result |
| --- | --- |
| `topology.border` | Outward-oriented `(n_dim - 1)`-dimensional `Topology` of facets incident to one top simplex |
| `topology.subset(selection, k)` | Sub-complex of chosen `k`-simplices keeping the parent vertex numbering |
| `topology.cone()` | `(n_dim + 1)`-dimensional cone joining every top simplex to one added apex vertex |
| `topology.product(path)` | Staircase prism extrusion along a 1D `path` topology with matching side facets |
| `orient_coherently(simplices)` | Top-simplex array with connected orientable components aligned across shared facets |
| `topology.loop_subdivided(pinned, n)` | Refined triangulated surface carrying the composed `Sparse` vertex map `.subdivision` |

```python
torus = Topology.torus(4, 4)
disc = Topology.rings(2, 6)
slab = disc.product(Topology.chains([2]))
refined = disc.loop_subdivided(np.zeros(disc.n_elements[0], dtype=bool), n_steps=2)

print(torus.is_closed, torus.is_oriented, torus.is_manifold, torus.euler_characteristic)
print(disc.border.n_elements[-1], slab.n_dim, np.allclose((slab.d[2] * slab.d[1]).cells, 0))
print(refined.n_elements, refined.subdivision.shape)
```

```text
True True True 0
6 3 True
(157, 444, 288) (157, 13)
```

## Fields on a topology

When a `Sparse` map meets an `Extensor` field ([Fields and field maps](fields.md)), its plain-number weights promote to extrinsic scalars of that field's algebra and act through whichever product an expression writes (`*`, `^`, `|`, `&`). Placing projective points on `sheet` turns `incidence[-1, 0]` and `off[-1, 0]` directly into each triangle's simplex gradients and assembled gradient operator:

```python
from numga import NumpyContext
from numga.algebras import PGA2D
from numga.sparse import spdiag

mv = NumpyContext(PGA2D).multivector
Scalar, Plane = PGA2D.gatype.scalar(), PGA2D.gatype.vector()

rows, cols = np.divmod(np.arange(V), 3)
points = mv.point(np.column_stack([cols, rows]).astype(float)).field()   # Point-[V]
triangle_corners = sheet.incidence[-1, 0] * points                       # [3] Point-[F]
tails, heads = sheet.off[-1, 0] * points                                 # [3] Point-[F], [3] Point-[F]

sides = tails & heads                                                    # [3] Antibivector-[F]
gradients = sides / (sides & triangle_corners)                           # [3] Plane+[F]
gradient = (spdiag(gradients * Scalar) * sheet.incidence[-1, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[F <- V]

potential = (2 * mv.x + 3 * mv.y + mv.w) & points                        # Scalar+[V]
print(gradient(potential).batch()[0].formula())
```

```text
out[x] = 2
out[y] = 3
out[w] = 1
```

Pulling a per-simplex material map back through `gradient.adjugate()` (`gradient.adjugate()(material(gradient))`) then assembles coupled operators across any dimension and signature, as developed in [Discrete differential geometric algebra](../articles/ddga.md).
