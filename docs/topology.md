# Simplicial topologies

`Topology` represents an oriented simplicial complex from integer vertex connectivity alone. In both concept and implementation it is independent of geometric algebra, importing none of `numga`'s algebraic layers and holding no coordinates, metric or signature. It lives in `numga` because its plain-number `Sparse` incidences, boundaries and exterior derivatives ([Fields and field maps](fields.md#sparse-storage)) compose directly with fields of multivectors and extensors. Joins, regressive quotients and `.adjugate()` then turn pure simplex connectivity into simplex gradients, surface connections and material pullbacks across any algebra and signature ([Discrete differential geometric algebra](../articles/ddga.md)).

```python
import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA2D
from numga.sparse import spdiag
from numga.topology import Topology

mv = NumpyContext(PGA2D).multivector
Scalar, Plane, Point = PGA2D.gatype.scalar(), PGA2D.gatype.vector(), PGA2D.gatype.antivector()

# A 3-by-3 grid of vertices triangulated into eight coherently oriented triangles.
sheet = Topology.grid(3, 3)
V, E, F = sheet.n_elements
rows, cols = np.divmod(np.arange(V), 3)
points = mv.point(np.column_stack([cols, rows]).astype(float)).field()   # Point-[V]
print((V, E, F))
```

```text
(9, 16, 8)
```

`sheet.elements[k]` holds the sorted vertex table of each `k`-simplex (`k = 0, ..., sheet.n_dim`), while `sheet.elements[-1]` retains the caller's vertex order on top simplices to fix their orientation. Every lower simplex is deduplicated in increasing vertex order with induced `+1` and `-1` facet signs. Negative dimension indices count down from `n_dim`, so `[-1, 0]` addresses top simplices against vertices in any dimension.

## Incidences and the simplex gradient

`incidence[n, m]` gathers the `m`-simplices of each `n`-simplex, and `off[n, m]` gathers the vertices of each `n`-simplex that lie off each of its `m`-simplices. Above dimension one, facet `i` omits corner `i`, so corner values from `incidence[-1, 0]` align with the opposite facets from `off[-1, 0]` along the leading local axis:

```python
corners = sheet.incidence[-1, 0] * points                                # [3] Point-[F]
tails, heads = sheet.off[-1, 0] * points                                 # [3] Point-[F], [3] Point-[F]
# Joining the vertices opposite each corner and dividing by the triangle's join gives its simplex gradient.
sides = tails & heads                                                    # [3] Antibivector-[F]
gradients = sides / (sides & corners)                                    # [3] Plane+[F]

# Assembling the three corner maps takes vertex potentials to one plane per triangle.
gradient = (spdiag(gradients * Scalar) * sheet.incidence[-1, 0]).sum(axis=0)  # (Plane+ <- Scalar+)[F <- V]
potential = (2 * mv.x + 3 * mv.y + mv.w) & points                        # Scalar+[V]
print(gradient(potential).batch()[0].formula())
```

```text
out[x] = 2
out[y] = 3
out[w] = 1
```

In a projective algebra, `gradients[i] & corners[j]` equals `1` when `i == j` and `0` on the opposite facet, so `gradient(potential)` recovers both the spatial slope (`2 * mv.x + 3 * mv.y`) and the value at the origin (`1 * mv.w`) on every triangle. `on[n, m]` reads each sub-simplex's vertices in its stored global order, while `oriented[n, m]` weights each incidence by `+1` or `-1` according to whether the local sub-simplex agrees with that stored orientation.

## Boundaries, exterior derivatives and pullbacks

`boundary[k + 1, k]` weights each facet of a `(k + 1)`-simplex by its induced boundary sign (`-1` at an edge's tail and `+1` at its head, `+1` along a triangle's winding and `-1` against it, and `+1` on outward tetrahedron faces). Summing over the local facets gives the discrete exterior derivative `d[k] = boundary[k + 1, k].sum(axis=0)`:

```python
d0, d1 = sheet.d
print(np.allclose((d1 * d0).cells, 0))
```

```text
True
```

`exterior[k]` and `interior[k]` partition `k`-simplices into those on the topological border and those strictly inside, and `reading(indices, k)` builds a `Sparse` selection map whose `.adjugate()` scatters values on the subset back to the full complex. Pulling a per-simplex material map back through `gradient.adjugate()` assembles a coupled vertex operator directly:

```python
areas = (corners[0] & corners[1] & corners[2]).norm() / 2                # Scalar+[F]
directions = stack((mv.x.dual(), mv.y.dual()))                           # [2] Point-
material = areas * (directions * (Plane & directions)).sum(axis=0)       # (Point- <- Plane+)[F]
laplacian = gradient.adjugate()(material(gradient))                      # (Antipseudoscalar- <- Scalar+)[V <- V]

# Clamp the border vertices to 2x + 3y and solve for the single interior vertex at (1, 1).
free = sheet.reading(sheet.interior[0], 0)
fixed = sheet.reading(sheet.exterior[0], 0)
rim = (2 * mv.x + 3 * mv.y) & (fixed * points)                           # Scalar+[V_rim]
rhs = -(free * laplacian(fixed.adjugate() * rim))                        # Antipseudoscalar-[V_free]
interior_value = (free * laplacian * free.adjugate()).solve(rhs)         # Scalar+[V_free]
solution = fixed.adjugate() * rim + free.adjugate() * interior_value     # Scalar+[V]
print(solution.batch()[4].formula())
```

```text
out[1] = 5
```

## The combinatorial dual

`sheet.dual` reverses the simplex grading by codimension: each top simplex becomes a dual vertex, each facet becomes a dual edge (`dual.incidence[1, 0]`), each codimension-two ridge becomes a dual face, and each primal vertex becomes a dual cell. Because dual elements above edges are general polytopes with variable face counts rather than simplices, `DualTopology` carries coboundary operators directly as the adjugates of the complementary primal coboundaries (`dual.d[k] == sheet.d[n_dim - 1 - k].adjugate()`):

```python
dual = sheet.dual
print(dual.n_elements)
# `dual.d[0]` takes differences between adjacent triangles across shared edges; `dual.d[1]` sums edge fluxes out of each vertex's dual cell.
flux = mv.scalar(np.ones((E, 1))).field()                                # Scalar+[E]
print((dual.d[1] * flux).sites.sum().formula())
```

```text
(8, 16, 9)
out[1] = 0
```

Every interior edge enters one endpoint's dual cell with `+1` and the other's with `-1`, while each border edge has both endpoints on the complex, so the total net flux summed across all dual cells vanishes identically.

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

Because `refined.subdivision` is a plain-number `Sparse` map from the 13 coarse vertices to the 157 refined vertices, `refined.subdivision * coarse_points` refines control points in `PGA3D`, `VGA3D` or any joined algebra without coordinate unpacking.
