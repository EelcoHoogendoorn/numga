# Fields and field maps

A batch axis holds independent copies: a batch of rotors is many rotors, each acting on its own. A field is one object spread over sites, the vertices of a triangulated surface, the cameras of a rig, the atoms of a lattice, with an element of the same type at each site. Its sites belong to its type, so a map can couple them.

```python
import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D

mv = NumpyContext(VGA3D).multivector
Scalar, Vector = VGA3D.gatype.scalar(), VGA3D.gatype.vector()       # Scalar+, Vector+

# Four beads on a string, the sites of one field.
positions = mv.vector([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]).field()  # Vector+[4]
print(positions)
```

```text
Extensor(extrinsic vector[4], shape=())
```

`.field()` reads the trailing batch axis as the sites of the output. The type `extrinsic vector[4]` holds a vector at each of four sites; `shape`, which counts batch axes only, is empty because there is one string. `.batch()` reads the sites back as a batch axis. Neither touches the coefficients: the kernel of `extrinsic vector[4]` has the same `[4, 3]` layout as a batch of four vectors.

## Operations act at every site

An operation on elements acts on a field site by site, and a value without sites applies identically at every site. Batch axes stay independent copies in front of the sites:

```python
turn = (mv.xy * 0.25).exp()                                         # [] Rotor+
turns = (mv.xy * np.array([0.1, 0.2, 0.3])).exp()                   # [3] Rotor+
# One turn of the whole string, three turned strings, and each bead's distance from the origin.
print(turn >> positions)
print(turns >> positions)
print(positions.norm())
```

```text
Extensor(extrinsic vector[4], shape=())
Extensor(extrinsic vector[4], shape=(3,))
Extensor(extrinsic scalar[4], shape=())
```

A batch with the same count is still a batch: a `[4] Scalar+` times `positions` produces `[4] Vector+[4]`, four copies of the string each scaled as a whole. A quantity that varies along the string is itself a field over the four sites (`Scalar+[4]`) and multiplies bead by bead.

Indexing, `len`, `.sum()` and `.mean()` address batch axes. Reductions across sites live on `.sites`: `positions.sites.mean()` computes the centre of the string and `positions.norm().sites.sum()` sums the beads' distances from the origin.

## Field maps couple sites

Any axis of a map can range over sites. A map `Vector+[4] <- Vector+[4]` holds a `Vector+ <- Vector+` block for every pair of an output site and an input site, built from a batch over pairs by `.field(0, 1)`:

```python
# Each bead tied to its neighbours and to its rest position.
weights = np.array([[2, -1, 0, 0], [-1, 3, -1, 0], [0, -1, 3, -1], [0, 0, -1, 2.0]])
stiffness = (mv.scalar(weights[..., None]) * Vector).field(0, 1)    # Vector+[4] <- Vector+[4]
downward = [[0, 0, 0], [0, 0, -1], [0, 0, -1], [0, 0, 0]]
loads = mv.vector(downward).field()                                 # Vector+[4]

forces = stiffness(loads)                                           # Vector+[4]
displacements = stiffness.solve(loads)                              # Vector+[4]
print(displacements)
print(stiffness(stiffness))
```

```text
Extensor(extrinsic vector[4], shape=())
Extensor(extrinsic vector[4] <- extrinsic vector[4], shape=())
```

Binding a field into a field axis sums over the paired input sites so each output site receives every input site's contribution. Composition sums along every path through the shared sites, and `solve` solves the coupled system across all sites and blades at once; solving against a map preserves its inputs, so `stiffness.solve(stiffness)` is the identity on `Vector+[4]`.

`stiffness.adjoint()` takes each block's adjoint and exchanges output and input sites so the scalar product summed over sites carries over, and `.adjugate()` does the same with `&` read extrinsic first. A map with sites on only one axis maps into or out of a single field: the adjoint of `Vector+[4] <- Vector+` (which spreads one vector across four sites) is `Vector+ <- Vector+[4]` (which sums them back). Independent maps at each site use a batch axis (`.batch()`), whereas `maps.on_diagonal()` places a field of maps `Vector+[4] <- Vector+` along the site diagonal as `Vector+[4] <- Vector+[4]`.

Values and maps without sites act identically at every site: `stiffness.solve(-mv.z)` computes the sag under a uniform load on every bead, `stiffness + Vector` adds `Vector` along the site diagonal (`(stiffness + Vector)(loads) == stiffness(loads) + loads`), `metric.solve(slope)` solves `Scalar+ <- Bivector+[n]` site by site against `metric`, `(turn >> Vector)(stiffness)` turns every output block, and `turn >> stiffness(turn << Vector)` transports the coupled stiffness into the turned frame.

Dense field maps support the full linear-algebra interface across all blocks at once: `lstsq` and `pinv` (leaving singular gauge directions at zero), `inverse`, `det`, `trace`, `cholesky`, `eigh` and `eig` (including against a metric field form, returning modes as a batch of fields), and `svd`. Symmetric forms over fields likewise support `eigh`, `lstsq` and `cholesky`.

In block-matrix notation a field map is a matrix of blocks, one block per pair of sites, with kernel shape `[4, 4, 3, 3]`: binding is the block matrix-vector product, composition is the block matrix product, and `solve` solves the coupled $4 \cdot 3$ linear system (`Scalar+[n] <- Scalar+[n]` being an ordinary $n \times n$ matrix).

## Derivatives with respect to a field

Differentiating with respect to a field keeps its sites inside the derivative's input axis, coupling every site to every other. The second derivative of a scalar function of `Vector+[4]` has type `Scalar+ <- (Vector+[4], Vector+[4])`, and solving it against the gradient gives the coupled Newton step:

```python
from numga.backend.jax import JaxContext, derivative

jax_mv = JaxContext(VGA3D, np.float64).multivector
springs = (jax_mv.scalar(weights[..., None]) * Vector).field(0, 1)  # Vector+[4] <- Vector+[4]
pull = jax_mv.vector(downward).field()                              # Vector+[4]


def energy(beads):
    """Spring energy less the work of the loads, summed over the beads."""
    return (beads.scalar_product(springs(beads)) / 2 - pull.scalar_product(beads)).sites.sum()


rest = jax_mv.vector(np.zeros((4, 3))).field()                      # Vector+[4]
gradient = derivative(energy)(rest)                                 # Scalar+ <- Vector+[4]
curvature = derivative(derivative(energy))(rest)                    # Scalar+ <- (Vector+[4], Vector+[4])
step = -curvature.solve(gradient)                                   # Vector+[4]
print(curvature.gatype.signature)
print((step - springs.solve(pull)).norm().batch().max() < 1e-12)
```

```text
extrinsic scalar <- extrinsic vector[4], extrinsic vector[4]
True
```

Because `energy` is quadratic, a single Newton step recovers `springs.solve(pull)`. Differentiating across a batch axis, by contrast, differentiates independent copies element by element.

## Sparse storage

Across a simplicial topology or a lattice, most site pairs do not interact directly: a bead couples to its neighbours and a vertex to its incident edges. `SparseExtensor` stores only the nonzero couplings, each linking an input site (`columns[k]`) to an output site (`rows[k]`), and acts on the same fields as a dense field map:

```python
from numga.sparse import SparseExtensor, spdiag

# Three springs along the string, each taking the difference of its two beads.
ends = SparseExtensor(np.array([-1, 1, -1, 1, -1, 1]),
                      np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2, 2, 3]), (3, 4))
stretches = ends * positions                                        # Vector+[3]
anchors = mv.scalar(np.ones((4, 1))).field()                        # Scalar+[4]
sparse_stiffness = (~ends * ends + anchors) * Vector                # (Vector+ <- Vector+)[4 <- 4]
print((sparse_stiffness.solve(loads) - displacements).norm().batch().max() < 1e-12)
```

```text
True
```

`cells` has shape `[..., couplings]`, with any leading axes indexing separate sparse maps that share one sparsity pattern (`S[..., None]`), and `shape` is `(output_sites, input_sites)`. Cells may be typed `Extensor` batches or plain NumPy number arrays (such as incidence signs on a topology); plain-number cells belong to no algebra until they meet a field, a `GAType` or a typed `SparseExtensor`, where they become scalars of that algebra. Adding a field or a site-free map/value to a square sparse map adds along the diagonal: `(Vector - S)(f) == f - S(f)` and `(~ends * ends + anchors) * f == ~ends * ends * f + anchors * f`.

A `SparseExtensor` of number or multivector cells couples sites through whichever product an expression writes: `ends * positions`, `ends ^ positions`, `ends | positions` and `ends & positions` multiply each cell with the element at its column and sum at the rows, while a field on the left (`stretches * ends`) pairs with the rows. Two sparse extensors multiply their cells along shared sites, and multiplying by a `GAType` (`(~ends * ends) * Vector`) produces map cells that apply (`sparse_stiffness(positions)`) and compose. Calling a field of maps `maps` (`Vector+[4] <- Vector+`) on a sparse map (`maps(sparse_stiffness)`) composes each site's map with that row's cells.

Operations that reverse product order transpose the couplings from rows to columns; all other elementwise operations act cell by cell:

| Cells | Couplings transposed | Cell by cell |
| --- | --- | --- |
| Numbers and multivectors | reverse (`~`), Clifford conjugate | involute, grade selection, scaling |
| Maps | `adjoint`, `adjugate` | reverse and Clifford conjugate of outputs, involute, grade selection, scaling |

Because `~(a * b) == ~b * ~a`, reversing a product swaps the coupled index (`~(ends * positions) == ~positions * ~ends`), so `~ends * ends` builds the string's graph Laplacian and `~d0 * spdiag(edge_weights) * d0` builds a weighted Laplacian from edge differences `d0` and per-edge weights. On map cells, `~` reverses each cell's output while keeping its row and column, whereas `.adjoint()` and `.adjugate()` reverse composition and transpose the couplings.

`spdiag(field)` (alias for `SparseExtensor.from_diagonal(field)`) places each site's value or map on its own diagonal cell for products, solves or inverses (`spdiag(maps).solve(f)`), `.diagonal()` extracts the diagonal back as a field, `SparseExtensor.selection(context, index, size)` builds a sparse gather from an index array, and `.to_dense()` materializes the dense `Out[rows] <- In[cols]` extensor. Sparse `solve`, `lstsq` and `eigh` take map cells and multivector fields and dispatch through SciPy on the NumPy backend, with `eigh` returning modes as a batch of fields.

In block-matrix notation a sparse extensor of multivector cells is a sparse matrix over the geometric algebra, and its reverse is the conjugate transpose with reversion as the conjugation.
