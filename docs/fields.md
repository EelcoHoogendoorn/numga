# Fields and field maps

A batch axis holds independent copies: a batch of rotors is many rotors, each acting on its own. A field is one object spread over sites, the vertices of a mesh, the cameras of a rig, the atoms of a lattice, with an element of the same type at each site. Its sites belong to its type, so a map can couple them.

```python
import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D

mv = NumpyContext(VGA3D).multivector
Scalar, Vector = VGA3D.gatype.scalar(), VGA3D.gatype.vector()

# Four beads on a string, the sites of one field:
positions = mv.vector([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]).field()
print(positions)
```

```text
Extensor(vector[4], shape=())
```

`.field()` reads the last batch axis as the sites of the output. The type `vector[4]` is a vector at each of four sites; the shape, which counts batch axes only, is empty, since there is one string. `.batch()` reads the sites as a batch axis again. Neither touches the coefficients: the kernel of `vector[4]` is laid out as that of a batch of four vectors, `[4, 3]`.

## Operations act at every site

An operation on elements acts on a field site by site, and a value without sites is the same at every site. Batch axes stay independent copies in front of the sites:

```python
turn = (mv.xy * 0.25).exp()
turns = (mv.xy * np.array([0.1, 0.2, 0.3])).exp()            # [3] Rotor
print(turn >> positions)                                    # one turn of the whole string
print(turns >> positions)                                   # three turned strings
print(positions.norm())                                     # each bead's distance from the origin
```

```text
Extensor(vector[4], shape=())
Extensor(vector[4], shape=(3,))
Extensor(scalar[4], shape=())
```

A batch over the same count is still a batch: a `[4] Scalar` times `positions` is four copies of the string, each scaled as a whole, `[4] vector[4]`. A quantity that varies along the string is a field itself, over the same sites, and multiplies site by site.

Indexing, `len`, `.sum()` and `.mean()` address batch axes; the sites are reached through `.batch()`, so the sum of a field over its sites is `field.batch().sum(axis=-1)`.

## Field maps couple sites

A slot of a map can range over sites too. A map from a field over four sites to a field over four sites, `vector[4] <- vector[4]`, holds a `Vector <- Vector` for every pair of an output site and an input site. A batch of maps over the pairs becomes one, `.field(0, 1)` reading its last two batch axes as the sites of the output and of the input:

```python
# Each bead tied to its neighbours and, more weakly, to its rest position:
weights = np.array([[2, -1, 0, 0], [-1, 3, -1, 0], [0, -1, 3, -1], [0, 0, -1, 2.0]])
stiffness = (mv.scalar(weights[..., None]) * Vector).field(0, 1)       # vector[4] <- vector[4]
downward = [[0, 0, 0], [0, 0, -1], [0, 0, -1], [0, 0, 0]]
loads = mv.vector(downward).field()

forces = stiffness(loads)                                   # vector[4]: summed over the input sites
displacements = stiffness.solve(loads)                      # vector[4]: stiffness(displacements) == loads
print(displacements)
print(stiffness(stiffness))                                 # composition: summed over the sites between
```

```text
Extensor(vector[4], shape=())
Extensor(vector[4] <- vector[4], shape=())
```

Binding a field into a field slot sums over the paired sites: each output site receives every input site's contribution. Composition sums along every path through the shared sites, and `solve` solves the coupled system over every pair of a site and a blade at once, which no batch of independent solves can. A solve keeps the open slots of its right-hand side, as on any map, so `stiffness.solve(stiffness)` is the identity on `vector[4]`.

`stiffness.adjoint()` takes each block's adjoint and exchanges the sites of output and input, so that the scalar product summed over the sites carries over, and `.adjugate()` does the same with the pairing `&`. A map with sites in a slot is one map into or out of a field, not a map per site: the adjoint of a `vector[4] <- vector`, which spreads one vector over four sites, is a `vector <- vector[4]`, which gathers them back, summed. Maps acting independently at each site, each with its own inverse or eigenvectors, are a batch of maps, `.batch()`; placed on the site diagonal, `maps.on_diagonal()` turns a field of maps `vector[4] <- vector` into the field map `vector[4] <- vector[4]` that acts on each site's element by that site's map and on no other. A value without sites is the same at every site in a solve as anywhere: `stiffness.solve(-mv.z)` is the sag under the same load on every bead. A map without sites is likewise the same map at every site, added as it applies: `stiffness + Vector` is the stiffness with the identity on its site diagonal, so that `(stiffness + Vector)(loads) == stiffness(loads) + loads`.

A map without sites in a slot acts at every site of a field bound into it, whatever it is bound into: `(turn >> Vector)(stiffness)` turns every block of the stiffness, and `turn >> stiffness(turn << Vector)` moves the whole stiffness into the turned frame, as for a map on one element.

The linear algebra of a field map is that of all its blocks at once: `lstsq` and `pinv` over every pair of a site and a blade, leaving a singular system's gauge at zero; `inverse`, `det` and `trace`; `cholesky`; `eigh` and `eig`, also against a metric over the same field, with their modes a batch of fields; and `svd`. A Hermitian form over a field has `eigh` and `cholesky` too.

In block-matrix notation a field map is a matrix of blocks, one block per pair of sites, its kernel `[4, 4, 3, 3]`. Binding is the block matrix-vector product, composition the block matrix product, and `solve` the solve of the matrix of $4 \cdot 3$ rows and columns. A field map of scalar cells, `scalar[n] <- scalar[n]`, is an ordinary $n \times n$ matrix.

## Derivatives with respect to a field

The derivative with respect to a field keeps its sites inside the step's slot, so every site is coupled to every other. The second derivative of a scalar function of `vector[4]` is the bilinear form `scalar <- vector[4], vector[4]`, and solving it against the gradient is the Newton step of the whole string:

```python
import jax

from numga.backend.jax import JaxContext, derivative

jax.config.update("jax_enable_x64", True)
# The same string, on JAX:
jax_mv = JaxContext(VGA3D, np.float64).multivector
springs = (jax_mv.scalar(weights[..., None]) * Vector).field(0, 1)      # vector[4] <- vector[4]
pull = jax_mv.vector(downward).field()                                  # vector[4]


def energy(beads):
    """The springs' energy less the work of the loads, summed over the beads."""
    return (beads.scalar_product(springs(beads)) / 2 - pull.scalar_product(beads)).batch().sum(axis=-1)


rest = jax_mv.vector(np.zeros((4, 3))).field()                           # vector[4]
gradient = derivative(energy)(rest)                         # scalar <- vector[4]
curvature = derivative(derivative(energy))(rest)            # scalar <- vector[4], vector[4]
step = -curvature.solve(gradient)                           # vector[4]
print(curvature.gatype.signature)
print(np.abs(np.asarray((step - springs.solve(pull)).kernel)).max() < 1e-12)
```

```text
scalar <- vector[4], vector[4]
True
```

The energy is quadratic, so one Newton step lands on the displacements the solve gave. A batch axis of the value, by contrast, holds independent copies and is differentiated element by element.

## Sparse storage

Most couplings in a mesh or a lattice are zero: a bead feels its neighbours, a vertex its edges. A `SparseExtensor` stores only the cells that are there, each coupling one input site, its column, to one output site, its row, and acts on the same fields as a dense field map.

```python
from numga.sparse import SparseExtensor

# The three springs of the string, each the difference of its two beads:
ends = SparseExtensor(mv.scalar([[-1], [1], [-1], [1], [-1], [1]]),
                      np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2, 2, 3]), (3, 4))
stretches = ends * positions                               # vector[3]
anchors = SparseExtensor.from_diagonal(mv.scalar(np.ones((4, 1))).field())
sparse_stiffness = (~ends * ends + anchors) * Vector         # [4, 4] cells Vector <- Vector
print(np.abs((sparse_stiffness.solve(loads) - displacements).kernel).max() < 1e-12)
```

```text
True
```

The stored cells have a batch axis of couplings, with leading axes for separate maps sharing one pattern; `shape` is `(output sites, input sites)`, and `gatype` is the type of the cells, by which the sparse operations dispatch. The fields it takes and returns are typed as fields.

A sparse extensor of multivector cells has no action of its own: it is a matrix over the algebra, waiting for a product. `ends * positions`, `ends ^ positions`, `ends | positions` and `ends & positions` take each cell's product with the element at its column and sum at the rows. A field on the left pairs with the rows instead: `stretches * ends` is a field over the beads. Two sparse extensors multiply their cells along the paths through shared sites. With an open type the product leaves its slot open and the cells become maps, `(~ends * ends) * Vector`; map cells are applied, `sparse_stiffness(positions)`, and compose.

An operation that reverses the order of the cells' products runs every coupling the other way, from its row to its column; any other acts cell by cell.

| Cells | Couplings run back | Cell by cell |
| --- | --- | --- |
| Multivectors | reverse, Clifford conjugate | involute, grade selection, scaling |
| Maps | adjoint, adjugate | reverse and Clifford conjugate of their outputs, involute, grade selection, scaling |

The reverse turns products around, `~(a * b) == ~b * ~a`, so a sparse extensor that moves to the other side of a product pairs with the field by its other index: `~(ends * positions) == ~positions * ~ends`. The string's Laplacian, `~ends * ends`, arises this way, and the cotangent Laplacian of a mesh likewise as `~d0 * H1 * d0`, from the edge differences `d0` and the edge weights `H1`. A map cell is applied rather than multiplied; reversing it reverses its output and keeps the couplings, while its adjoint and adjugate turn composition around and run them back, their pairing identities holding once summed over the sites.

`SparseExtensor.from_diagonal(field)` places each element of a field on its own site as a cell: weights such as areas or masses, applied where they belong. The sparse solves, least squares and eigenproblems take map cells and fields of multivectors and run through SciPy on the NumPy backend; `eigh` returns its modes as a batch of fields.

In block-matrix notation a sparse extensor of multivector cells is a sparse matrix over the geometric algebra, and its reverse the conjugate transpose with the reverse as the conjugation.
