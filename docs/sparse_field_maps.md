# Sparse maps between fields

A field is a collection of extensor elements indexed by the last batch axis. Each element has
the same type and arity: a field can contain multivectors, unary maps, or multilinear maps.
Leading batch axes hold independent fields. The field axis needs no separate type.

```python
import numpy as np

from numga import Algebra, NumpyContext
from numga.sparse import SparseExtensor

ga = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()

input_sites = 2
output_sites = 2
rows = np.array([0, 0, 1])
columns = np.array([0, 1, 1])
weights = mv.scalar([[1], [-1], [1]])
directions = mv.vector([[1, 0], [0, 1]])                       # [input_sites] Vector
probe = mv.x + 2 * mv.y

# Output site zero takes the difference; output site one takes the second input.
couplings = SparseExtensor(weights, rows, columns, (output_sites, input_sites))
readouts = directions | Vector                               # [input_sites] Scalar <- Vector

# The field contains unary maps. Mixing sites preserves their open Vector inputs.
mixed_readouts = couplings * readouts                        # [output_sites] Scalar <- Vector
measured = mixed_readouts(probe)                             # [output_sites] Scalar
```

## Cells and couplings

A `SparseExtensor` holds cells. Each cell couples one input site, named by its column, to one
output site, named by its row. The sparse extensor has no action of its own: an operation on it is
its cells' operation, lifted to the couplings, and the contributions arriving at an output site
add. This is the block decomposition of a linear map on the whole field: the response to a sum of
site contributions is the sum of their responses.

A sparse extensor has structure, two site axes and a pattern of couplings, but no product and no
metric of its own. Its cells multiply by whichever product an expression names, and a shared site
axis is contracted by a plain sum that counts every site once. Weights such as areas or masses are
diagonal sparse extensors of their own, applied where they belong, and an adjoint pairs through its
cells' pairing, summed over the sites.

`shape` is `(output_sites, input_sites)`. The stored `cells` have a last batch axis of
couplings, with leading batch axes for independent field maps sharing the same sparse
pattern. Their leading axes broadcast against the leading axes of the input field.

Three distinct counts describe the action:

| Object | Meaning of its arity |
| --- | --- |
| A field element | Its open GA inputs; every element in the field has the same type and arity. |
| A cell | Its own: a multivector waits for a product, a unary map is applied. |
| The sparse field map | Takes one input field and returns one output field. Site counts do not add GA slots. |

Use **field element** for a value at a site and **cell** for a stored coupling. The example has
scalar cells acting on unary field elements; the result is another field of unary elements.
`couplings.gatype` describes the cells, not the complete field-to-field signature.

## Products and application

A multivector cell is like any multivector: it is not a map until an expression names a product.
`couplings * field`, `couplings ^ field`, `couplings | field` and `couplings & field` are four
field maps, each taking every cell's product with the element at its input site and summing at
the output sites. A field on the left pairs with the output sites instead: `field * couplings`
sums over the rows. A product of two sparse extensors multiplies their cells along the paths
through shared intermediate sites, and links the inner one's input sites to the outer one's
output sites.

With an open type the product leaves its slot open, so the cells become maps:

```python
field_map = couplings * Scalar                              # cells: Scalar <- Scalar
mixed_by_application = field_map(readouts)                   # [output_sites] Scalar <- Vector
measured_before_mixing = couplings * readouts(probe)         # [output_sites] Scalar
```

`mixed_by_application` is the same field as `mixed_readouts`. Binding `probe` before mixing
also gives the same field of scalars as `measured`: the site sum preserves the readout's
linearity. With multivector cells the same construction uses the product named, `couplings ^
Vector` the wedge.

A map cell is applied, `field_map(field)`, and composes, `outer(inner)`. Either way each
operation acts on the field elements' outputs and keeps their open inputs, so a field of bilinear
maps keeps both.

## Which operations run the couplings back

An operation that reverses the order of the cells' products runs every coupling the other way,
from its output site to its input site; any other acts cell by cell.

| Cells | Couplings run back | Cell by cell |
| --- | --- | --- |
| Multivectors | reverse, Clifford conjugate | involute, grade selection, scaling |
| Maps | adjoint, adjugate | reverse and Clifford conjugate of their outputs, involute, grade selection, scaling |

The reverse is defined by what it does to products, `~(a * b) == ~b * ~a`: each factor moves to
the other side. A sparse extensor that moves to the other side of a product pairs with the field
by its other index, so its reverse is every cell reversed with the couplings run back:
`~(couplings * field) == ~field * ~couplings`. The same holds for every product the reverse turns
around, the wedge among them. The cotangent Laplacian arises this way: the edge energy
`~edges * H1 * edges`, with `edges = T10 * vertices` and the vertices left open on both sides, is
`~T10 * H1 * T10`.

A map cell is applied, not multiplied, and reversing a map reverses its output:
`(~field_map)(field) == ~(field_map(field))`. Nothing changes sides, and the couplings stay. The
adjoint of a map cell does turn composition around, `(A(B)).adjoint() == B.adjoint()(A.adjoint())`,
so `field_map.adjoint()` and `field_map.adjugate()` run the couplings back. Their pairing
identities hold after summing over the field sites.

In block-matrix notation, a sparse extensor of multivector cells is a matrix over the geometric
algebra, and its reverse is the conjugate transpose with the reverse as the conjugation.

## Diagonals and solvers

`SparseExtensor.from_diagonal(field)` places each field element on its own site as a cell. For
multivector elements a product with it is pointwise; for unary elements its application is
pointwise.

Sparse solves and eigenproblems use unary map cells and multivector-valued fields on the NumPy
backend through SciPy. Fields of maps take part in application and composition; the sparse
solvers return fields of multivectors.
