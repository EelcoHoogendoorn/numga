"""Field maps stored sparsely: couplings between the sites of fields.

A field is an Extensor whose output ranges over sites, `Vector[vertices]`, read from a batch axis
with `.field()`; its elements may be multivectors or maps with open slots of their own. Leading
batch axes hold separate fields.

A SparseExtensor holds cells, each coupling an input site (a column) to an output site (a row).
It has no action of its own: an operation on it is its cells' operation, lifted to the couplings.
A product with a field takes each cell's product with the element at its input site, by whichever
product the expression names, `S * f`, `S ^ f`, `S | f` or `S & f`, and sums the contributions at
each output site; with an open type, `S * Even`, the slot stays open and the cells become maps. A
field on the left pairs with the output sites instead, `f * S`. A product of two sparse extensors
multiplies their cells along the paths through shared sites. A map cell is applied, `S(field)`, and
composes, `S(T)`. Field elements keep their own open slots throughout.

An operation that reverses the order of the cells' products runs the couplings the other way: the
reverse and the Clifford conjugate of multivector cells, so that `~(S * f) == ~f * ~S`, and the
adjoint and adjugate of map cells. The others act cell by cell: the involute, grade selection, and
the reverse of a map cell, which reverses its output. The stored cells have shape
`[..., couplings]`; leading axes hold separate maps sharing one pattern of couplings. Solves and
eigenproblems use map cells and multivector-valued fields, through SciPy on the NumPy backend. Every
field taken or returned is typed as one. See docs/fields.md for examples.
"""

from __future__ import annotations

import operator
from functools import singledispatch
from numbers import Number
from typing import overload

import numpy as np

from numga.extension import ExtensionMethod
from numga.extensor import Extensor, concatenate
from numga.gatype import GAType


class SparseExtensor:
    """Cells coupling `shape[1]` input sites to `shape[0]` output sites.

    `cells[..., coupling]` couples `columns[coupling]` to `rows[coupling]`; duplicate couplings add
    at construction. `gatype` describes the cells, not the whole field map, and leading cell batch
    axes hold separate maps of the same pattern.
    """

    __slots__ = ("cells", "rows", "columns", "shape")
    apply = ExtensionMethod("apply")
    reverse = ExtensionMethod("reverse")
    clifford_conjugate = ExtensionMethod("clifford_conjugate")
    adjugate = ExtensionMethod("adjugate")
    adjoint = ExtensionMethod("adjoint")
    solve = ExtensionMethod("solve")
    lstsq = ExtensionMethod("lstsq")
    eigh = ExtensionMethod("eigh")

    def __init__(self, cells: Extensor, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> None:
        if cells.ndim == 0 or not cells.shape[-1] == len(rows) == len(columns):
            raise ValueError("a sparse extensor needs one row and one column per cell")
        pairs, slots = np.unique(np.asarray(rows) * shape[1] + np.asarray(columns), return_inverse=True)
        self.cells = _scatter(cells, slots, len(pairs))
        self.rows, self.columns = np.divmod(pairs, shape[1])
        self.shape = tuple(shape)

    @classmethod
    def from_columns(cls, columns: np.ndarray, cells: Extensor, size: int) -> SparseExtensor:
        """The map from a field of `size` elements coupling output element r to the input elements
        `columns[r]` names, `[R, k]`, through `cells[..., r, :]` of the same shape."""
        rows = np.repeat(np.arange(len(columns)), columns.shape[1])
        return cls(cells.reshape(cells.shape[:-2] + (-1,)), rows, columns.reshape(-1), (len(columns), size))

    @classmethod
    def from_indices(cls, cells: Extensor, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> SparseExtensor:
        """Cells `[..., *indices]`, each coupling the row and the column given at its place in `rows` and
        `columns`, which broadcast against each other to the cells' trailing axes."""
        rows, columns = np.broadcast_arrays(rows, columns)
        return cls(cells.reshape(cells.shape[:cells.ndim - rows.ndim] + (rows.size,)), rows.reshape(-1), columns.reshape(-1), shape)

    @classmethod
    def selection(cls, context, index: np.ndarray, size: int) -> SparseExtensor:
        """The map that reads, at each of its output sites, the input site of `size` an index names,
        `index[..., output]`: applied to a field, the elements at those sites. Leading axes of the index
        give one map each, all of one pattern, each map's cells one where it reads and zero elsewhere."""
        leading, count = index.shape[:-1], index.shape[-1]
        cases = int(np.prod(leading, dtype=int))
        reads = np.broadcast_to(np.eye(cases)[:, None, :], (cases, count, cases)).reshape(leading + (count, cases))
        return cls.from_indices(context.multivector.scalar(reads[..., None]), np.arange(count)[:, None],
                                index.reshape(cases, count).T, (count, size))

    @classmethod
    def from_diagonal(cls, field: Extensor) -> SparseExtensor:
        """A field map with the element at each site as the coupling cell of that site to itself."""
        cells = _sites(field)
        index = np.arange(cells.shape[-1])
        return cls(cells, index, index, (len(index), len(index)))

    def diagonal(self) -> Extensor:
        """The cells on the diagonal as a field, each at its row; zero where a row has none."""
        on = self.rows == self.columns
        return _scatter(self.cells[..., on], self.rows[on], self.shape[0]).field()

    @property
    def gatype(self) -> GAType:
        """The coupling cells' type, by which the operations dispatch; the sites are `shape`."""
        return self.cells.gatype

    # --- products: the cells' own, summed over the shared sites ----------------------------------
    @overload
    def __mul__(self, other: SparseExtensor) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: Extensor) -> Extensor: ...
    @overload
    def __mul__(self, other: GAType) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: float) -> SparseExtensor: ...

    def __mul__(self, other):
        """Times a number, every cell; otherwise the cells' geometric product."""
        return _scaled(self, other) if isinstance(other, Number) else _right(other, self, operator.mul)

    def __rmul__(self, other):
        return _scaled(self, other) if isinstance(other, Number) else _left(other, self, operator.mul)

    def __truediv__(self, number: Number) -> SparseExtensor:
        return _scaled(self, 1 / number)

    def __xor__(self, other):
        return _right(other, self, operator.xor)

    def __rxor__(self, other):
        return _left(other, self, operator.xor)

    def __or__(self, other):
        return _right(other, self, operator.or_)

    def __ror__(self, other):
        return _left(other, self, operator.or_)

    def __and__(self, other):
        return _right(other, self, operator.and_)

    def __rand__(self, other):
        return _left(other, self, operator.and_)

    # --- operations that keep the order of products, cell by cell ---------------------------
    def __invert__(self) -> SparseExtensor:
        return self.reverse()

    def involute(self) -> SparseExtensor:
        return _cellwise(self, self.cells.involute())

    def select_grade(self, grade: int) -> SparseExtensor:
        return _cellwise(self, self.cells.select_grade(grade))

    def restrict_grade(self, grade: int) -> SparseExtensor:
        return _cellwise(self, self.cells.restrict_grade(grade))

    def select_subspace(self, subspace) -> SparseExtensor:
        return _cellwise(self, self.cells.select_subspace(subspace))

    def restrict_subspace(self, subspace) -> SparseExtensor:
        return _cellwise(self, self.cells.restrict_subspace(subspace))

    # --- unary coupling cells: application --------------------------------------------------
    @overload
    def __call__(self, operand: SparseExtensor) -> SparseExtensor: ...
    @overload
    def __call__(self, operand: Extensor) -> Extensor: ...

    def __call__(self, operand):
        return self.apply(operand)

    def __getitem__(self, index) -> SparseExtensor:
        """The maps at an index of the leading axes, which hold separate maps of one pattern:
        `S[..., None]` gives the maps an axis to broadcast against the batch of a field."""
        index = index if isinstance(index, tuple) else (index,)
        within = (slice(None),) if any(item is Ellipsis for item in index) else (Ellipsis,)
        return SparseExtensor(self.cells[index + within], self.rows, self.columns, self.shape)

    # --- sums ----------------------------------------------------------------------------------
    def __add__(self, other: SparseExtensor | Extensor | GAType) -> SparseExtensor:
        """The sum of two maps of the same sites; a map or value without sites is the same at every
        site, on the diagonal, as it applies: `(Vector + S)(f) == f + S(f)`."""
        if not isinstance(other, SparseExtensor):
            other = _on_diagonal(self, other)
        if self.shape != other.shape:
            raise ValueError(f"sparse shapes {self.shape} and {other.shape} differ")
        batch = np.broadcast_shapes(self.cells.shape[:-1], other.cells.shape[:-1])
        return SparseExtensor(
            concatenate([self.cells.broadcast_to(batch + self.cells.shape[-1:]),
                         other.cells.broadcast_to(batch + other.cells.shape[-1:])], axis=-1),
            np.concatenate([self.rows, other.rows]),
            np.concatenate([self.columns, other.columns]),
            self.shape,
        )

    def sum(self, axis: int) -> SparseExtensor:
        """The sum of the separate maps along one of the leading axes: a map of the same pattern."""
        return SparseExtensor(self.cells.sum(axis=axis - 1 if axis < 0 else axis), self.rows, self.columns, self.shape)

    def __neg__(self) -> SparseExtensor:
        return SparseExtensor(-self.cells, self.rows, self.columns, self.shape)

    def __radd__(self, other: Extensor | GAType) -> SparseExtensor:
        return self + other

    def __sub__(self, other: SparseExtensor | Extensor | GAType) -> SparseExtensor:
        return self + (-other)

    def __rsub__(self, other: Extensor | GAType) -> SparseExtensor:
        return -self + other


def _on_diagonal(value: SparseExtensor, other: Extensor | GAType) -> SparseExtensor:
    """A map or value without sites, the same at every site of a square sparse map, on its diagonal."""
    from numga.extensor.extensor import _promote_identity

    if value.shape[0] != value.shape[1]:
        raise TypeError(f"a value without sites has no diagonal in a sparse map of shape {value.shape}")
    other = value.cells.context.lower(_promote_identity(other))
    return SparseExtensor.from_diagonal(other[..., None].broadcast_to(other.shape + (value.shape[0],)).field())


# A sparse extensor with a field on its diagonal, each element its own cell.
spdiag = SparseExtensor.from_diagonal

# --- products and applications, by cells and by operand ----------------------------------------
def _scaled(value: SparseExtensor, number: Number) -> SparseExtensor:
    return _cellwise(value, value.cells * number)


@singledispatch
def _right(other: GAType, value: SparseExtensor, product) -> SparseExtensor:
    """An open type: each cell's product with it, a map cell with the type's slot."""
    return _cellwise(value, product(value.cells, other))


@_right.register
def _right_field(other: Extensor, value: SparseExtensor, product) -> Extensor:
    return _scatter(product(value.cells, _sites(other)[..., value.columns]), value.rows, value.shape[0]).field()


@_right.register
def _right_sparse(other: SparseExtensor, value: SparseExtensor, product) -> SparseExtensor:
    left, right = _chain(value, other)
    return SparseExtensor(
        product(value.cells[..., left], other.cells[..., right]), value.rows[left], other.columns[right],
        (value.shape[0], other.shape[1]),
    )


@singledispatch
def _left(other: GAType, value: SparseExtensor, product) -> SparseExtensor:
    """An open type on the left: each cell's product with it, the couplings read from the output
    sites."""
    return _transposed(value, product(other, value.cells))


@_left.register
def _left_field(other: Extensor, value: SparseExtensor, product) -> Extensor:
    return _scatter(product(_sites(other)[..., value.rows], value.cells), value.columns, value.shape[1]).field()


@SparseExtensor.apply.register(lambda cell_type: cell_type.arity == 1)
def apply(value: SparseExtensor, operand: object):
    """Unary coupling cells applied to field elements, summed by output site. A field element
    with open inputs composes into its coupling cell and keeps those inputs; a sparse operand
    composes the field maps through their shared sites."""
    return _application(operand, value)


@singledispatch
def _application(operand: Extensor, value: SparseExtensor) -> Extensor:
    return _scatter(value.cells(_sites(operand)[..., value.columns]), value.rows, value.shape[0]).field()


@_application.register
def _application_sparse(operand: SparseExtensor, value: SparseExtensor) -> SparseExtensor:
    left, right = _chain(value, operand)
    return SparseExtensor(value.cells[..., left](operand.cells[..., right]), value.rows[left], operand.columns[right], (value.shape[0], operand.shape[1]))


# --- operations that reverse the order of the cells' products run the couplings back ---------
def _cellwise(value: SparseExtensor, cells: Extensor) -> SparseExtensor:
    return SparseExtensor(cells, value.rows, value.columns, value.shape)


def _transposed(value: SparseExtensor, cells: Extensor) -> SparseExtensor:
    return SparseExtensor(cells, value.columns, value.rows, value.shape[::-1])


@SparseExtensor.reverse.register(lambda cell_type: cell_type.arity == 0)
def reverse(value: SparseExtensor) -> SparseExtensor:
    """Every multivector cell reversed, the couplings run back: the reverse of a product is the
    product of the reverses in turn, `~(S * T) == ~T * ~S`, and composing couplings in turn is
    running them back."""
    return _transposed(value, value.cells.reverse())


@SparseExtensor.reverse.register(lambda cell_type: cell_type.arity > 0)
def reverse_maps(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's output reversed, the couplings kept: `(~S)(f) == ~(S(f))`."""
    return _cellwise(value, value.cells.reverse())


@SparseExtensor.clifford_conjugate.register(lambda cell_type: cell_type.arity == 0)
def clifford_conjugate(value: SparseExtensor) -> SparseExtensor:
    """Every multivector cell conjugated, the couplings run back, as for the reverse."""
    return _transposed(value, value.cells.clifford_conjugate())


@SparseExtensor.clifford_conjugate.register(lambda cell_type: cell_type.arity > 0)
def clifford_conjugate_maps(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's output conjugated, the couplings kept."""
    return _cellwise(value, value.cells.clifford_conjugate())


@SparseExtensor.adjugate.register(lambda cell_type: cell_type.arity == 1)
def adjugate(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's adjugate, the couplings run back: `S.adjugate()(c) & x` summed over the
    elements equals `c & S(x)` summed."""
    return _transposed(value, value.cells.adjugate())


@SparseExtensor.adjoint.register(lambda cell_type: cell_type.arity == 1)
def adjoint(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's adjoint, the couplings run back: `S.adjoint()(b).scalar_product(x)` summed
    over the elements equals `b.scalar_product(S(x))` summed."""
    return _transposed(value, value.cells.adjoint())


# --- linear algebra of map cells, on the NumPy backend --------------------------------------
def _maps(cell_type: GAType, other: GAType) -> bool:
    return cell_type.arity == 1


def _square_maps(cell_type: GAType, other: GAType) -> bool:
    return cell_type.arity == 1 and len(cell_type.subspaces[0]) == len(cell_type.subspaces[1])


@SparseExtensor.solve.register(_square_maps, fields=True)
def solve(value: SparseExtensor, rhs: Extensor) -> Extensor:
    """The fields x with value(x) == rhs: one factorization for each of the map's leading indices,
    the leading axes of rhs that the map lacks as further right sides of it."""
    from scipy.sparse.linalg import spsolve

    return _per_map(value, rhs, lambda matrix, right: spsolve(matrix, right.T).reshape(matrix.shape[1], len(right)).T)


@SparseExtensor.lstsq.register(_maps, fields=True)
def lstsq(value: SparseExtensor, rhs: Extensor) -> Extensor:
    """The smallest fields x, in the sum of their squared coefficients, minimizing that of
    value(x) - rhs, for each of the map's leading indices and each right side: a singular system's
    gauge, such as a translation, is left at zero."""
    from scipy.sparse.linalg import lsqr

    return _per_map(value, rhs, lambda matrix, right: np.stack([
        lsqr(matrix, column, atol=0.0, btol=0.0, iter_lim=10 * matrix.shape[1])[0] for column in right
    ]))


@SparseExtensor.eigh.register(_square_maps)
def eigh(value: SparseExtensor, metric: SparseExtensor, count: int) -> tuple[Extensor, Extensor]:
    """The count eigenpairs nearest zero of value(x) == eigenvalue * metric(x), for a symmetric
    value and a positive-definite metric, for each of their leading indices: values `[..., count]
    Scalar`, fields `[..., count]` over the elements, orthonormal in the metric. The spectrum is inverted
    about a point below zero by the square root of the precision, relative to the pencil's scale, so
    a semidefinite value with a null space still factorizes, to half the precision's digits."""
    from scipy.sparse.linalg import eigsh

    batch = np.broadcast_shapes(value.cells.shape[:-1], metric.cells.shape[:-1])
    values, vectors = [], []
    for case in np.ndindex(batch):
        matrix, mass = _matrix(value, _blocks(value, batch)[case]), _matrix(metric, _blocks(metric, batch)[case])
        scale = np.abs(matrix.diagonal()).max() / np.abs(mass.diagonal()).max()
        found, modes = eigsh(matrix, k=count, M=mass, sigma=-np.sqrt(np.finfo(matrix.dtype).eps) * scale)
        values.append(found)
        vectors.append(modes.T)
    fields = _field(value, np.reshape(vectors, batch + (count, -1)))
    return fields.context.multivector.scalar(np.reshape(values, batch + (count, 1))), fields


def _per_map(value: SparseExtensor, rhs: Extensor, solver) -> Extensor:
    """The solver's fields for each of the map's leading indices, on its matrix and its right sides
    `[sides, elements * blades]`: those of rhs at that index, along every leading axis the map
    lacks or holds once."""
    rhs = _sites(rhs)
    batch = np.broadcast_shapes(value.cells.shape[:-1], rhs.shape[:-1])
    maps = (1,) * (len(batch) - (value.cells.ndim - 1)) + value.cells.shape[:-1]
    right = np.broadcast_to(np.asarray(rhs.kernel), batch + rhs.kernel.shape[rhs.ndim - 1:])
    right = right.reshape(batch + (-1,))
    solved = np.empty(batch + (value.shape[1] * len(value.cells.axes[1]),), dtype=right.dtype)
    blocks = _blocks(value, maps)
    for case in np.ndindex(maps):
        sides = tuple(slice(None) if size == 1 else index for index, size in zip(case, maps))
        block = right[sides]
        right_sides = block.reshape(int(np.prod(block.shape[:-1])), block.shape[-1])
        solved[sides] = solver(_matrix(value, blocks[case]), right_sides).reshape(block.shape[:-1] + (-1,))
    return _field(value, solved)


def _blocks(value: SparseExtensor, batch: tuple[int, ...]) -> np.ndarray:
    """The map cells' coefficients `[..., n, out blades, in blades]`, broadcast to the batch."""
    blocks = np.asarray(value.cells.kernel)
    return np.broadcast_to(blocks, batch + blocks.shape[-3:])


def _matrix(value: SparseExtensor, blocks: np.ndarray):
    """One map's cell coefficients `[n, out blades, in blades]` as a SciPy matrix of element blocks."""
    from scipy.sparse import coo_matrix

    _, height, width = blocks.shape
    rows = (value.rows[:, None, None] * height + np.arange(height)[:, None]).repeat(width, axis=2)
    columns = (value.columns[:, None, None] * width + np.arange(width)[None, :]).repeat(height, axis=1)
    return coo_matrix(
        (blocks.ravel(), (rows.ravel(), columns.ravel())), shape=(value.shape[0] * height, value.shape[1] * width),
    ).tocsc()


def _field(value: SparseExtensor, coefficients: np.ndarray) -> Extensor:
    """Coefficients `[..., elements * blades]` as a field of the map cells' input type."""
    subspace = value.cells.axes[1]
    return value.cells.context.extensor(
        value.cells.algebra.gatype(subspace), coefficients.reshape(coefficients.shape[:-1] + (-1, len(subspace))),
    ).field()


def _chain(first: SparseExtensor, second: SparseExtensor) -> tuple[np.ndarray, np.ndarray]:
    """The pairs of cells a composition joins: where the first's column is the second's row."""
    if first.shape[1] != second.shape[0]:
        raise ValueError(f"sparse shapes {first.shape} and {second.shape} do not chain")
    return _matching_pairs(first.columns, second.rows)


def _matching_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of positions (i, j) with left[i] == right[j]."""
    order = np.argsort(right, kind="stable")
    ordered = right[order]
    start = np.searchsorted(ordered, left, side="left")
    counts = np.searchsorted(ordered, left, side="right") - start
    first = np.cumsum(counts) - counts
    return np.repeat(np.arange(len(left)), counts), order[np.repeat(start - first, counts) + np.arange(counts.sum())]


def _scatter(cells: Extensor, index: np.ndarray, size: int) -> Extensor:
    """Cells `[..., n]` summed by their index into `[..., size]`."""
    zeros = cells.context.xp.zeros(cells.shape[:-1] + (size,) + cells.structural_shape, dtype=cells.dtype)
    return cells.context.extensor(cells.gatype, zeros).at[..., index].add(cells)


def _sites(field: Extensor) -> Extensor:
    """A field's elements along its last batch axis, where the couplings index them."""
    if [slot for slot, _ in field.gatype.fields] != [0]:
        raise TypeError(f"a sparse field map acts on fields over its output's sites; {field.gatype.signature} is not one")
    return field.batch()
