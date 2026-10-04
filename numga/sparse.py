"""Sparse extensors: linear maps between fields whose elements couple sparsely.

A field is an Extensor whose last batch axis indexes the elements of a collection: the vertices or
the faces of a mesh, the poses of a graph; leading batch axes hold separate fields. A sparse
extensor couples a few elements of one field to a few of another; each coupling is a cell, an
ordinary extensor, and the sparsity is in which elements couple, not in the blades of a cell. Its
cells carry the couplings on their last batch axis too, and leading batch axes hold separate maps
that share one pattern of couplings, broadcasting against the leading axes of what they act on.

A sparse extensor acts as its cells do, and forwards to them what is defined on them. A cell with
slots is a map, and is applied: `S(field)` and `S(T)`, with `S.adjugate()` and `S.adjoint()` taking
every cell's adjugate and adjoint. A nullary cell, a multivector, acts by its product: `S * field` and `S * T`, with `~S`
reversing every cell. Either swaps inputs and outputs, as reversing a product swaps its factors and
an adjugate pulls back. A product with an open type, `S * Even`, leaves a slot open, so that
multivector cells become maps; solves and eigenproblems on the NumPy backend, through SciPy, act on
map cells.
"""

from __future__ import annotations

from functools import singledispatch
from numbers import Number
from typing import overload

import numpy as np

from numga.extension import ExtensionMethod
from numga.extensor import Extensor, concatenate
from numga.gatype import GAType


class SparseExtensor:
    """A linear map from a field of `shape[1]` elements to one of `shape[0]`: extensor cells
    `[..., n]`, each coupling the input element its column names to the output element its row
    names, with leading axes for separate maps of the same pattern. Cells coupling the same pair are
    summed at construction, and stored by row."""

    __slots__ = ("cells", "rows", "columns", "shape")
    product = ExtensionMethod("product")
    apply = ExtensionMethod("apply")
    reverse = ExtensionMethod("reverse")
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
    def from_diagonal(cls, field: Extensor) -> SparseExtensor:
        """Each element of the field as the cell on its own row and column."""
        index = np.arange(field.shape[-1])
        return cls(field, index, index, (len(index), len(index)))

    def diagonal(self) -> Extensor:
        """The cells on the diagonal as a field, each at its row; zero where a row has none."""
        on = self.rows == self.columns
        return _scatter(self.cells[..., on], self.rows[on], self.shape[0])

    @property
    def gatype(self) -> GAType:
        """The cells' type: a sparse extensor acts as its cells do."""
        return self.cells.gatype

    # --- multivector cells: the product -----------------------------------------------------
    @overload
    def __mul__(self, other: SparseExtensor) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: Extensor) -> Extensor: ...
    @overload
    def __mul__(self, other: GAType) -> SparseExtensor: ...
    @overload
    def __mul__(self, other: float) -> SparseExtensor: ...

    def __mul__(self, other):
        """Times a number, every cell; otherwise the product of multivector cells."""
        return _times(other, self)

    def __rmul__(self, scalar: Number) -> SparseExtensor:
        return _scaled(scalar, self)

    def __invert__(self) -> SparseExtensor:
        return self.reverse()

    # --- map cells: application ------------------------------------------------------------------
    @overload
    def __call__(self, operand: SparseExtensor) -> SparseExtensor: ...
    @overload
    def __call__(self, operand: Extensor) -> Extensor: ...

    def __call__(self, operand):
        return self.apply(operand)

    # --- sums ----------------------------------------------------------------------------------
    def __add__(self, other: SparseExtensor) -> SparseExtensor:
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

    def __neg__(self) -> SparseExtensor:
        return SparseExtensor(-self.cells, self.rows, self.columns, self.shape)

    def __sub__(self, other: SparseExtensor) -> SparseExtensor:
        return self + (-other)


# A sparse extensor with a field on its diagonal, each element its own cell.
spdiag = SparseExtensor.from_diagonal

# --- products and applications, by cells and by operand ----------------------------------------
@singledispatch
def _times(other: object, value: SparseExtensor):
    return value.product(other)


@_times.register
def _scaled(other: Number, value: SparseExtensor) -> SparseExtensor:
    return SparseExtensor(value.cells * other, value.rows, value.columns, value.shape)


@SparseExtensor.product.register(lambda t: t.arity == 0)
def product(value: SparseExtensor, other: object):
    """Multivector cells by their product: times a sparse extensor, the composition; times a
    field, each cell times the element its column names, summed by row; times an open type, the
    cells as maps with that input."""
    return _product(other, value)


@SparseExtensor.apply.register(lambda t: t.arity == 1)
def apply(value: SparseExtensor, operand: object):
    """Map cells applied: to a field, each cell to the element its column names, summed by row;
    to a sparse extensor, composed with it."""
    return _application(operand, value)


@singledispatch
def _product(other: GAType, value: SparseExtensor) -> SparseExtensor:
    return SparseExtensor(value.cells * other, value.rows, value.columns, value.shape)


@_product.register
def _product_field(other: Extensor, value: SparseExtensor) -> Extensor:
    return _scatter(value.cells * other[..., value.columns], value.rows, value.shape[0])


@_product.register
def _product_sparse(other: SparseExtensor, value: SparseExtensor) -> SparseExtensor:
    left, right = _chain(value, other)
    return SparseExtensor(value.cells[..., left] * other.cells[..., right], value.rows[left], other.columns[right], (value.shape[0], other.shape[1]))


@singledispatch
def _application(operand: Extensor, value: SparseExtensor) -> Extensor:
    return _scatter(value.cells(operand[..., value.columns]), value.rows, value.shape[0])


@_application.register
def _application_sparse(operand: SparseExtensor, value: SparseExtensor) -> SparseExtensor:
    left, right = _chain(value, operand)
    return SparseExtensor(value.cells[..., left](operand.cells[..., right]), value.rows[left], operand.columns[right], (value.shape[0], operand.shape[1]))


# --- forwarded to the cells -------------------------------------------------------------------
@SparseExtensor.reverse.register(lambda t: t.arity == 0)
def reverse(value: SparseExtensor) -> SparseExtensor:
    """Every multivector cell reversed, inputs and outputs swapped: the reverse of a product of
    sparse extensors is the product of their reverses in turn."""
    return SparseExtensor(value.cells.reverse(), value.columns, value.rows, value.shape[::-1])


@SparseExtensor.adjugate.register(lambda t: t.arity == 1)
def adjugate(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's adjugate, inputs and outputs swapped: `S.adjugate()(c) & x` summed over the
    elements equals `c & S(x)` summed."""
    return SparseExtensor(value.cells.adjugate(), value.columns, value.rows, value.shape[::-1])


@SparseExtensor.adjoint.register(lambda t: t.arity == 1)
def adjoint(value: SparseExtensor) -> SparseExtensor:
    """Every map cell's adjoint, inputs and outputs swapped: `S.adjoint()(b).scalar_product(x)` summed
    over the elements equals `b.scalar_product(S(x))` summed."""
    return SparseExtensor(value.cells.adjoint(), value.columns, value.rows, value.shape[::-1])


# --- linear algebra of map cells, on the NumPy backend --------------------------------------
def _maps(cells: GAType, other: GAType) -> bool:
    return cells.arity == 1


def _square_maps(cells: GAType, other: GAType) -> bool:
    return cells.arity == 1 and len(cells.subspaces[0]) == len(cells.subspaces[1])


@SparseExtensor.solve.register(_square_maps)
def solve(value: SparseExtensor, rhs: Extensor) -> Extensor:
    """The fields x with value(x) == rhs: one factorization for each of the map's leading indices,
    the leading axes of rhs that the map lacks as further right sides of it."""
    from scipy.sparse.linalg import spsolve

    return _per_map(value, rhs, lambda matrix, right: spsolve(matrix, right.T).reshape(matrix.shape[1], len(right)).T)


@SparseExtensor.lstsq.register(_maps)
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
    Scalar`, fields `[..., count, elements]` orthonormal in the metric. The spectrum is inverted
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
    )


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
    """Cells `[..., n]` summed by their index into a field `[..., size]`."""
    zeros = cells.context.xp.zeros(cells.shape[:-1] + (size,) + cells.structural_shape, dtype=cells.dtype)
    return cells.context.extensor(cells.gatype, zeros).at[..., index].add(cells)
