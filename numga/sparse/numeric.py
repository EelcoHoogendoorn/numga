"""Maps of plain numbers between the sites of fields, stored sparsely, and the bookkeeping every sparse map shares.

A `Sparse` map holds numbers, `cells` of shape `[..., couplings]`, each coupling an input site (a column) to an output
site (a row). Leading axes hold separate maps sharing one pattern of couplings, as the corners and the facets of a
topology's simplices do. Such a map belongs to no algebra; the incidences of a mesh are of its kind.

A number has one action, to scale, and it scales whatever it meets, so a map of numbers needs no algebra to act. Called
on an array `[..., N]` it reads it as a field of numbers over its input sites: `S(values)` is at each output site the
sum of each number times the value at its input site. Between themselves these maps compose, `S * T`, and add, and
stay numbers; times a number, or an array of them broadcast against the maps' leading axes, every cell is scaled.

`S.adjugate()` runs the couplings back and keeps the numbers. It is defined by a pairing, which asks for no metric: a
number passes through one, so `c & (S * x)` summed over the output sites equals `(S.adjugate() * c) & x` summed over
the input sites, whatever the types of `c` and `x`. `S.adjoint()` is the same map: a number passes through a scalar
product as well, whatever the metric, and between plain numbers the scalar product is their product.

Where a map of numbers meets an operand of an algebra, a field, a type or a map of typed cells, its numbers become
scalars of that algebra, as an untyped array does beside an extensor. This module knows no algebra: `typed` is where a
library that has one registers its operands.
"""

from __future__ import annotations

from functools import singledispatch
from numbers import Number
from typing import Any

import numpy as np


class Sparse:
    """Numbers coupling `shape[1]` input sites to `shape[0]` output sites.

    `cells[..., coupling]` couples `columns[coupling]` to `rows[coupling]`; duplicate couplings add at construction.
    Leading axes of the cells hold separate maps of the same pattern.
    """

    __slots__ = ("cells", "rows", "columns", "shape")
    # NumPy leaves an array's product with a map to the map.
    __array_ufunc__ = None

    def __init__(self, cells: Any, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> None:
        if cells.ndim == 0 or not cells.shape[-1] == len(rows) == len(columns):
            raise ValueError("a sparse map needs one row and one column per cell")
        pairs, axes = np.unique(np.asarray(rows) * shape[1] + np.asarray(columns), return_inverse=True)
        self.cells = self._scattered(cells, axes, len(pairs))
        self.rows, self.columns = np.divmod(pairs, shape[1])
        self.shape = tuple(shape)

    def _scattered(self, cells: np.ndarray, index: np.ndarray, size: int) -> np.ndarray:
        """Cells `[..., n]` summed by their index into `[..., size]`."""
        return summed_runs(np.asarray(cells, dtype=float), index, size, cells.ndim - 1)

    @classmethod
    def from_columns(cls, columns: np.ndarray, cells: Any, size: int) -> Sparse:
        """The map from a field of `size` elements coupling output element r to the input elements
        `columns[r]` names, `[R, k]`, through `cells[..., r, :]` of the same shape."""
        rows = np.repeat(np.arange(len(columns)), columns.shape[1])
        return cls(cells.reshape(cells.shape[:-2] + (-1,)), rows, columns.reshape(-1), (len(columns), size))

    @classmethod
    def from_indices(cls, cells: Any, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> Sparse:
        """Cells `[..., *indices]`, each coupling the row and the column given at its place in `rows` and
        `columns`, which broadcast against each other to the cells' trailing axes."""
        rows, columns = np.broadcast_arrays(rows, columns)
        return cls(cells.reshape(cells.shape[:cells.ndim - rows.ndim] + (rows.size,)), rows.reshape(-1), columns.reshape(-1), shape)

    @staticmethod
    def selection(index: np.ndarray, size: int) -> Sparse:
        """The map that reads, at each of its output sites, the input site of `size` an index names,
        `index[..., output]`: applied to a field, the elements at those sites. Leading axes of the index
        give one map each, all of one pattern, each map's numbers one where it reads and zero elsewhere."""
        leading, count = index.shape[:-1], index.shape[-1]
        cases = int(np.prod(leading, dtype=int))
        reads = np.broadcast_to(np.eye(cases)[:, None, :], (cases, count, cases)).reshape(leading + (count, cases))
        return Sparse.from_indices(reads, np.arange(count)[:, None], index.reshape(cases, count).T, (count, size))

    def diagonal(self) -> Any:
        """The cells on the diagonal `[..., shape[0]]`, each at its row; zero where a row has none."""
        on = self.rows == self.columns
        return self._scattered(self.cells[..., on], self.rows[on], self.shape[0])

    # --- products: a number scales what it meets, on either side -----------------------------
    def __mul__(self, other: Any) -> Any:
        """Times a number or an array of them, every cell scaled, an array broadcast against the maps' leading axes;
        times a map of numbers, their composition along the paths through shared sites; times an operand of an algebra,
        the map typed in that algebra times it."""
        return _product(other, self)

    __rmul__ = __mul__

    def __truediv__(self, number: Number) -> Sparse:
        return type(self)(self.cells / number, self.rows, self.columns, self.shape)

    def __call__(self, operand: Any) -> Any:
        """The map applied: to an array `[..., N]`, a field of numbers over the input sites, at each output site the
        sum of each number times the value at its input site; to a field or a map, as its product applies or composes
        it."""
        return _applied(operand, self)

    def adjugate(self) -> Sparse:
        """The same numbers, the couplings run back: the map that carries a pairing over, `c & (S * x)` summed over
        the output sites equal to `(S.adjugate() * c) & x` summed over the input sites, in every algebra."""
        return Sparse(self.cells, self.columns, self.rows, self.shape[::-1])

    def adjoint(self) -> Sparse:
        """The same numbers, the couplings run back: the map that carries a scalar product over,
        `b.scalar_product(S * x)` summed over the output sites equal to `(S.adjoint() * b).scalar_product(x)` summed
        over the input sites, whatever the metric."""
        return Sparse(self.cells, self.columns, self.rows, self.shape[::-1])

    def __getitem__(self, index: Any) -> Sparse:
        """The maps at an index of the leading axes, which hold separate maps of one pattern:
        `S[..., None]` gives the maps an axis to broadcast against the batch of a field."""
        index = index if isinstance(index, tuple) else (index,)
        within = (slice(None),) if any(item is Ellipsis for item in index) else (Ellipsis,)
        return type(self)(self.cells[index + within], self.rows, self.columns, self.shape)

    # --- sums ----------------------------------------------------------------------------------
    def __add__(self, other: Any) -> Any:
        """The sum of two maps of the same sites; beside an operand of an algebra, the map typed in that algebra
        plus it."""
        return _sum(other, self)

    def __radd__(self, other: Any) -> Any:
        return self + other

    def __neg__(self) -> Sparse:
        return type(self)(-self.cells, self.rows, self.columns, self.shape)

    def __sub__(self, other: Any) -> Any:
        return self + (-other)

    def __rsub__(self, other: Any) -> Any:
        return -self + other

    def sum(self, axis: int) -> Sparse:
        """The sum of the separate maps along one of the leading axes: a map of the same pattern."""
        return type(self)(self.cells.sum(axis=axis - 1 if axis < 0 else axis), self.rows, self.columns, self.shape)

    def mean(self, axis: int) -> Sparse:
        """The mean of the separate maps along one of the leading axes: a map of the same pattern."""
        return self.sum(axis) / self.cells.shape[axis - 1 if axis < 0 else axis]


@singledispatch
def typed(operand: object, value: Sparse) -> Any:
    """A map of plain numbers as it meets an operand of an algebra: the same couplings, its numbers scalars of that
    algebra, as an untyped array beside an extensor becomes a batch of its scalars. A library that has an algebra
    registers its kinds of operand here."""
    raise TypeError(f"a map of plain numbers meets numbers, arrays, maps of numbers and operands of an algebra, not {type(operand).__name__}")


# --- products and sums, by operand -------------------------------------------------------------
@singledispatch
def _product(other: object, value: Sparse) -> Any:
    """An operand of an algebra: the map typed in it, times the operand."""
    return typed(other, value) * other


@_product.register(Number)
@_product.register(np.ndarray)
def _product_numbers(other: Number | np.ndarray, value: Sparse) -> Sparse:
    return Sparse(value.cells * other, value.rows, value.columns, value.shape)


@_product.register
def _product_sparse(other: Sparse, value: Sparse) -> Sparse:
    left, right = chain(value, other)
    return Sparse(value.cells[..., left] * other.cells[..., right], value.rows[left], other.columns[right], (value.shape[0], other.shape[1]))


@singledispatch
def _applied(operand: object, value: Sparse) -> Any:
    """A field or a map: what the product makes of it."""
    return value * operand


@_applied.register
def _applied_array(operand: np.ndarray, value: Sparse) -> np.ndarray:
    return value._scattered(value.cells * operand[..., value.columns], value.rows, value.shape[0])


@singledispatch
def _sum(other: object, value: Sparse) -> Any:
    """An operand of an algebra: the map typed in it, plus the operand."""
    return typed(other, value) + other


@_sum.register
def _sum_sparse(other: Sparse, value: Sparse) -> Sparse:
    if value.shape != other.shape:
        raise ValueError(f"sparse shapes {value.shape} and {other.shape} differ")
    batch = np.broadcast_shapes(value.cells.shape[:-1], other.cells.shape[:-1])
    return Sparse(
        np.concatenate([np.broadcast_to(value.cells, batch + value.cells.shape[-1:]),
                        np.broadcast_to(other.cells, batch + other.cells.shape[-1:])], axis=-1),
        np.concatenate([value.rows, other.rows]),
        np.concatenate([value.columns, other.columns]),
        value.shape,
    )


# --- the bookkeeping of couplings, for maps of any cells ---------------------------------------
def chain(first: Sparse, second: Sparse) -> tuple[np.ndarray, np.ndarray]:
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


def summed_runs(array: np.ndarray, index: np.ndarray, size: int, axis: int) -> np.ndarray:
    """An array summed along an axis by index, into `size` places along it: sorted by index and each run summed by
    `reduceat`, pairwise and so accurate on long runs."""
    summed = np.zeros(array.shape[:axis] + (size,) + array.shape[axis + 1:], dtype=array.dtype)
    if len(index):
        order = np.argsort(index, kind="stable")
        ordered = index[order]
        starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        summed[(slice(None),) * axis + (ordered[starts],)] = np.add.reduceat(np.take(array, order, axis=axis), starts, axis=axis)
    return summed
