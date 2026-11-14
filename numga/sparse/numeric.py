"""`Sparse` stores linear couplings of plain floating-point numbers between `shape[1]` input sites
and `shape[0]` output sites, such as the incidence and boundary operators of a simplicial
`Topology`. In `cells` of shape `[..., n_couplings]`, entry `i` along the trailing axis couples
`columns[i]` to `rows[i]`, summing duplicate `(row, column)` pairs into canonical sorted order at
construction. Leading axes of `cells` batch independent maps that share one sparsity pattern, such
as the local corners or facets of a simplex family.

Because real numbers act by scaling in any vector space, `Sparse` belongs to no geometric algebra.
Calling `S(values)` on a NumPy array `[..., shape[1]]` gathers input entries along `columns`,
multiplies by `cells`, and sums into `[..., shape[0]]`; multiplying two `Sparse` maps (`S * T`)
chains paths across shared intermediate sites via `chain`, and multiplying by scalars or
leading-batch arrays scales `cells` directly. Both `adjugate` and `adjoint` swap `rows` and
`columns` while keeping `cells` unchanged, since real numbers pass unchanged through regressive
pairings (`(S.adjugate() * c) & x` summed over inputs equals `c & (S * x)` summed over outputs) and
through metric scalar products alike.

When an arithmetic operation or call pairs a `Sparse` map with an algebraic operand (`Extensor`,
`GAType`, or `SparseExtensor`), `typed` promotes its numeric weights to extrinsic `Scalar` cells in
that operand's algebra without importing `numga`'s algebraic layers into this module. Sparse field
couplings and simplicial operators are introduced in `docs/fields.md` ("Sparse storage").
"""

from __future__ import annotations

from functools import singledispatch
from numbers import Number
from typing import Any

import numpy as np


class Sparse:
    """Numeric site couplings store deduplicated `(rows, columns)` indices shared across any
    leading batch axes of `cells`."""

    __slots__ = ("cells", "rows", "columns", "shape")
    # Defer NumPy array ufuncs so `ndarray * Sparse` dispatches through `Sparse.__rmul__`.
    __array_ufunc__ = None

    def __init__(self, cells: Any, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> None:
        if cells.ndim == 0 or not cells.shape[-1] == len(rows) == len(columns):
            raise ValueError("a sparse map needs one row and one column per cell")
        pairs, axes = np.unique(np.asarray(rows) * shape[1] + np.asarray(columns), return_inverse=True)
        self.cells = self._scattered(cells, axes, len(pairs))
        self.rows, self.columns = np.divmod(pairs, shape[1])
        self.shape = tuple(shape)

    def _scattered(self, cells: np.ndarray, index: np.ndarray, size: int) -> np.ndarray:
        return summed_runs(np.asarray(cells, dtype=float), index, size, cells.ndim - 1)

    @classmethod
    def from_columns(cls, columns: np.ndarray, cells: Any, n_inputs: int) -> Sparse:
        """Each output row `r` couples to the input sites listed in `columns[r]` with weights `cells[..., r, :]`."""
        rows = np.repeat(np.arange(len(columns)), columns.shape[1])
        return cls(cells.reshape(cells.shape[:-2] + (-1,)), rows, columns.reshape(-1), (len(columns), n_inputs))

    @classmethod
    def from_indices(cls, cells: Any, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> Sparse:
        """Broadcasting `rows` and `columns` flattens them against the trailing axes of `cells`."""
        rows, columns = np.broadcast_arrays(rows, columns)
        return cls(cells.reshape(cells.shape[:cells.ndim - rows.ndim] + (rows.size,)), rows.reshape(-1), columns.reshape(-1), shape)

    @staticmethod
    def selection(indices: np.ndarray, n_inputs: int) -> Sparse:
        """Output site `i` reads input site `indices[..., i]` with weight `1`, batching leading
        index axes across separate maps of a shared sparsity pattern."""
        leading, count = indices.shape[:-1], indices.shape[-1]
        cases = int(np.prod(leading, dtype=int))
        reads = np.broadcast_to(np.eye(cases)[:, None, :], (cases, count, cases)).reshape(leading + (count, cases))
        return Sparse.from_indices(reads, np.arange(count)[:, None], indices.reshape(cases, count).T, (count, n_inputs))

    def diagonal(self) -> Any:
        """Gathering entries with `rows == columns` returns a dense array `[..., shape[0]]`, filling
        uncoupled rows with zero."""
        on = self.rows == self.columns
        return self._scattered(self.cells[..., on], self.rows[on], self.shape[0])

    # --- products --------------------------------------------------------------------------
    def __mul__(self, other: Any) -> Any:
        """Numbers and leading-batch arrays scale `cells`, a second `Sparse` map composes across
        shared sites, and algebraic operands promote `self` via `typed`."""
        return _product(other, self)

    __rmul__ = __mul__

    def __truediv__(self, number: Number) -> Sparse:
        return type(self)(self.cells / number, self.rows, self.columns, self.shape)

    def __call__(self, operand: Any) -> Any:
        """Applying `self` to a NumPy array `[..., shape[1]]` sums weighted input entries into
        `[..., shape[0]]`, while algebraic operands evaluate through `*`."""
        return _applied(operand, self)

    def adjugate(self) -> Sparse:
        """Swapping `rows` and `columns` transposes the site couplings so `(S.adjugate() * c) & x`
        summed over inputs equals `c & (S * x)` summed over outputs."""
        return Sparse(self.cells, self.columns, self.rows, self.shape[::-1])

    def adjoint(self) -> Sparse:
        """Swapping `rows` and `columns` transposes the site couplings across metric scalar
        products, coinciding with `adjugate` for real numeric weights."""
        return Sparse(self.cells, self.columns, self.rows, self.shape[::-1])

    def __getitem__(self, index: Any) -> Sparse:
        """Indexing slices or inserts leading batch axes of `cells` while keeping the
        `(rows, columns)` coupling pattern intact."""
        index = index if isinstance(index, tuple) else (index,)
        within = (slice(None),) if any(item is Ellipsis for item in index) else (Ellipsis,)
        return type(self)(self.cells[index + within], self.rows, self.columns, self.shape)

    # --- sums ------------------------------------------------------------------------------
    def __add__(self, other: Any) -> Any:
        """Adding two `Sparse` maps of equal `shape` merges their couplings, while algebraic
        operands promote `self` via `typed`."""
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
        """Summing `cells` along a leading batch `axis` combines separate maps of the same sparsity pattern."""
        return type(self)(self.cells.sum(axis=axis - 1 if axis < 0 else axis), self.rows, self.columns, self.shape)

    def mean(self, axis: int) -> Sparse:
        """Averaging `cells` along a leading batch `axis` divides `sum(axis)` by that batch axis length."""
        return self.sum(axis) / self.cells.shape[axis - 1 if axis < 0 else axis]


@singledispatch
def typed(operand: object, numeric_map: Sparse) -> Any:
    """Algebraic operand types register here to promote a numeric `Sparse` map into extrinsic
    `Scalar` cells of `operand`'s algebra."""
    raise TypeError(f"a map of plain numbers meets numbers, arrays, maps of numbers and operands of an algebra, not {type(operand).__name__}")


# --- products and sums, by operand ---------------------------------------------------------
@singledispatch
def _product(other: object, numeric_map: Sparse) -> Any:
    return typed(other, numeric_map) * other


@_product.register(Number)
@_product.register(np.ndarray)
def _product_numbers(other: Number | np.ndarray, numeric_map: Sparse) -> Sparse:
    return Sparse(numeric_map.cells * other, numeric_map.rows, numeric_map.columns, numeric_map.shape)


@_product.register
def _product_sparse(other: Sparse, numeric_map: Sparse) -> Sparse:
    left, right = chain(numeric_map, other)
    return Sparse(
        numeric_map.cells[..., left] * other.cells[..., right],
        numeric_map.rows[left], other.columns[right], (numeric_map.shape[0], other.shape[1]),
    )


@singledispatch
def _applied(operand: object, numeric_map: Sparse) -> Any:
    return numeric_map * operand


@_applied.register
def _applied_array(operand: np.ndarray, numeric_map: Sparse) -> np.ndarray:
    return numeric_map._scattered(numeric_map.cells * operand[..., numeric_map.columns], numeric_map.rows, numeric_map.shape[0])


@singledispatch
def _sum(other: object, numeric_map: Sparse) -> Any:
    return typed(other, numeric_map) + other


@_sum.register
def _sum_sparse(other: Sparse, numeric_map: Sparse) -> Sparse:
    if numeric_map.shape != other.shape:
        raise ValueError(f"sparse shapes {numeric_map.shape} and {other.shape} differ")
    batch = np.broadcast_shapes(numeric_map.cells.shape[:-1], other.cells.shape[:-1])
    return Sparse(
        np.concatenate([np.broadcast_to(numeric_map.cells, batch + numeric_map.cells.shape[-1:]),
                        np.broadcast_to(other.cells, batch + other.cells.shape[-1:])], axis=-1),
        np.concatenate([numeric_map.rows, other.rows]),
        np.concatenate([numeric_map.columns, other.columns]),
        numeric_map.shape,
    )


# --- coupling bookkeeping ------------------------------------------------------------------
def chain(first: Sparse, second: Sparse) -> tuple[np.ndarray, np.ndarray]:
    """Composing `first` after `second` pairs coupling positions `(i, j)` wherever `first.columns[i] == second.rows[j]`."""
    if first.shape[1] != second.shape[0]:
        raise ValueError(f"sparse shapes {first.shape} and {second.shape} do not chain")
    return _matching_pairs(first.columns, second.rows)


def _matching_pairs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(right, kind="stable")
    ordered = right[order]
    start = np.searchsorted(ordered, left, side="left")
    counts = np.searchsorted(ordered, left, side="right") - start
    first = np.cumsum(counts) - counts
    return np.repeat(np.arange(len(left)), counts), order[np.repeat(start - first, counts) + np.arange(counts.sum())]


def summed_runs(array: np.ndarray, index: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Sorting `index` stably and reducing contiguous runs via `np.add.reduceat` sums `array` along
    `axis` into `size` buckets with pairwise accuracy."""
    summed = np.zeros(array.shape[:axis] + (size,) + array.shape[axis + 1:], dtype=array.dtype)
    if len(index):
        order = np.argsort(index, kind="stable")
        ordered = index[order]
        starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        summed[(slice(None),) * axis + (ordered[starts],)] = np.add.reduceat(np.take(array, order, axis=axis), starts, axis=axis)
    return summed
