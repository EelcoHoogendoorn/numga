"""`SymbolicKernel` stores the immutable, backend-independent coefficient tensors of `Extensor`
objects in `algebra.exact`. `OperatorFactory` builds basis-blade operators with `int8` entries
because basis products, duals, and layout transforms carry integer signs (`-1`, `0`, `+1`), while
exact arithmetic promotes to wider integer or floating NumPy dtypes only when scaled by rational or
floating scalars. Precomputing a content hash over shape, dtype, and raw bytes lets exact kernels
participate in memoized operator tables while keeping the underlying NumPy buffer read-only.

Structural tensor operations (`transpose`, `moveaxis`, `take`, `trace`, `expand_dims`, `tensordot`)
mirror backend array primitives so `ExactContext` can compose, slice, and contract operators
symbolically before any numerical backend is chosen. Specialized exact helpers preserve integer
exactness where linear algebra permits. `halved` uses floor division whenever every entry is even
(as in symmetrized basis products and commutators), `inverse` accepts a rounded matrix inverse only
when integer multiplication recovers the identity exactly, and `materialize` caches read-only host
arrays per target dtype without ever caching traced JAX or PyTorch arrays. Exact table storage,
cached materialization, and integer inverses are described in `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from numbers import Integral, Rational, Real
from types import NotImplementedType
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _scalar(scalar: object) -> Real:
    if not isinstance(scalar, Real):
        raise TypeError(f"symbolic kernels scale by real scalars; got {type(scalar).__name__}")
    return float(scalar) if isinstance(scalar, Rational) and not isinstance(scalar, Integral) else scalar


class SymbolicKernel:
    """Read-only NumPy storage with precomputed content hashing backs exact, unbatched `Extensor`
    kernels."""

    __slots__ = ("_values", "_hash")

    def __init__(self, values: object, shape: Sequence[int] | None = None) -> None:
        if isinstance(values, SymbolicKernel):
            array = values._values
        else:
            array = np.asarray(values)
            if array.dtype == object:
                array = np.array([_scalar(entry) for entry in array.flat]).reshape(array.shape)
            if array.dtype.kind not in "iubf":
                raise TypeError(f"symbolic coefficients must be real numbers; got {array.dtype}")
            if array.dtype.kind == "b":
                array = array.astype(np.int8)
        if shape is not None:
            shape = tuple(int(size) for size in shape)
            if any(size < 0 for size in shape):
                raise ValueError("symbolic kernel dimensions must be non-negative")
            if int(np.prod(shape, dtype=int)) != array.size:
                raise ValueError(f"shape {shape} does not contain {array.size} values")
            array = array.reshape(shape)
        array = np.array(array)
        array.flags.writeable = False
        object.__setattr__(self, "_values", array)
        object.__setattr__(self, "_hash", hash((array.shape, array.dtype.str, array.tobytes())))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SymbolicKernel instances are immutable")

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> SymbolicKernel:
        """Zero-filled `int8` storage initializes an empty exact kernel of the given `shape`."""
        return cls(np.zeros(tuple(int(size) for size in shape), dtype=np.int8))

    @classmethod
    def identity(cls, size: int) -> SymbolicKernel:
        """Diagonal unit entries in `int8` represent the `size`-by-`size` exact identity matrix."""
        return cls(np.eye(size, dtype=np.int8))

    @property
    def values(self) -> np.ndarray:
        """Read-only NumPy view exposes the underlying coefficient array without copying."""
        return self._values

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def ndim(self) -> int:
        return self._values.ndim

    @property
    def size(self) -> int:
        return self._values.size

    def to_array(self) -> np.ndarray:
        """Copying the underlying buffer returns a writable NumPy array of the coefficients."""
        return self._values.copy()

    @lru_cache(maxsize=None)
    def materialize(self, dtype: DTypeLike = np.float64) -> np.ndarray:
        """Casting the stored coefficients to `dtype` caches a read-only host NumPy array per
        target dtype without retaining traced backend arrays."""
        values = self._values.astype(dtype)
        values.flags.writeable = False
        return values

    def transpose(self, permutation: Sequence[int]) -> SymbolicKernel:
        return type(self)(self._values.transpose(tuple(int(axis) for axis in permutation)))

    def moveaxis(
        self,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> SymbolicKernel:
        return type(self)(np.moveaxis(self._values, source, destination))

    def take(self, indices: Iterable[int], axis: int) -> SymbolicKernel:
        return self._take(tuple(indices), axis)

    @lru_cache(maxsize=None)
    def _take(self, indices: tuple[int, ...], axis: int) -> SymbolicKernel:
        return type(self)(np.take(self._values, indices, axis=axis))

    def trace(self, axis1: int, axis2: int) -> SymbolicKernel:
        return type(self)(np.trace(self._values, axis1=axis1, axis2=axis2))

    def expand_dims(self, axis: int) -> SymbolicKernel:
        return type(self)(np.expand_dims(self._values, axis))

    def tensordot(self, other: SymbolicKernel, axes: tuple) -> SymbolicKernel:
        return type(self)(np.tensordot(self._values, other._values, axes=axes))

    def halved(self) -> SymbolicKernel:
        """Dividing by two uses exact floor division whenever all integer entries are even,
        preserving integer dtype on symmetrized basis products."""
        if self._values.dtype.kind == "i" and not np.any(self._values % 2):
            return type(self)(self._values // 2)
        return type(self)(self._values / 2)

    def reciprocal(self) -> SymbolicKernel:
        return type(self)(1 / self._values)

    @lru_cache(maxsize=None)
    def inverse(self) -> SymbolicKernel:
        """Rounding the floating-point matrix inverse recovers an `int64` inverse whenever integer
        multiplication reproduces the identity matrix exactly."""
        candidate = np.rint(np.linalg.inv(self._values)).astype(np.int64)
        if not np.array_equal(self._values @ candidate, np.eye(len(candidate))):
            raise np.linalg.LinAlgError("the inverse of this kernel has non-integer coefficients")
        return type(self)(candidate)

    def first_last_input(self) -> SymbolicKernel:
        """Dropping the length-one scalar output axis and moving the last input axis first arranges
        a scalar-valued form into test-input rows."""
        return type(self)(np.moveaxis(self._values[0], -1, 0))

    def output_nonzero_indices(self) -> tuple[int, ...]:
        """Output-axis indices with at least one non-zero entry across all input slots identify the
        kernel's active output support."""
        if self.ndim == 0:
            raise ValueError("a symbolic extensor kernel must have an output axis")
        nonzero = self._values != 0
        keep = nonzero if self.ndim == 1 else np.any(nonzero, axis=tuple(range(1, self.ndim)))
        return tuple(int(index) for index in np.flatnonzero(keep))

    def __neg__(self) -> SymbolicKernel:
        return type(self)(-self._values)

    def __add__(self, other: object) -> SymbolicKernel | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"cannot add kernel shapes {self.shape} and {other.shape}")
        return type(self)(self._values + other._values)

    def __sub__(self, other: object) -> SymbolicKernel | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        return self + (-other)

    def __mul__(self, scalar: object) -> SymbolicKernel:
        return type(self)(self._values * _scalar(scalar))

    def __rmul__(self, scalar: object) -> SymbolicKernel:
        return self * scalar

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        return self.shape == other.shape and bool(np.all(self._values == other._values))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"SymbolicKernel(shape={self.shape}, values={self._values.tolist()!r})"
