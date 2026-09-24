"""Exact, backend-independent symbolic tensor storage."""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral, Rational, Real
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


def _scalar(value: object) -> object:
    if not isinstance(value, Real):
        raise TypeError(f"symbolic kernels scale by real scalars; got {type(value).__name__}")
    return float(value) if isinstance(value, Rational) and not isinstance(value, Integral) else value


class SymbolicKernel:
    """Immutable symbolic tensor used by exact Extensors.

    The operator factory builds its kernels as int8: basis products have unit coefficients.
    Arithmetic follows NumPy's promotion rules, so scaling by a float makes the kernel floating.
    """

    __slots__ = ("_values", "_hash")

    def __init__(self, values: Any, shape: Sequence[int] | None = None) -> None:
        if isinstance(values, SymbolicKernel):
            array = values._values
        else:
            array = np.asarray(values)
            if array.dtype == object:
                array = np.array([_scalar(value) for value in array.flat]).reshape(array.shape)
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
    def zeros(cls, shape: Sequence[int]) -> "SymbolicKernel":
        return cls(np.zeros(tuple(int(size) for size in shape), dtype=np.int8))

    @classmethod
    def identity(cls, size: int) -> "SymbolicKernel":
        return cls(np.eye(size, dtype=np.int8))

    @property
    def values(self) -> np.ndarray:
        """The coefficients, read-only."""
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
        """A writable copy of the coefficients."""
        return self._values.copy()

    @lru_cache(maxsize=None)
    def materialize(self, dtype: DTypeLike = np.float64) -> np.ndarray:
        """Cache immutable host coefficients; never cache traced backend arrays."""

        values = self._values.astype(dtype)
        values.flags.writeable = False
        return values

    def transpose(self, permutation: Sequence[int]) -> "SymbolicKernel":
        return type(self)(self._values.transpose(tuple(int(axis) for axis in permutation)))

    def moveaxis(
        self,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> "SymbolicKernel":
        return type(self)(np.moveaxis(self._values, source, destination))

    def take(self, indices: Iterable[int], axis: int) -> "SymbolicKernel":
        return self._take(tuple(indices), axis)

    @lru_cache(maxsize=None)
    def _take(self, indices: tuple[int, ...], axis: int) -> SymbolicKernel:
        return type(self)(np.take(self._values, indices, axis=axis))

    def trace(self, axis1: int, axis2: int) -> "SymbolicKernel":
        return type(self)(np.trace(self._values, axis1=axis1, axis2=axis2))

    def expand_dims(self, axis: int) -> "SymbolicKernel":
        return type(self)(np.expand_dims(self._values, axis))

    def tensordot(
        self,
        other: "SymbolicKernel",
        axes: tuple[int | Sequence[int], int | Sequence[int]],
    ) -> "SymbolicKernel":
        return type(self)(np.tensordot(self._values, other._values, axes=axes))

    def halved(self) -> "SymbolicKernel":
        """Division by two that keeps an integer kernel integer when its entries are even, as the
        symmetrization of basis products always leaves them."""
        if self._values.dtype.kind == "i" and not np.any(self._values % 2):
            return type(self)(self._values // 2)
        return type(self)(self._values / 2)

    def reciprocal(self) -> "SymbolicKernel":
        return type(self)(1 / self._values)

    def output_nonzero_indices(self) -> tuple[int, ...]:
        if self.ndim == 0:
            raise ValueError("a symbolic extensor kernel must have an output axis")
        nonzero = self._values != 0
        keep = nonzero if self.ndim == 1 else np.any(nonzero, axis=tuple(range(1, self.ndim)))
        return tuple(int(index) for index in np.flatnonzero(keep))

    def __neg__(self) -> "SymbolicKernel":
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

    def __mul__(self, scalar: object) -> "SymbolicKernel":
        return type(self)(self._values * _scalar(scalar))

    def __rmul__(self, scalar: object) -> "SymbolicKernel":
        return self * scalar

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        return self.shape == other.shape and bool(np.all(self._values == other._values))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"SymbolicKernel(shape={self.shape}, values={self._values.tolist()!r})"
