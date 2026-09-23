"""Exact, backend-independent symbolic tensor storage."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from numbers import Integral, Rational
from types import NotImplementedType
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


ExactScalar = Fraction


def _as_fraction(value: object) -> Fraction:
    """Normalize one supported exact scalar.

    Floating-point values are intentionally rejected.  Accepting them here
    would reintroduce the epsilon-based symbolic semantics that the rewrite is
    meant to remove.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, Rational):
        return Fraction(value.numerator, value.denominator)
    raise TypeError(
        "symbolic coefficients must be exact integers or rational numbers; "
        f"got {type(value).__name__}"
    )


class SymbolicKernel:
    """Immutable exact-rational tensor used by exact Extensors.

    The tuple-backed representation is deliberately simple.  It is a valid
    correctness implementation that can remain as a reference if a compact
    numerator/denominator representation is added later.
    """

    __slots__ = ("_shape", "_values", "_hash")

    def __init__(
        self, values: Any, shape: Sequence[int] | None = None,
    ) -> None:
        if isinstance(values, SymbolicKernel):
            inferred_shape = values.shape
            flat = values._values
        else:
            array = np.asarray(values, dtype=object)
            inferred_shape = tuple(int(size) for size in array.shape)
            flat = tuple(_as_fraction(value) for value in array.flat)

        normalized_shape = (
            inferred_shape if shape is None else tuple(int(size) for size in shape)
        )
        if any(size < 0 for size in normalized_shape):
            raise ValueError("symbolic kernel dimensions must be non-negative")
        if int(np.prod(normalized_shape, dtype=int)) != len(flat):
            raise ValueError(
                f"shape {normalized_shape} does not contain {len(flat)} values"
            )

        object.__setattr__(self, "_shape", normalized_shape)
        object.__setattr__(self, "_values", flat)
        object.__setattr__(self, "_hash", hash((self._shape, self._values)))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SymbolicKernel instances are immutable")

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> "SymbolicKernel":
        normalized_shape = tuple(int(size) for size in shape)
        count = int(np.prod(normalized_shape, dtype=int))
        return cls((Fraction(0),) * count, normalized_shape)

    @classmethod
    def identity(cls, size: int) -> "SymbolicKernel":
        values = np.zeros((size, size), dtype=object)
        for index in range(size):
            values[index, index] = Fraction(1)
        return cls(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return len(self._values)

    def to_object_array(self) -> np.ndarray:
        """Return a fresh object array containing immutable Fractions."""

        return np.asarray(self._values, dtype=object).reshape(self._shape).copy()

    @lru_cache(maxsize=None)
    def materialize(self, dtype: DTypeLike = np.float64) -> np.ndarray:
        """Cache immutable host coefficients; never cache traced backend arrays."""

        if np.dtype(dtype) == np.dtype(object):
            values = self.to_object_array()
        else:
            values = np.fromiter((float(value) for value in self._values), dtype=dtype)
        values.flags.writeable = False
        result = values.reshape(self._shape)
        return result

    def transpose(self, permutation: Sequence[int]) -> "SymbolicKernel":
        permutation = tuple(int(axis) for axis in permutation)
        return type(self)(self.to_object_array().transpose(permutation))

    def moveaxis(
        self,
        source: int | Sequence[int],
        destination: int | Sequence[int],
    ) -> "SymbolicKernel":
        return type(self)(np.moveaxis(self.to_object_array(), source, destination))

    def take(self, indices: Iterable[int], axis: int) -> "SymbolicKernel":
        return self._take(tuple(indices), axis)

    @lru_cache(maxsize=None)
    def _take(self, indices: tuple[int, ...], axis: int) -> SymbolicKernel:
        return type(self)(np.take(self.to_object_array(), indices, axis=axis))

    def tensordot(
        self,
        other: "SymbolicKernel",
        axes: tuple[int | Sequence[int], int | Sequence[int]],
    ) -> "SymbolicKernel":
        return type(self)(
            np.tensordot(
                self.to_object_array(),
                other.to_object_array(),
                axes=axes,
            )
        )

    def output_nonzero_indices(self) -> tuple[int, ...]:
        if self.ndim == 0:
            raise ValueError("a symbolic extensor kernel must have an output axis")
        array = self.to_object_array()
        if self.ndim == 1:
            keep = array != 0
        else:
            keep = np.any(array != 0, axis=tuple(range(1, self.ndim)))
        return tuple(int(index) for index in np.flatnonzero(keep))

    def __neg__(self) -> "SymbolicKernel":
        return type(self)(tuple(-value for value in self._values), self._shape)

    def __add__(self, other: object) -> SymbolicKernel | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"cannot add kernel shapes {self.shape} and {other.shape}")
        return type(self)(
            tuple(left + right for left, right in zip(self._values, other._values)),
            self._shape,
        )

    def __sub__(self, other: object) -> SymbolicKernel | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        return self + (-other)

    def __mul__(self, scalar: object) -> "SymbolicKernel":
        exact = _as_fraction(scalar)
        return type(self)(tuple(value * exact for value in self._values), self._shape)

    def __rmul__(self, scalar: object) -> "SymbolicKernel":
        return self * scalar

    def __truediv__(self, scalar: object) -> "SymbolicKernel":
        exact = _as_fraction(scalar)
        if exact == 0:
            raise ZeroDivisionError("cannot divide a symbolic kernel by zero")
        return type(self)(tuple(value / exact for value in self._values), self._shape)

    def __eq__(self, other: object) -> bool | NotImplementedType:
        if not isinstance(other, SymbolicKernel):
            return NotImplemented
        return self.shape == other.shape and self._values == other._values

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"SymbolicKernel(shape={self.shape}, values={self._values!r})"
