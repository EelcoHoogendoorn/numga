"""NumPy policy for the shared concrete Extensor runtime."""

from __future__ import annotations

from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Literal

import numpy as np

from .context import Context

if TYPE_CHECKING:
    from types import ModuleType

    from numga.algebra import Algebra


def _dtype(value: object) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.kind not in "fc":
        raise TypeError("NumpyContext currently requires a real or complex floating dtype")
    return dtype


class NumpyContext(Context):
    """NumPy storage with dense or sparse-exact execution, bound to one algebra."""

    __slots__ = ("_algebra", "_dtype")

    def __init__(
        self, algebra: Algebra | str, dtype: object = np.float64,
        *, execution: Literal["dense", "sparse"] = "dense",
    ) -> None:
        from numga.algebra import Algebra

        object.__setattr__(self, "_algebra", Algebra(algebra) if isinstance(algebra, str) else algebra)
        object.__setattr__(self, "_dtype", _dtype(dtype))
        super().__init__(execution)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("NumpyContext instances are immutable")

    @classmethod
    def from_key(cls, algebra: Algebra, key: tuple[object, ...]) -> NumpyContext:
        if len(key) != 3 or key[0] != "numpy":
            raise ValueError(f"invalid NumPy Context key {key!r}")
        context = cls(algebra, key[1], execution=key[2])
        if context.key != key:
            raise ValueError(f"NumPy Context key is not canonical: {key!r}")
        return context

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def key(self) -> tuple[object, ...]:
        return ("numpy", self.dtype.str, self.execution)

    @property
    def xp(self) -> ModuleType:
        return np

    def prepare_kernel(self, value: object) -> np.ndarray:
        source = np.asarray(value)
        if not np.can_cast(source.dtype, self.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot represent coefficients of dtype {source.dtype} in "
                f"Context dtype {self.dtype} without changing numeric kind"
            )
        return np.array(source, dtype=self.dtype, copy=True)

    def prepare_scalar(self, scalar: object) -> np.generic:
        if isinstance(scalar, Rational):
            scalar = float(Fraction(scalar))
        source = np.asarray(scalar)
        if source.ndim != 0:
            raise TypeError(f"expected a scalar, got shape {source.shape}")
        if not np.can_cast(source.dtype, self.dtype, casting="same_kind"):
            raise TypeError(
                f"cannot represent scalar dtype {source.dtype} in Context dtype "
                f"{self.dtype} without changing numeric kind"
            )
        return self.dtype.type(source)

    def generalized_eig(self, matrix: np.ndarray, metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Broadcast SciPy's generalized eigensolver, including singular pencils."""
        from scipy.linalg import eig

        dtype = np.result_type(self.dtype, np.complex64)
        return np.vectorize(eig, signature="(n,n),(n,n)->(n),(n,n)", otypes=[dtype, dtype])(matrix, metric)

    def generalized_eigh(self, matrix: np.ndarray, metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Broadcast SciPy's Hermitian, positive-definite-metric eigensolver."""
        from scipy.linalg import eigh

        return np.vectorize(eigh, signature="(n,n),(n,n)->(n),(n,n)", otypes=[self.dtype, self.dtype])(matrix, metric)

    def generalized_eigvals(self, matrix: np.ndarray, metric: np.ndarray) -> np.ndarray:
        from scipy.linalg import eigvals

        dtype = np.result_type(self.dtype, np.complex64)
        return np.vectorize(eigvals, signature="(n,n),(n,n)->(n)", otypes=[dtype])(matrix, metric)

    def generalized_eigvalsh(self, matrix: np.ndarray, metric: np.ndarray) -> np.ndarray:
        from scipy.linalg import eigvalsh

        return np.vectorize(eigvalsh, signature="(n,n),(n,n)->(n)", otypes=[self.dtype])(matrix, metric)

    def functional_set(
        self, kernel: np.ndarray, index: object, value: object
    ) -> np.ndarray:
        result = np.array(kernel, copy=True)
        result[index] = value
        return result

    def functional_add(
        self, kernel: np.ndarray, index: object, value: object
    ) -> np.ndarray:
        result = np.array(kernel, copy=True)
        np.add.at(result, index, value)
        return result

    def __repr__(self) -> str:
        return f"NumpyContext(algebra={self.algebra!r}, dtype={self.dtype}, execution={self.execution!r})"
