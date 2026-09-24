"""PyTorch policy for the shared concrete Extensor runtime.

The library calls its array backend through a NumPy-shaped namespace (``context.xp``). Torch
spells a number of those functions differently (``dim`` for ``axis``, ``permute`` for
``transpose``, ``cat`` for ``concatenate``); ``TorchNamespace`` is that translation, and nothing
more. Coefficients are torch tensors on the context's device, so autograd flows through every
Extensor operation.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache, wraps
from numbers import Rational
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from .context import Context

if TYPE_CHECKING:
    from numga.algebra import Algebra
    from numga.operator.kernel import SymbolicKernel


_TORCH_DTYPES = {
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
    np.dtype(np.complex64): torch.complex64,
    np.dtype(np.complex128): torch.complex128,
}
_NUMPY_DTYPES = {value: key for key, value in _TORCH_DTYPES.items()}


def _numpy_dtype(value: object) -> np.dtype:
    dtype = _NUMPY_DTYPES.get(value) if isinstance(value, torch.dtype) else np.dtype(value)
    if dtype not in _TORCH_DTYPES:
        raise TypeError("TorchContext requires a real or complex floating dtype")
    return dtype


def _torch_dtype(value: object) -> torch.dtype | None:
    """Translate a NumPy dtype argument; integer and boolean index dtypes included."""
    if value is None or isinstance(value, torch.dtype):
        return value
    if value is int:
        return torch.int64
    return torch.from_numpy(np.zeros((), dtype=value)).dtype


def _numpy_errors(function: Any) -> Any:
    """Raise NumPy's LinAlgError where torch raises its own, as for a singular matrix."""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except torch.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(str(error)) from error
    return wrapped


class TorchNamespace:
    """The NumPy-shaped array namespace the library executes through, over torch on one device."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.linalg = SimpleNamespace(
            inv=_numpy_errors(torch.linalg.inv),
            det=torch.linalg.det,
            solve=_numpy_errors(torch.linalg.solve),
            cholesky=_numpy_errors(torch.linalg.cholesky),
            eig=_numpy_errors(torch.linalg.eig),
            eigvals=_numpy_errors(torch.linalg.eigvals),
            eigh=_numpy_errors(torch.linalg.eigh),
            eigvalsh=_numpy_errors(torch.linalg.eigvalsh),
            pinv=_numpy_errors(lambda matrix, rcond=None: torch.linalg.pinv(matrix, rtol=rcond)),
            svd=_numpy_errors(self._svd),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(torch, name)

    def __repr__(self) -> str:
        return f"TorchNamespace({self.device})"

    @staticmethod
    def _svd(matrix: torch.Tensor, full_matrices: bool = True, compute_uv: bool = True) -> Any:
        if not compute_uv:
            return torch.linalg.svdvals(matrix)
        return torch.linalg.svd(matrix, full_matrices=full_matrices)

    def asarray(self, value: Any, dtype: object = None) -> torch.Tensor:
        return torch.as_tensor(value, dtype=_torch_dtype(dtype), device=self.device)

    def broadcast_to(self, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.broadcast_to(torch.as_tensor(value, device=self.device), shape)

    def zeros(self, shape: tuple[int, ...], dtype: object = None) -> torch.Tensor:
        return torch.zeros(shape, dtype=_torch_dtype(dtype), device=self.device)

    def reshape(self, value: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(torch.as_tensor(value, device=self.device), shape)

    def transpose(self, value: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
        return value.permute(axes)

    def moveaxis(self, value: torch.Tensor, source: Any, destination: Any) -> torch.Tensor:
        return torch.movedim(value, source, destination)

    def expand_dims(self, value: torch.Tensor, axis: int) -> torch.Tensor:
        return value.unsqueeze(axis)

    def trace(self, value: torch.Tensor, axis1: int = 0, axis2: int = 1) -> torch.Tensor:
        return torch.diagonal(value, dim1=axis1, dim2=axis2).sum(-1)

    def take(self, value: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.index_select(value, axis, torch.as_tensor(indices, device=value.device))

    def tensordot(self, left: torch.Tensor, right: torch.Tensor, axes: Any) -> torch.Tensor:
        if not isinstance(axes, int):
            axes = tuple((axis,) if isinstance(axis, int) else tuple(axis) for axis in axes)
        return torch.tensordot(left, right, dims=axes)

    def einsum(self, expression: str, *operands: torch.Tensor, optimize: bool = False) -> torch.Tensor:
        return torch.einsum(expression, *operands)

    def stack(self, values: Any, axis: int = 0) -> torch.Tensor:
        return torch.stack(tuple(values), dim=axis)

    def concatenate(self, values: Any, axis: int = 0) -> torch.Tensor:
        return torch.cat(tuple(values), dim=axis)

    def sum(self, value: torch.Tensor, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        # NumPy reduces nothing over axis=(); torch reduces everything over dim=().
        return value if axis == () else torch.sum(value, dim=axis, keepdim=keepdims)

    def mean(self, value: torch.Tensor, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return value if axis == () else torch.mean(value, dim=axis, keepdim=keepdims)

    def argsort(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.argsort(value, dim=axis)

    def argmax(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.argmax(value, dim=axis)

    def argmin(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.argmin(value, dim=axis)

    def clip(self, value: torch.Tensor, minimum: Any, maximum: Any) -> torch.Tensor:
        # Torch takes both bounds as numbers or both as tensors; NumPy takes any mix.
        bound = lambda limit: None if limit is None else self.asarray(limit, value.dtype)
        return torch.clip(value, bound(minimum), bound(maximum))

    def repeat(self, value: torch.Tensor, repeats: Any, axis: int) -> torch.Tensor:
        return torch.repeat_interleave(value, repeats, dim=axis)


@lru_cache(maxsize=None)
def _materialize(kernel: SymbolicKernel, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Exact kernels are applied on every call; each is uploaded once per dtype and device.
    return torch.tensor(kernel.values, dtype=dtype, device=device)


@lru_cache(maxsize=None)
def _namespace(device: torch.device) -> TorchNamespace:
    # One namespace per device, so plans compiled against it are shared across contexts.
    return TorchNamespace(device)


class TorchContext(Context):
    """Torch storage on one device with dense or sparse-exact execution, bound to one algebra."""

    __slots__ = ("_algebra", "_dtype", "_device")

    def __init__(
        self, algebra: Algebra | str, dtype: object = torch.float32, device: object = "cpu",
        *, execution: Literal["dense", "sparse"] = "dense",
    ) -> None:
        from numga.algebra import Algebra

        object.__setattr__(self, "_algebra", Algebra(algebra) if isinstance(algebra, str) else algebra)
        object.__setattr__(self, "_dtype", _numpy_dtype(dtype))
        object.__setattr__(self, "_device", torch.device(device))
        super().__init__(execution)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TorchContext instances are immutable")

    @classmethod
    def from_key(cls, algebra: Algebra, key: tuple[object, ...]) -> TorchContext:
        if len(key) != 4 or key[0] != "torch":
            raise ValueError(f"invalid torch Context key {key!r}")
        context = cls(algebra, key[1], key[2], execution=key[3])
        if context.key != key:
            raise ValueError(f"torch Context key is not canonical: {key!r}")
        return context

    @property
    def algebra(self) -> Algebra:
        return self._algebra

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def torch_dtype(self) -> torch.dtype:
        return _TORCH_DTYPES[self._dtype]

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def key(self) -> tuple[object, ...]:
        return ("torch", self.dtype.str, str(self.device), self.execution)

    @property
    def xp(self) -> TorchNamespace:
        return _namespace(self.device)

    def prepare_kernel(self, value: object) -> torch.Tensor:
        source = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
        if source.dtype.is_complex and not self.torch_dtype.is_complex:
            raise TypeError(
                f"cannot represent coefficients of dtype {source.dtype} in "
                f"Context dtype {self.dtype} without changing numeric kind"
            )
        return source.to(dtype=self.torch_dtype, device=self.device, copy=True)

    def prepare_scalar(self, scalar: object) -> torch.Tensor:
        if isinstance(scalar, Rational):
            scalar = float(Fraction(scalar))
        if np.ndim(scalar) if not isinstance(scalar, torch.Tensor) else scalar.ndim:
            raise TypeError(f"expected a scalar, got shape {tuple(np.shape(scalar))}")
        return self.prepare_kernel(scalar)

    def scalar_kernel(self, scalar: Any) -> torch.Tensor:
        return torch.as_tensor(scalar, dtype=self.torch_dtype, device=self.device).reshape(1)

    def materialize(self, kernel: SymbolicKernel) -> torch.Tensor:
        return _materialize(kernel, self.torch_dtype, self.device)

    def with_dtype(self, dtype: object) -> TorchContext:
        return type(self)(self.algebra, dtype, self.device, execution=self.execution)

    def kernel_dtype(self, kernel: torch.Tensor) -> np.dtype:
        return _NUMPY_DTYPES[kernel.dtype]

    def functional_set(self, kernel: torch.Tensor, index: object, value: object) -> torch.Tensor:
        result = kernel.clone()
        result[index] = torch.as_tensor(value, dtype=kernel.dtype, device=kernel.device)
        return result

    def __repr__(self) -> str:
        return (f"TorchContext(algebra={self.algebra!r}, dtype={self.dtype}, "
                f"device={str(self.device)!r}, execution={self.execution!r})")
