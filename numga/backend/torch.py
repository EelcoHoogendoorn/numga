"""Device-bound PyTorch execution stores coefficients as `torch.Tensor` buffers on a chosen CPU or
accelerator `torch.device`, defaulting to `torch.float32` on `"cpu"` to match PyTorch conventions
while reporting precision as a canonical `np.dtype`. Every contraction, slice, and linear-algebra
primitive operates directly on `torch.Tensor` values so `torch.autograd` tracks gradients across
`Extensor` operations.

Core `numga` executors and extensions call backend operations through a NumPy-shaped `context.xp`
namespace. `TorchNamespace` translates parameter spellings and edge cases (`dim` vs. `axis`,
`permute` vs. `transpose`, `cat` vs. `concatenate`, no-op reductions over `axis=()`, mixed scalar and
tensor `clip` bounds, and `torch.linalg.LinAlgError` translation to `np.linalg.LinAlgError`). Caching
one `TorchNamespace` per `torch.device` lets `ordered_binding` and `compile_sparse_bind` share
compiled contraction plans across contexts on that device.

Exact `SymbolicKernel` tables materialize once per `(kernel, torch_dtype, device)` into device
tensors, avoiding repeated host-to-device uploads across calls. Out-of-place indexed updates
(`functional_set`, `functional_add`) clone the underlying tensor, flattening multi-dimensional
indices into `index_add_` so repeated scatter indices accumulate identically to `np.add.at` and
`jax.Array.at[].add`. PyTorch namespace translation and device caching are summarized in
`docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _numpy_dtype(dtype: object) -> np.dtype:
    resolved = _NUMPY_DTYPES.get(dtype) if isinstance(dtype, torch.dtype) else np.dtype(dtype)
    if resolved not in _TORCH_DTYPES:
        raise TypeError("TorchContext requires a real or complex floating dtype")
    return resolved


def _torch_dtype(dtype: object) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    # Index construction passes integer and boolean NumPy dtypes alongside floating context dtypes.
    if dtype is int:
        return torch.int64
    return torch.from_numpy(np.zeros((), dtype=dtype)).dtype


def _numpy_errors(function: Callable) -> Callable:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except torch.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(str(error)) from error
    return wrapped


class TorchNamespace:
    """Exposing a NumPy-compatible array and `linalg` namespace adapts `torch` operations on a
    single `device` to `numga`'s executor interface."""

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
        # NumPy leaves the tensor unchanged over axis=(), whereas torch.sum reduces every axis over dim=().
        return value if axis == () else torch.sum(value, dim=axis, keepdim=keepdims)

    def cumsum(self, value: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.cumsum(value, dim=axis)

    def mean(self, value: torch.Tensor, axis: Any = None, keepdims: bool = False) -> torch.Tensor:
        return value if axis == () else torch.mean(value, dim=axis, keepdim=keepdims)

    def argsort(self, value: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.argsort(value, dim=axis)

    def argmax(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.argmax(value, dim=axis)

    def argmin(self, value: torch.Tensor, axis: int | None = None) -> torch.Tensor:
        return torch.argmin(value, dim=axis)

    def clip(self, value: torch.Tensor, minimum: Any, maximum: Any) -> torch.Tensor:
        # Torch requires both bounds to share scalar or tensor form, whereas NumPy accepts mixed bounds.
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
    # Caching one namespace per device shares compiled contraction plans across contexts.
    return TorchNamespace(device)


class TorchContext(Context):
    """Immutable device-bound policy stores coefficients as `torch.Tensor` buffers with dense or
    sparse-exact execution."""

    __slots__ = ("_algebra", "_dtype", "_device")

    def __init__(
        self, algebra: Algebra | str, dtype: object = torch.float32, device: object = "cpu",
        *, execution: Literal["dense", "sparse"] = "dense", units: bool = False,
    ) -> None:
        from numga.algebra import Algebra

        object.__setattr__(self, "_algebra", Algebra(algebra) if isinstance(algebra, str) else algebra)
        object.__setattr__(self, "_dtype", _numpy_dtype(dtype))
        object.__setattr__(self, "_device", torch.device(device))
        super().__init__(execution, units=units)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TorchContext instances are immutable")

    @classmethod
    def from_key(cls, algebra: Algebra, key: tuple[object, ...], *, units: bool = False) -> TorchContext:
        if len(key) != 4 or key[0] != "torch":
            raise ValueError(f"invalid torch Context key {key!r}")
        context = cls(algebra, key[1], key[2], execution=key[3], units=units)
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

    def prepare_kernel(self, coefficients: object) -> torch.Tensor:
        source = coefficients if isinstance(coefficients, torch.Tensor) else torch.as_tensor(np.asarray(coefficients))
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
        return type(self)(self.algebra, dtype, self.device, execution=self.execution, units=self.units)

    def _with_units(self, units: bool) -> TorchContext:
        return type(self)(self.algebra, self.dtype, self.device, execution=self.execution, units=units)

    def kernel_dtype(self, kernel: torch.Tensor) -> np.dtype:
        return _NUMPY_DTYPES[kernel.dtype]

    def functional_set(self, kernel: torch.Tensor, index: object, value: object) -> torch.Tensor:
        result = kernel.clone()
        result[index] = torch.as_tensor(value, dtype=kernel.dtype, device=kernel.device)
        return result

    def functional_add(self, kernel: torch.Tensor, index: object, value: object) -> torch.Tensor:
        # Indexing a flat position grid preserves repeated index entries so index_add_ accumulates
        # duplicates identically to np.add.at and jax.Array.at[].add.
        positions = torch.arange(kernel.numel(), device=kernel.device).reshape(kernel.shape)[index]
        values = torch.as_tensor(value, dtype=kernel.dtype, device=kernel.device).expand(positions.shape)
        result = kernel.contiguous().clone()
        result.view(-1).index_add_(0, positions.reshape(-1), values.reshape(-1))
        return result

    def __repr__(self) -> str:
        return (f"TorchContext(algebra={self.algebra!r}, dtype={self.dtype}, "
                f"device={str(self.device)!r}, execution={self.execution!r})")
