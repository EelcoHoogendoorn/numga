"""Binary and comparison operators on `Extensor` accept Python numbers, backend arrays, and
`GAType` descriptors alongside `Extensor` instances. `kind` classifies each operand once per Python
class into `Kind` and caches that lookup, keeping operand-class dispatch out of individual operator
methods. Unoriented `SubSpace` instances classify separately so arithmetic operators raise
`UnorientedSubspace` rather than guessing an orientation, while foreign types produce `None` so
dunder operators return `NotImplemented` and allow reflected operators such as `Sparse.__rmul__` to
run.

Each promotion maps its operand category to a canonical `Extensor`. Plain Python numbers become
extrinsic `Scalar` values, backend arrays of any rank append a length-1 blade axis to form `Scalar`
batches of the array's exact shape (promoting an exact context to the array's backend and switching
a NumPy context to `JaxContext` when paired with a JAX array or tracer), and a nullary `GAType`
promotes to its identity linear map under `LINEAR` operators while remaining invalid as a `DIVISOR`.
"""

from __future__ import annotations

from enum import Enum
from numbers import Number
from typing import TYPE_CHECKING, Any

import numpy as np

import numga
from numga.gatype import GAType, UnorientedSubspace
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.backend.context import Context

    from .extensor import Extensor


class Kind(Enum):
    """Operand categories determine how operator families promote non-`Extensor` values."""

    VALUE = "value"
    TYPE = "type"
    NUMBER = "number"
    ARRAY = "array"
    SUBSPACE = "subspace"
    FOREIGN = "foreign"


_KINDS = {}


def kind(operand: object) -> Kind:
    """Operand classification caches one `Kind` entry per Python class."""
    try:
        return _KINDS[operand.__class__]
    except KeyError:
        return _classify(operand)


def _classify(operand: object) -> Kind:
    cls = operand.__class__
    if issubclass(cls, numga.extensor.Extensor):
        found = Kind.VALUE
    elif issubclass(cls, GAType):
        found = Kind.TYPE
    elif issubclass(cls, SubSpace):
        found = Kind.SUBSPACE
    elif issubclass(cls, Number):
        found = Kind.NUMBER
    elif hasattr(cls, "shape") and hasattr(cls, "dtype"):
        # Zero-dimensional arrays and traced scalars carry shape and dtype, forming scalar batches.
        found = Kind.ARRAY
    else:
        found = Kind.FOREIGN
    _KINDS[cls] = found
    return found


# --- promotions ------------------------------------------------------------------------
def scalar(context: Context, number: Number) -> Extensor:
    """Plain Python numbers promote to extrinsic `Scalar` values."""
    return numga.extensor.Extensor._from_prepared_kernel(
        context, context.algebra.gatype.scalar(), context.scalar_kernel(context.prepare_scalar(number)),
    )


def batch(context: Context, array: Any) -> Extensor:
    """Appending a length-one blade axis wraps a backend array as a `Scalar` batch of the array's
    shape, promoting an exact context to the array's backend and switching a NumPy context to
    `JaxContext` when paired with a JAX array."""
    module = type(array).__module__.split(".")[0]
    if context.is_exact:
        if module in ("jax", "jaxlib"):
            from numga.backend.jax import JaxContext
            dtype = np.dtype(array.dtype)
            context = JaxContext(context.algebra, dtype if dtype.kind in "fc" else np.float64)
        elif module == "torch":
            import torch
            from numga.backend.torch import TorchContext
            dtype = array.dtype if array.is_floating_point() or array.is_complex() else torch.get_default_dtype()
            context = TorchContext(context.algebra, dtype, array.device)
        else:
            from numga.backend.numpy import NumpyContext
            dtype = np.dtype(array.dtype)
            context = NumpyContext(context.algebra, dtype if dtype.kind in "fc" else np.float64)
    elif context.key[0] == "numpy" and module in ("jax", "jaxlib"):
        from numga.backend.jax import JaxContext
        context = JaxContext(context.algebra)
    return context.multivector.scalar(array[..., None])


def identity(gatype: GAType) -> Extensor:
    """Nullary `GAType` descriptors promote to their identity linear maps."""
    return gatype._identity()


def no_orientation(*_: object) -> None:
    raise UnorientedSubspace()


# --- operator families -----------------------------------------------------------------
# Linear operators promote numbers to scalars, arrays to scalar batches, and types to identity maps.
LINEAR = {
    Kind.VALUE: lambda context, operand: operand,
    Kind.TYPE: lambda context, gatype: identity(gatype),
    Kind.NUMBER: scalar,
    Kind.ARRAY: batch,
    Kind.SUBSPACE: no_orientation,
    Kind.FOREIGN: lambda context, operand: None,
}

# Division rejects GAType operands because a type descriptor carries no inverse.
DIVISOR = LINEAR | {Kind.TYPE: lambda context, gatype: None}


def compared(context: Context, operand: object) -> Extensor:
    """Comparison operators leave `Extensor` operands unchanged and promote numbers or arrays to `Scalar` batches."""
    return operand if kind(operand) is Kind.VALUE else batch(context, context.xp.asarray(operand))


def linear(context: Context, operand: object) -> Extensor | None:
    """Linear operators promote numbers, arrays, and nullary `GAType` descriptors to `Extensor` instances and return `None` for foreign operands."""
    return LINEAR[kind(operand)](context, operand)


def divisor(context: Context, operand: object) -> Extensor | None:
    """Division promotes numbers and arrays like linear operators while rejecting `GAType` divisors with `None`."""
    return DIVISOR[kind(operand)](context, operand)
