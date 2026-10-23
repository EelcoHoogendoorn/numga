"""The other operand of an operator, sorted by its Python class into the kind of operand it is.

Operator syntax takes more than Extensors: numbers, arrays, and types. Each kind of operand is
promoted to an Extensor one way, here and nowhere else, and each family of operators reads its
promotions from a table, so the policy is data: a linear operator reads a type as its identity map,
and refuses an operand it does not know by returning NotImplemented.
"""

from __future__ import annotations

from enum import Enum
from numbers import Number
from typing import TYPE_CHECKING, Any, Callable

import numga
from numga.gatype import GAType, UnorientedSubspace
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.backend.context import Context

    from .extensor import Extensor


class Kind(Enum):
    VALUE = "value"
    TYPE = "type"
    NUMBER = "number"
    ARRAY = "array"
    SUBSPACE = "subspace"
    FOREIGN = "foreign"


_KINDS: dict[type, Kind] = {}


def kind(value: object) -> Kind:
    """The kind of an operand, classified once per Python class."""
    try:
        return _KINDS[value.__class__]
    except KeyError:
        return _classify(value)


def _classify(value: object) -> Kind:
    cls = value.__class__
    if issubclass(cls, numga.extensor.Extensor):
        found = Kind.VALUE
    elif issubclass(cls, GAType):
        found = Kind.TYPE
    elif issubclass(cls, SubSpace):
        found = Kind.SUBSPACE
    elif issubclass(cls, Number):
        found = Kind.NUMBER
    elif hasattr(cls, "shape") and hasattr(cls, "dtype"):
        # An array of any rank, 0-d arrays and traced scalars included, is a batch of scalars.
        found = Kind.ARRAY
    else:
        found = Kind.FOREIGN
    _KINDS[cls] = found
    return found


# --- promotions ------------------------------------------------------------------------
def scalar(context: Context, number: Number) -> Extensor:
    """A plain number as a scalar. It is an ordinary number, unchanged by any mirror: extrinsic."""
    return numga.extensor.Extensor._from_prepared_kernel(
        context, context.algebra.gatype.scalar(), context.scalar_kernel(context.prepare_scalar(number)),
    )


def batch(context: Context, values: Any) -> Extensor:
    """A raw array as a batch of scalars of exactly the array's shape: the structural axis is always
    appended, so a trailing axis of length one is batch, never structural. A JAX array, traced or not, is a batch of
    JAX scalars whatever it meets, so that NumPy operands are carried into JAX with it."""
    if context.key[0] == "numpy" and type(values).__module__.split(".")[0] in ("jax", "jaxlib"):
        from numga.backend.jax import JaxContext
        context = JaxContext(context.algebra)
    return context.multivector.scalar(values[..., None])


def identity(gatype: GAType) -> Extensor:
    """The identity map of a nullary type."""
    return gatype._identity()


def no_orientation(*_: object) -> None:
    raise UnorientedSubspace()


# --- operator families -----------------------------------------------------------------
Promotion = Callable[["Context", Any], "Extensor | None"]

# Linear operators: a number is a scalar, an array a batch of scalars, a type its identity map.
LINEAR: dict[Kind, Promotion] = {
    Kind.VALUE: lambda context, value: value,
    Kind.TYPE: lambda context, gatype: identity(gatype),
    Kind.NUMBER: scalar,
    Kind.ARRAY: batch,
    Kind.SUBSPACE: no_orientation,
    Kind.FOREIGN: lambda context, value: None,
}

# Division: a type has no inverse to divide by.
DIVISOR: dict[Kind, Promotion] = LINEAR | {Kind.TYPE: lambda context, gatype: None}


def compared(context: Context, value: object) -> Extensor:
    """The operand of a comparison: an Extensor, or a number or array read as a batch of scalars."""
    return value if kind(value) is Kind.VALUE else batch(context, context.xp.asarray(value))


def linear(context: Context, value: object) -> Extensor | None:
    """The operand of a linear operator as an Extensor, or None for an operand numga does not know."""
    return LINEAR[kind(value)](context, value)


def divisor(context: Context, value: object) -> Extensor | None:
    """The operand of a division as an Extensor, or None for an operand numga does not divide by."""
    return DIVISOR[kind(value)](context, value)
