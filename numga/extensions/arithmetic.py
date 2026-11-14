"""Sums, differences, quotients, and integer powers dispatch on operand `GAType` metadata after
operator syntax promotes Python numbers, backend arrays, and bare `GAType` identities to extensors.

Addition and subtraction require matching arities and matching orientations on every structural axis,
zero-embedding both operands into the union of their structural subspaces. For dense operands, the
output `GAType` and structural axis-embedding plans are staged once per pair of input types, leaving
only backend array combination per call; field operands align site axes as described in
`docs/fields.md`.

Division right-multiplies by the divisor's algebraic `inverse()`, and integer exponentiation
evaluates repeated geometric products by squaring, inverting negative powers and mapping the zeroth
power to the scalar unit `1`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numga.backend.context import binding_context
from numga.binding import AxisTransform
from numga.extensor import Extensor
from numga.extensor.fields import add_fields
from numga.gatype import GAType, rules
from numga.subspace import SubSpace


# --- sums and differences ---------------------------------------------------------------
@Extensor.add.register(lambda left, right: True, fields=True)
def add_with_fields(left: Extensor, right: Extensor) -> Extensor:
    return add_fields(left, right)


@Extensor.add.register(lambda left, right: left.arity == right.arity, staged=True)
def add_blades(left: GAType, right: GAType) -> Callable:
    """Adding two extensors of equal arity zero-embeds each operand into the union of their
    structural subspaces while preserving their shared axis orientations."""
    return _elementwise(left, right, lambda left_kernel, right_kernel: left_kernel + right_kernel)


@Extensor.subtract.register(lambda left, right: True, fields=True)
def subtract_with_fields(left: Extensor, right: Extensor) -> Extensor:
    return add_fields(left, -right)


@Extensor.subtract.register(lambda left, right: left.arity == right.arity, staged=True)
def subtract_blades(left: GAType, right: GAType) -> Callable:
    """Subtracting two extensors of equal arity zero-embeds each operand into the union of their
    structural subspaces while preserving their shared axis orientations."""
    return _elementwise(left, right, lambda left_kernel, right_kernel: left_kernel - right_kernel)


def _elementwise(left: GAType, right: GAType, combine: Callable) -> Callable:
    axes = tuple(left_axis.union(right_axis) for left_axis, right_axis in zip(left.subspaces, right.subspaces))
    # Addition preserves operand orientations and raises when the two operands disagree.
    result = rules.ADD.result((left, right), axes)
    embed_left, embed_right = _embedding(left.subspaces, axes), _embedding(right.subspaces, axes)

    def elementwise(first: Extensor, second: Extensor) -> Extensor:
        context = first._context
        if second._context is not context:
            context = _common_context(first, second)
            first, second = context.lower(first), context.lower(second)
        return Extensor._from_prepared_kernel(context, result, combine(embed_left(context, first._kernel), embed_right(context, second._kernel)))

    return elementwise


def _embedding(source: tuple[SubSpace, ...], target: tuple[SubSpace, ...]) -> Callable:
    transforms = tuple(
        (axis, AxisTransform.plan(from_axis, to_axis))
        for axis, (from_axis, to_axis) in enumerate(zip(source, target)) if from_axis != to_axis
    )
    if not transforms:
        return lambda context, kernel: kernel
    structural_ndim = len(source)

    def embed(context: Any, kernel: Any) -> Any:
        for axis, transform in transforms:
            kernel = context.transform_axis(kernel, structural_ndim, axis, transform)
        return kernel

    return embed


def _common_context(left: Extensor, right: Extensor) -> Any:
    # Exact operands adopt the concrete backend of their partner, and NumPy promotes into JAX.
    return binding_context(left.context, (right.context,))


# --- quotients -------------------------------------------------------------------------
@Extensor.divide.register(lambda dividend, divisor: divisor.is_scalar)
def divide_by_scalar(dividend: Extensor, divisor: Extensor) -> Extensor:
    return dividend * divisor.inverse()


@Extensor.divide.register(lambda dividend, divisor: not dividend.arity and not divisor.arity)
def divide_values(dividend: Extensor, divisor: Extensor) -> Extensor:
    """Dividing two nullary values multiplies `dividend` on the right by `divisor.inverse()`."""
    return dividend * divisor.inverse()


@Extensor.divide.register(lambda dividend, divisor: True, fields=True)
def divide_with_fields(dividend: Extensor, divisor: Extensor) -> Extensor:
    return dividend * divisor.inverse()


# --- integer powers --------------------------------------------------------------------
@Extensor.power.register(lambda value: value.arity == 0)
def power_of_value(value: Extensor, exponent: int) -> Extensor:
    """Raising a nullary value to an integer `exponent` multiplies by repeated squaring, inverting
    negative exponents and returning the scalar `1` at `exponent == 0`."""
    return _power(value, exponent)


@Extensor.power.register(lambda value: True, fields=True)
def power_with_fields(value: Extensor, exponent: int) -> Extensor:
    return _power(value, exponent)


def _power(value: Extensor, exponent: int) -> Extensor:
    if exponent < 0:
        return _power(value.inverse(), -exponent)
    if exponent == 0:
        context = value.context
        return context.multivector.scalar(context.xp.ones(value.shape + (1,)))
    if exponent == 1:
        return value
    half = _power(value, exponent // 2).squared()
    return half * value if exponent % 2 else half
