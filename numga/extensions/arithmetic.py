"""Sums, differences, quotients and integer powers, as cases matched on the operands' types.

Operator syntax promotes numbers, arrays and types to Extensors first (`numga.extensor.operands`), so
every case here sees Extensors. The general cases are staged: their type work, the result type and
the zero embedding of each operand's axes into the result's, runs once per pair of types, and what
remains per call is the kernel operation.
"""

from __future__ import annotations

from typing import Any, Callable

from numga.binding import AxisTransform
from numga.extensor import Extensor
from numga.extensor.fields import add_fields
from numga.gatype import GAType, rules
from numga.subspace import SubSpace


# --- sums and differences ---------------------------------------------------------------
@Extensor.add.register(lambda left, right: left.arity != right.arity)
def add_unequal_arity(left: Extensor, right: Extensor) -> Extensor:
    raise TypeError("a sum needs operands of equal arity; a number or array adds only to a nullary Extensor")


@Extensor.add.register(lambda left, right: True, fields=True)
def add_with_fields(left: Extensor, right: Extensor) -> Extensor:
    return add_fields(left, right)


@Extensor.add.register(lambda left, right: True, staged=True)
def add_blades(left: GAType, right: GAType) -> Callable[[Extensor, Extensor], Extensor]:
    """The sum of the operands, each zero-embedded into the union of their axes."""
    return _elementwise(left, right, lambda left_kernel, right_kernel: left_kernel + right_kernel)


@Extensor.subtract.register(lambda left, right: left.arity != right.arity)
def subtract_unequal_arity(left: Extensor, right: Extensor) -> Extensor:
    raise TypeError("a difference needs operands of equal arity; a number or array subtracts only from a nullary Extensor")


@Extensor.subtract.register(lambda left, right: True, fields=True)
def subtract_with_fields(left: Extensor, right: Extensor) -> Extensor:
    return add_fields(left, -right)


@Extensor.subtract.register(lambda left, right: True, staged=True)
def subtract_blades(left: GAType, right: GAType) -> Callable[[Extensor, Extensor], Extensor]:
    """The difference of the operands, each zero-embedded into the union of their axes."""
    return _elementwise(left, right, lambda left_kernel, right_kernel: left_kernel - right_kernel)


def _elementwise(left: GAType, right: GAType, combine: Callable[[Any, Any], Any]) -> Callable[[Extensor, Extensor], Extensor]:
    axes = tuple(left_axis.union(right_axis) for left_axis, right_axis in zip(left.subspaces, right.subspaces))
    # A sum keeps the orientation of its operands, which must agree: the rule raises where they differ.
    result = rules.ADD.result((left, right), axes)
    embed_left, embed_right = _embedding(left.subspaces, axes), _embedding(right.subspaces, axes)

    def elementwise(a: Extensor, b: Extensor) -> Extensor:
        context = a._context
        if b._context is not context:
            context = _common_context(a, b)
            a, b = context.lower(a), context.lower(b)
        return Extensor._from_prepared_kernel(context, result, combine(embed_left(context, a._kernel), embed_right(context, b._kernel)))

    return elementwise


def _embedding(source: tuple[SubSpace, ...], target: tuple[SubSpace, ...]) -> Callable[[Any, Any], Any]:
    """Zero-embed every structural axis of a kernel from the source axes into the target axes."""
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
    if left.context.is_compatible_with(right.context):
        return left.context
    if left.context.is_exact:
        return right.context
    if right.context.is_exact:
        return left.context
    raise ValueError("Extensor contexts are incompatible")


# --- quotients -------------------------------------------------------------------------
@Extensor.divide.register(lambda left, right: right.is_scalar)
def divide_by_scalar(left: Extensor, right: Extensor) -> Extensor:
    return left * right.inverse()


@Extensor.divide.register(lambda left, right: not left.arity and not right.arity)
def divide_values(left: Extensor, right: Extensor) -> Extensor:
    """The geometric product with the divisor's inverse, on the right."""
    return left * right.inverse()


@Extensor.divide.register(lambda left, right: True)
def divide_maps(left: Extensor, right: Extensor) -> Extensor:
    raise TypeError("geometric-product division requires nullary Extensors, or a scalar divisor")


@Extensor.divide.register(lambda left, right: True, fields=True)
def divide_with_fields(left: Extensor, right: Extensor) -> Extensor:
    return left * right.inverse()


# --- integer powers --------------------------------------------------------------------
@Extensor.power.register(lambda value: value.arity > 0)
def power_of_map(value: Extensor, exponent: int) -> Extensor:
    raise TypeError("integer powers are defined for values, not for maps")


@Extensor.power.register(lambda value: True)
def power_of_value(value: Extensor, exponent: int) -> Extensor:
    """Repeated geometric products, by squaring; a negative power is a power of the inverse, and the
    zeroth power is the scalar one."""
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
