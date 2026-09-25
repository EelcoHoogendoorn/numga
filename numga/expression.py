"""Binary expressions over structural/type holes and value Extensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numga.gatype import GAType
from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.extensor import Extensor


def is_expression_operand(value: object) -> bool:
    from numga.extensor import Extensor

    return isinstance(value, (SubSpace, GAType, Extensor))


def geometric_product(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate a geometric product according to operand staging.

    A SubSpace implicitly lifts to a no-traits nullary GAType. A nullary GAType
    is the refined form of the same typed hole. Value Extensors bind their
    corresponding slots; the first backend value promotes the whole expression
    out of the algebra's exact context.
    """

    return _binary_expression("geometric_product", left, right)


def wedge(left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor) -> Extensor:
    """Build or evaluate an exterior product at the operands' current stage."""

    return _binary_expression("wedge", left, right)


def scalar_product(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    return _binary_expression("scalar_product", left, right)


def inner(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the inner product, the grade |r - s| part of the product."""

    return _binary_expression("inner", left, right)


def bivector_product(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    return _binary_expression("bivector_product", left, right)


def trivector_product(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    return _binary_expression("trivector_product", left, right)


def commutator(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the geometric-product commutator."""

    return _binary_expression("commutator", left, right)


def left_contraction(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the left contraction, the grade s - r part of the product."""

    return _binary_expression("left_contraction", left, right)


def right_contraction(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the right contraction, the grade r - s part of the product."""

    return _binary_expression("right_contraction", left, right)


def left_interior(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the left interior product: the left complement of left, anti-wedged with right."""

    return _binary_expression("left_interior", left, right)


def right_interior(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the right interior product: left, anti-wedged with the right complement of right."""

    return _binary_expression("right_interior", left, right)


def regressive(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate a right-Hodge regressive product."""

    return _binary_expression("regressive", left, right)


def sandwich(
    sandwicher: SubSpace | GAType | Extensor, passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate a sandwich expression.

    A typed sandwicher declares distinct left and right input slots over one
    carrier. A nullary Extensor binds the same value into both slots atomically.
    An Extensor with open inputs is bound into both slots too: its inputs appear
    twice, those of the left copy first, then the passenger's, then those of the
    right copy, and the result keeps the carrier's symmetrized sandwich type.
    """

    return _sandwich_expression("sandwich", sandwicher, passenger)


def reverse_sandwich(
    sandwicher: SubSpace | GAType | Extensor, passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate ``reverse(sandwicher) * passenger * sandwicher``, one fused operator bound
    as a sandwich is: for a unit versor, the inverse of its sandwich."""

    return _sandwich_expression("reverse_sandwich", sandwicher, passenger)


def _sandwich_expression(
    operation_name: str,
    sandwicher: SubSpace | GAType | Extensor,
    passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    from numga.extensor import Extensor

    sandwicher_type = _operand_gatype(sandwicher)
    passenger_type = _operand_gatype(passenger)
    if sandwicher_type.algebra is not passenger_type.algebra:
        raise ValueError("sandwich operands belong to different algebras")
    if isinstance(sandwicher, Extensor) and sandwicher.arity:
        # The carrier is what the sandwicher produces; the traits of a map say
        # nothing about the values it produces.
        sandwicher_type = sandwicher_type.algebra.gatype(sandwicher.output_subspace)

    operation = getattr(sandwicher_type.algebra.operator, operation_name)(
        sandwicher_type, passenger_type
    )
    bindings: dict[int, Extensor] = {}
    if isinstance(sandwicher, Extensor):
        bindings[0] = sandwicher
        bindings[2] = sandwicher
    if isinstance(passenger, Extensor):
        bindings[1] = passenger
    if len(bindings) == 3:
        return operation(sandwicher, passenger, sandwicher)
    return operation.bind(bindings) if bindings else operation


def _binary_expression(
    operation_name: str, left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    left_type = _operand_gatype(left)
    right_type = _operand_gatype(right)
    if left_type.algebra is not right_type.algebra:
        raise ValueError(
            f"{operation_name.replace('_', '-')} operands belong to different "
            "algebras"
        )

    factory_method = getattr(left_type.algebra.operator, operation_name)
    operation = factory_method(left_type, right_type)
    bindings = {
        slot: operand
        for slot, operand in enumerate((left, right))
        if not isinstance(operand, (SubSpace, GAType))
    }
    if len(bindings) == 2:
        return operation(left, right)
    return operation.bind(bindings) if bindings else operation


def _operand_gatype(value: object) -> GAType:
    from numga.extensor import Extensor

    if isinstance(value, SubSpace):
        return value.algebra.gatype(value)

    if isinstance(value, GAType):
        if value.arity:
            raise ValueError(
                "only a nullary GAType can be used as an expression hole"
            )
        return value
    if isinstance(value, Extensor):
        return value.gatype
    raise TypeError(
        "an expression operand must be a SubSpace, GAType, or Extensor; "
        f"got {type(value).__name__}"
    )
