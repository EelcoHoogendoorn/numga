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


def anticommutator(
    left: SubSpace | GAType | Extensor, right: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the geometric-product anticommutator."""

    return _binary_expression("anticommutator", left, right)


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


def transform(
    versor: SubSpace | GAType | Extensor, passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the action of a versor on a passenger, `versor >> passenger`: the composition
    of the reflections in the versor's vector factors.

    A typed versor declares distinct left and right input slots over one carrier. A nullary Extensor
    binds the same value into both slots atomically. An Extensor with open inputs is bound into both
    slots too: its inputs appear twice, those of the left copy first, then the passenger's, then those
    of the right copy, and the result keeps the carrier's symmetrized type.
    """

    return _transform_expression("transform", versor, passenger)


def reverse_transform(
    versor: SubSpace | GAType | Extensor, passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    """Build or evaluate the action of the versor's reverse, `versor << passenger`, bound as a
    transform is: for a unit versor, the inverse of its transform."""

    return _transform_expression("reverse_transform", versor, passenger)


def _transform_expression(
    operation_name: str,
    versor: SubSpace | GAType | Extensor,
    passenger: SubSpace | GAType | Extensor,
) -> Extensor:
    from numga.extensor import Extensor

    versor_type = _operand_gatype(versor)
    passenger_type = _operand_gatype(passenger)
    if versor_type.algebra is not passenger_type.algebra:
        raise ValueError("transform operands belong to different algebras")
    if isinstance(versor, Extensor) and versor.arity:
        # The carrier is what the versor produces; the traits of a map say
        # nothing about the values it produces.
        versor_type = versor_type.derive.output

    operation = getattr(versor_type.algebra.operator, operation_name)(
        versor_type, passenger_type
    )
    bindings: dict[int, Extensor] = {}
    if isinstance(versor, Extensor):
        bindings[0] = versor
        bindings[2] = versor
    if isinstance(passenger, Extensor):
        bindings[1] = passenger
    if len(bindings) == 3:
        return operation(versor, passenger, versor)
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
        raise TypeError("a subspace has no orientation: build the type from a named constructor, or as "
            "ga.gatype(subspace, orientation=...)")
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
