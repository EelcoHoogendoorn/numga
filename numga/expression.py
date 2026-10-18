"""The products of the algebra, and the versor actions, over values and open type slots.

Each product is one record: the operator factory method that builds it from its operands' types, and
an extension method that evaluates it on values. On two values the method dispatches on their types,
and its staged case builds the operator once per pair of types; each call applies the compiled
operator. An operand that is a type is an open slot: the operator is built from the types and the
values bound into it, leaving a map, and the product's laws stay attached to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from numga.extension import ExtensionMethod
from numga.extensor import operands
from numga.gatype import GAType, UnorientedSubspace
from numga.operator.factory import OperatorFactory

if TYPE_CHECKING:
    from numga.extensor import Extensor
    from numga.subspace import SubSpace

Operand = "SubSpace | GAType | Extensor"


_EXPRESSION_KINDS = frozenset((operands.Kind.VALUE, operands.Kind.TYPE, operands.Kind.SUBSPACE))


def is_expression_operand(value: object) -> bool:
    return operands.kind(value) in _EXPRESSION_KINDS


class Product:
    """A product of two operands: values, open type slots, or one of each."""

    def __init__(self, name: str, build: Callable[[OperatorFactory, GAType, GAType], "Extensor"], doc: str) -> None:
        self.name = name
        self.build = build
        self.values = ExtensionMethod(name)
        self.values.register(lambda left, right: True, staged=True)(self._operator)
        self.values.register(lambda left, right: True, fields=True, staged=True)(self._operator)
        self.__doc__ = doc

    def _operator(self, left: GAType, right: GAType) -> "Extensor":
        """The operator for operands of these types, applied to every pair of values of them."""
        return self.build(left.algebra.operator, left, right)

    def __call__(self, left: Operand, right: Operand) -> "Extensor":
        if operands.kind(left) is operands.Kind.VALUE and operands.kind(right) is operands.Kind.VALUE:
            return self.values(left, right)
        left_type, right_type = _operand_gatype(left), _operand_gatype(right)
        if left_type.algebra is not right_type.algebra:
            raise ValueError(f"{self.name.replace('_', '-')} operands belong to different algebras")
        operation = self.build(left_type.algebra.operator, left_type, right_type)
        bindings = {slot: operand for slot, operand in enumerate((left, right)) if not isinstance(operand, GAType)}
        return operation.bind(bindings) if bindings else operation

    def __repr__(self) -> str:
        return f"Product({self.name!r})"


class VersorAction:
    """The action of a versor on a passenger: values, open type slots, or one of each.

    A typed versor declares distinct left and right input slots over one carrier. A nullary Extensor
    binds the same value into both slots atomically. An Extensor with open inputs is bound into both
    slots too: its inputs appear twice, those of the left copy first, then the passenger's, then those
    of the right copy, and the result keeps the carrier's symmetrized type.
    """

    def __init__(self, name: str, build: Callable[[OperatorFactory, GAType, GAType], "Extensor"], doc: str) -> None:
        self.name = name
        self.build = build
        self.values = ExtensionMethod(name)
        self.values.register(lambda versor, passenger: True, staged=True)(self._action)
        self.values.register(lambda versor, passenger: True, fields=True, staged=True)(self._action)
        self.__doc__ = doc

    def _operator(self, versor: GAType, passenger: GAType) -> "Extensor":
        # The carrier is what the versor produces; the traits of a map say nothing about the values it produces.
        carrier = versor.derive.output if versor.arity else versor
        return self.build(versor.algebra.operator, carrier, passenger)

    def _action(self, versor: GAType, passenger: GAType) -> Callable[["Extensor", "Extensor"], "Extensor"]:
        """The operator for operands of these types, with the versor bound on both sides."""
        operator = self._operator(versor, passenger)
        return lambda versor, passenger: operator(versor, passenger, versor)

    def __call__(self, versor: Operand, passenger: Operand) -> "Extensor":
        if operands.kind(versor) is operands.Kind.VALUE and operands.kind(passenger) is operands.Kind.VALUE:
            return self.values(versor, passenger)
        versor_type, passenger_type = _operand_gatype(versor), _operand_gatype(passenger)
        if versor_type.algebra is not passenger_type.algebra:
            raise ValueError("transform operands belong to different algebras")
        operation = self._operator(versor_type, passenger_type)
        bindings: dict[int, Extensor] = {}
        if not isinstance(versor, GAType):
            bindings[0] = versor
            bindings[2] = versor
        if not isinstance(passenger, GAType):
            bindings[1] = passenger
        return operation.bind(bindings) if bindings else operation

    def __repr__(self) -> str:
        return f"VersorAction({self.name!r})"


geometric_product = Product(
    "geometric_product", OperatorFactory.geometric_product,
    "The geometric product. A nullary GAType is a typed open slot; value Extensors bind their slots, "
    "and the first backend value promotes the whole expression out of the algebra's exact context.",
)
wedge = Product("wedge", OperatorFactory.wedge, "The exterior product.")
scalar_product = Product("scalar_product", OperatorFactory.scalar_product, "The scalar part of the geometric product.")
inner = Product("inner", OperatorFactory.inner, "The inner product, the grade |r - s| part of the product.")
bivector_product = Product("bivector_product", OperatorFactory.bivector_product, "The bivector part of the geometric product.")
trivector_product = Product("trivector_product", OperatorFactory.trivector_product, "The trivector part of the geometric product.")
commutator = Product("commutator", OperatorFactory.commutator, "The geometric-product commutator.")
anticommutator = Product("anticommutator", OperatorFactory.anticommutator, "The geometric-product anticommutator.")
left_contraction = Product("left_contraction", OperatorFactory.left_contraction, "The left contraction, the grade s - r part of the product.")
right_contraction = Product("right_contraction", OperatorFactory.right_contraction, "The right contraction, the grade r - s part of the product.")
left_interior = Product(
    "left_interior", OperatorFactory.left_interior, "The left interior product: the left complement of left, anti-wedged with right.",
)
right_interior = Product(
    "right_interior", OperatorFactory.right_interior, "The right interior product: left, anti-wedged with the right complement of right.",
)
regressive = Product("regressive", OperatorFactory.regressive, "The right-Hodge regressive product.")

transform = VersorAction(
    "transform", OperatorFactory.transform,
    "The action of a versor on a passenger, `versor >> passenger`: the composition of the reflections "
    "in the versor's vector factors.",
)
reverse_transform = VersorAction(
    "reverse_transform", OperatorFactory.reverse_transform,
    "The action of the versor's reverse, `versor << passenger`: for a unit versor, the inverse of its transform.",
)

PRODUCTS = (
    geometric_product, wedge, scalar_product, inner, bivector_product, trivector_product, commutator,
    anticommutator, left_contraction, right_contraction, left_interior, right_interior, regressive,
)


def _operand_gatype(value: object) -> GAType:
    found = operands.kind(value)
    if found is operands.Kind.SUBSPACE:
        raise UnorientedSubspace()
    if found is operands.Kind.TYPE:
        if value.arity:
            raise ValueError(
                "only a nullary GAType can be used as an expression hole"
            )
        return value
    if found is operands.Kind.VALUE:
        return value.gatype
    raise TypeError(
        "an expression operand must be a SubSpace, GAType, or Extensor; "
        f"got {type(value).__name__}"
    )
