"""Binary products and versor actions in `numga` accept runtime values (`arity == 0` extensors),
positive-arity maps and forms, and nullary `GAType` placeholders through one uniform call
interface. Because a nullary `GAType` stands for the identity map on its space, writing
`mv.x ^ Vector` or `Rotor >> Point` partially binds the exact product or sandwich tensor from
`OperatorFactory` and returns a typed linear or multilinear `Extensor` with its algebraic traits
attached.

Each `Product` pairs an `OperatorFactory` builder with a staged `ExtensionMethod` for pure value
evaluation. When both operands are runtime values, the staged dispatcher compiles the exact
operator once per pair of `GAType` signatures (and separately for field operands) and applies that
compiled contraction on every subsequent call. When either operand is a `GAType` or a
positive-arity `Extensor`, `Product.__call__` builds the exact operator on the operands' types and
binds whichever operands are `Extensor` instances, promoting out of the algebra's exact context as
soon as a backend tensor participates.

`VersorAction` (`transform` for `>>` and `reverse_transform` for `<<`) builds a ternary sandwich
operator with separate left and right versor slots flanking the passenger slot so contraction
remains linear in each axis. Passing a nullary versor binds the identical `Extensor` into axes `0`
and `2` simultaneously, forming an equality group on `BindingPlan` that preserves grade and versor
traits; passing a map as the versor composes it into both outer slots, ordering result inputs as
left-versor inputs, passenger inputs, and right-versor inputs. Products as maps, partial
application, and staged table execution are developed in `docs/extensor_syntax.md`,
`docs/tutorial/extensors/01-products-as-maps.md`, and `docs/internals.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numga.extension import ExtensionMethod
from numga.extensor import operands
from numga.gatype import GAType, UnorientedSubspace
from numga.operator.factory import OperatorFactory

if TYPE_CHECKING:
    from numga.extensor import Extensor


_EXPRESSION_KINDS = frozenset((operands.Kind.VALUE, operands.Kind.TYPE, operands.Kind.SUBSPACE))


def is_expression_operand(operand: object) -> bool:
    """Nullary `GAType` placeholders, `Extensor` instances, and `SubSpace` objects (caught later to
    report `UnorientedSubspace`) enter algebraic product dispatch."""
    return operands.kind(operand) in _EXPRESSION_KINDS


class Product:
    """Binary algebraic products evaluate two values through a staged `ExtensionMethod` and compose
    maps or nullary `GAType` placeholders by binding an exact `OperatorFactory` tensor."""

    def __init__(self, name: str, build: Callable, doc: str) -> None:
        self.name = name
        self.build = build
        self.values = ExtensionMethod(name, takes="two values")
        self.values.register(lambda left, right: True, staged=True)(self._operator)
        self.values.register(lambda left, right: True, fields=True, staged=True)(self._operator)
        self.__doc__ = doc

    def _operator(self, left: GAType, right: GAType) -> Extensor:
        return self.build(left.algebra.operator, left, right)

    def __call__(self, left: Extensor | GAType, right: Extensor | GAType) -> Extensor:
        if operands.kind(left) is operands.Kind.VALUE and operands.kind(right) is operands.Kind.VALUE:
            return self.values(left, right)
        left_type, right_type = _operand_gatype(left), _operand_gatype(right)
        if left_type.algebra is not right_type.algebra:
            raise ValueError(f"{self.name.replace('_', '-')} operands belong to different algebras")
        operation = self.build(left_type.algebra.operator, left_type, right_type)
        bindings = {axis: operand for axis, operand in enumerate((left, right)) if not isinstance(operand, GAType)}
        return operation.bind(bindings) if bindings else operation

    def __repr__(self) -> str:
        return f"Product({self.name!r})"


class VersorAction:
    """Sandwich actions bind a nullary versor into both outer slots of a ternary operator or
    compose map operands across left-versor, passenger, and right-versor inputs."""

    def __init__(self, name: str, build: Callable, doc: str) -> None:
        self.name = name
        self.build = build
        self.values = ExtensionMethod(name, takes="two values")
        self.values.register(lambda versor, passenger: True, staged=True)(self._action)
        self.values.register(lambda versor, passenger: True, fields=True, staged=True)(self._action)
        self.__doc__ = doc

    def _operator(self, versor: GAType, passenger: GAType) -> Extensor:
        # The sandwich acts from the versor's output type; a map's own traits say nothing about the values it returns.
        output_type = versor.derive.output if versor.arity else versor
        return self.build(versor.algebra.operator, output_type, passenger)

    def _action(self, versor: GAType, passenger: GAType) -> Callable:
        operator = self._operator(versor, passenger)
        return lambda versor, passenger: operator(versor, passenger, versor)

    def __call__(self, versor: Extensor | GAType, passenger: Extensor | GAType) -> Extensor:
        if operands.kind(versor) is operands.Kind.VALUE and operands.kind(passenger) is operands.Kind.VALUE:
            return self.values(versor, passenger)
        versor_type, passenger_type = _operand_gatype(versor), _operand_gatype(passenger)
        if versor_type.algebra is not passenger_type.algebra:
            raise ValueError("transform operands belong to different algebras")
        operation = self._operator(versor_type, passenger_type)
        bindings = {}
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
    "Clifford multiplication combines metric contraction and exterior spanning across all grades. "
    "Nullary `GAType` operands act as identity maps, `Extensor` maps compose into the product, "
    "and the first backend operand promotes the expression out of the exact context.",
)
wedge = Product(
    "wedge", OperatorFactory.wedge,
    "Exterior multiplication spans the grade-`r + s` subspace of blades of grades `r` and `s` without consulting the metric.",
)
scalar_product = Product(
    "scalar_product", OperatorFactory.scalar_product,
    "Grade-zero projection of the geometric product retains only scalar terms.",
)
dot = Product(
    "dot", OperatorFactory.dot,
    "Metric pairing across every grade evaluates the scalar part of `left * right.reverse()` on extrinsic operands "
    "and pairs the duals of intrinsic operands.",
)
inner = Product(
    "inner", OperatorFactory.inner,
    "Symmetric grade-difference projection retains the grade-`|r - s|` part of the geometric product of grades `r` and `s`.",
)
bivector_product = Product(
    "bivector_product", OperatorFactory.bivector_product,
    "Grade-two projection of the geometric product retains only bivector terms.",
)
trivector_product = Product(
    "trivector_product", OperatorFactory.trivector_product,
    "Grade-three projection of the geometric product retains only trivector terms.",
)
commutator = Product(
    "commutator", OperatorFactory.commutator,
    "Antisymmetric half-difference `(left * right - right * left) / 2` isolates the non-commuting part of the geometric product.",
)
anticommutator = Product(
    "anticommutator", OperatorFactory.anticommutator,
    "Symmetric half-sum `(left * right + right * left) / 2` isolates the commuting part of the geometric product.",
)
left_contraction = Product(
    "left_contraction", OperatorFactory.left_contraction,
    "Asymmetric contraction onto the right operand retains the grade-`s - r` part of the geometric product of grades `r` and `s`.",
)
right_contraction = Product(
    "right_contraction", OperatorFactory.right_contraction,
    "Asymmetric contraction onto the left operand retains the grade-`r - s` part of the geometric product of grades `r` and `s`.",
)
left_interior = Product(
    "left_interior", OperatorFactory.left_interior,
    "Anti-wedging the metric left complement of `left` with `right` contracts `left` into `right` through the Hodge complement.",
)
right_interior = Product(
    "right_interior", OperatorFactory.right_interior,
    "Anti-wedging `left` with the metric right complement of `right` contracts `right` into `left` through the Hodge complement.",
)
regressive = Product(
    "regressive", OperatorFactory.regressive,
    "Metric-free meet against the intrinsic unit volume extracts the subspace shared by two operands and flips orientation once.",
)

transform = VersorAction(
    "transform", OperatorFactory.transform,
    "Sandwich action `versor >> passenger` composes the reflections in the versor's vector factors, keeping grade "
    "involution and intrinsic reflection signs determined by the versor's parity.",
)
reverse_transform = VersorAction(
    "reverse_transform", OperatorFactory.reverse_transform,
    "Reverse sandwich action `versor << passenger` applies `versor.reverse()` on the left and `versor` on the right, "
    "inverting `transform` for unit versors.",
)

PRODUCTS = (
    geometric_product, wedge, scalar_product, dot, inner, bivector_product, trivector_product, commutator,
    anticommutator, left_contraction, right_contraction, left_interior, right_interior, regressive,
)


def _operand_gatype(operand: object) -> GAType:
    found = operands.kind(operand)
    if found is operands.Kind.SUBSPACE:
        raise UnorientedSubspace()
    if found is operands.Kind.TYPE:
        if operand.arity:
            raise ValueError(
                "only a nullary GAType can be used as an expression hole"
            )
        return operand
    if found is operands.Kind.VALUE:
        return operand.gatype
    raise TypeError(
        "an expression operand must be a GAType or Extensor; "
        f"got {type(operand).__name__}"
    )
