"""Algebraic operations propagate certified `Trait` facts from operand `GAType`s to result
`GAType`s in two stages: symbolic map laws attached when an operator is constructed, and argument
substitution when inputs bind to that operator.

Operator laws (`geometric_product_laws`, `blade_laws`, `transform_laws`, `conjugation_laws`,
`negation_laws`, `transpose_laws`, and `shared_laws`) record conditional `ProductRelation`s,
`ClosureRelation`s (`VersorProduct`, `BladeProduct`), `Sandwich` markers, and `CoefficientOrthogonal`
invariants directly on the operator's `GAType`. Grade transforms whose signs fix every blade on an
operand's output subspace bypass general laws via `unless_identity` and retain all traits of their
operand.

During argument binding, `bind_subspaces` narrows output support before kernel construction whenever
equal certified `Versor` operands occupy both outer slots (`0` and `2`) of a `Sandwich` expression,
restricting the output to the passenger's grades. `bind_traits` then substitutes bound operand
`GAType`s into each `ProductRelation` and `ClosureRelation`, reducing fully bound relations on
nullary results to `ProductFact`, `Versor`, or `Blade` assertions and certifying
`CoefficientOrthogonal` on Euclidean unit-versor actions.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from .gatype import GAType
from .traits import (
    EMPTY_TRAITS,
    Blade,
    BladeProduct,
    CliffordConjugateProduct,
    ClosureRelation,
    CoefficientOrthogonal,
    GradeInvolutionProduct,
    Identity,
    ProductFact,
    ProductRelation,
    ProductResult,
    ReverseProduct,
    ReverseProductNonzero,
    ReverseProductOne,
    Sandwich,
    SelfProduct,
    Trait,
    TraitSet,
    Versor,
    VersorProduct,
)

if TYPE_CHECKING:
    from numga.binding import BindingPlan
    from numga.subspace import SubSpace


def multiply_product_results(left: ProductResult, right: ProductResult) -> ProductResult:
    """Scalar self-product categories multiply without assuming real or positive coefficients."""
    if left is ProductResult.ONE:
        return right
    if right is ProductResult.ONE:
        return left
    if left is ProductResult.ZERO or right is ProductResult.ZERO:
        return ProductResult.ZERO
    if left is ProductResult.NONZERO and right is ProductResult.NONZERO:
        return ProductResult.NONZERO
    return ProductResult.SCALAR


def no_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Operations without certified trait laws attach `EMPTY_TRAITS`."""
    return EMPTY_TRAITS


def geometric_product_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Anti-automorphisms satisfying `T(a * b) == T(b) * T(a)` (`reverse` and `clifford_conjugate`)
    factor the product's self-product across scalar-product operands, and multiplying versors yields
    a versor."""
    return TraitSet((
        ProductRelation(ReverseProduct, ProductResult.ONE, (0, 1)),
        ProductRelation(CliffordConjugateProduct, ProductResult.ONE, (0, 1)),
        VersorProduct(axes=(0, 1)),
    ))


def blade_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Both the exterior product and the regressive product (the exterior product of complements
    read back) carry blades to blades."""
    return TraitSet((BladeProduct((0, 1)),))


def transform_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Trilinear `L * P * ~R` expressions carry `Sandwich` so binding the same versor to both outer
    slots triggers grade preservation and self-product scaling."""
    return TraitSet((Sandwich,))


def negation_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Negation and grade involution preserve quadratic self-products, versors, blades, closure
    relations, and coefficient orthogonality while dropping `Identity` and uncertified user traits."""
    return TraitSet(
        trait for trait in operands[0].effective_traits
        if (
            isinstance(trait, (ProductFact, ProductRelation))
            and trait.self_product in (ReverseProduct, CliffordConjugateProduct, GradeInvolutionProduct)
        )
        or isinstance(trait, ClosureRelation)
        or trait in (Versor, Blade, CoefficientOrthogonal)
    )


def transpose_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Transposing a square coefficient-orthogonal map preserves `CoefficientOrthogonal`."""
    return TraitSet((CoefficientOrthogonal,)) if operands[0].entails(CoefficientOrthogonal) else EMPTY_TRAITS


def coefficient_orthogonality_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Sign changes on output blade coordinates preserve `CoefficientOrthogonal`."""
    return TraitSet((CoefficientOrthogonal,)) if operands[0].entails(CoefficientOrthogonal) else EMPTY_TRAITS


def shared_laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
    """Stacked extensors retain the intersection of their entailed traits; positive-arity maps keep
    traits only when no operand's codomain widens by zero-padding."""
    if operands[0].arity and any(gatype.output_subspace != axes[0] for gatype in operands):
        return EMPTY_TRAITS
    shared = set(operands[0].effective_traits.closure)
    for gatype in operands[1:]:
        shared.intersection_update(gatype.effective_traits.closure)
    return TraitSet(shared)


def conjugation_laws(family: SelfProduct) -> Callable:
    """For `reverse` or `clifford_conjugate`, nonzero scalar self-products commute in finite
    dimensions (`x * T(x) == T(x) * x`), while versors, blades, and coefficient orthogonality
    survive coordinate sign flips."""

    def laws(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
        operand = operands[0]
        inferred = []
        if operand.entails(CoefficientOrthogonal):
            inferred.append(CoefficientOrthogonal)
        for result in (ProductResult.ONE, ProductResult.NONZERO):
            fact = ProductFact(family, result)
            if operand.entails(fact):
                inferred.append(fact)
                break
        inferred.extend(fact for fact in (Versor, Blade) if operand.entails(fact))
        relation = operand.self_products.relation(family)
        if relation is not None and relation.factor in (ProductResult.ONE, ProductResult.NONZERO):
            inferred.append(ProductRelation(family, relation.factor, relation.axes, nonzero_inputs=True))
        inferred.extend(
            relation for relation in (operand.self_products.closure(kind) for kind in _CLOSURES)
            if relation is not None
        )
        return TraitSet(inferred)

    return laws


def unless_identity(transform_name: str, laws: Callable) -> Callable:
    """When a grade transform fixes every blade on the operand's output subspace, all operand traits
    carry through unchanged."""

    def guarded(operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> TraitSet:
        if operands[0].grade_transform_is_identity(transform_name):
            return operands[0].traits
        return laws(operands, axes)

    return guarded


def _transform_versor(plan: BindingPlan) -> GAType | None:
    if not plan.target_gatype.entails(Sandwich):
        return None
    if not any(0 in group and 2 in group for group in plan.equality_groups):
        return None
    operand = next(binding.operand_gatype for binding in plan.bindings if binding.axis == 0)
    # Self-product propagation through a sandwich requires an invertible versor.
    return operand if operand.entails(Versor) and operand.entails(ReverseProductNonzero) else None


def _transform_passenger(plan: BindingPlan) -> SubSpace:
    return next(
        (binding.operand_gatype.output_subspace for binding in plan.bindings if binding.axis == 1),
        plan.target_gatype.input_subspaces[1],
    )


@lru_cache(maxsize=None)
def bind_subspaces(plan: BindingPlan) -> tuple[SubSpace, ...]:
    """Binding the same certified `Versor` to both outer slots of a `Sandwich` expression restricts
    the output blade support to the passenger's grades."""
    if not plan.target_gatype.entails(Sandwich):
        return plan.result_subspaces
    if not any(0 in group and 2 in group for group in plan.equality_groups):
        return plan.result_subspaces

    versor = next(binding.operand_gatype for binding in plan.bindings if binding.axis == 0)
    passenger = _transform_passenger(plan)
    algebra = plan.target_gatype.algebra

    if not versor.entails(Versor):
        return plan.result_subspaces
    allowed_grades = {algebra.grade(mask) for mask in passenger.masks}

    output = plan.result_subspaces[0].restrict(
        mask for mask in plan.result_subspaces[0].masks
        if algebra.grade(mask) in allowed_grades
    )
    return (output,) + plan.result_subspaces[1:]


def bind_traits(plan: BindingPlan) -> TraitSet:
    """Substituting bound operand traits into an operator's relational laws infers the result's
    `TraitSet`."""
    if plan.target_gatype.entails(Identity) and plan.bindings:
        operand = plan.bindings[0].operand_gatype
        if plan.result_subspaces[0].same_support(operand.output_subspace):
            return operand.traits
        # Widening output layout preserves value traits while dropping layout-dependent matrix orthogonality.
        return TraitSet(
            trait for trait in operand.effective_traits
            if isinstance(trait, (ProductFact, ProductRelation, ClosureRelation))
            or trait in (Versor, Blade)
        )

    inferred = []
    for relation in plan.target_gatype.traits:
        if isinstance(relation, ProductRelation):
            result = _bind_self_product(plan, relation)
            if result is not None:
                inferred.append(result)

    for kind, fact in _CLOSURES.items():
        relation = plan.target_gatype.self_products.closure(kind)
        if relation is not None:
            result = _bind_closure(plan, relation, fact)
            if result is not None:
                inferred.append(result)

    versor = _transform_versor(plan)
    if versor is not None:
        factor = (
            ProductResult.ONE if versor.entails(ReverseProductOne)
            else ProductResult.NONZERO
        )
        # Sandwiching x by m scales x * ~x by (m * ~m)**2 whenever x * ~x is scalar.
        result = _bind_self_product(plan, ProductRelation(ReverseProduct, factor, (1,)))
        if result is not None:
            inferred.append(result)
        for kind, fact in _CLOSURES.items():
            result = _bind_closure(plan, kind((1,)), fact)
            if result is not None:
                inferred.append(result)

        axes = bind_subspaces(plan)
        passenger = _transform_passenger(plan)
        if (
            factor is ProductResult.ONE
            and len(axes) == 2
            and axes[0].same_support(passenger)
            and axes[1].same_support(passenger)
            and all(square == 1 for square in versor.algebra.signature)
            and not any(binding.axis == 1 for binding in plan.bindings)
        ):
            # In Euclidean signatures the reverse scalar product equals the Euclidean dot product on blade coordinates.
            inferred.append(CoefficientOrthogonal)

    if plan.target_gatype.entails(CoefficientOrthogonal) and plan.bindings:
        binding = plan.bindings[0]
        if (
            binding.operand_gatype.entails(CoefficientOrthogonal)
            and binding.transform.is_lossless
            and binding.transform.source.same_support(binding.transform.target)
        ):
            inferred.append(CoefficientOrthogonal)
    return TraitSet(inferred) if inferred else EMPTY_TRAITS


def _bind_self_product(plan: BindingPlan, relation: ProductRelation) -> Trait | None:
    factor = relation.factor
    nonzero_inputs = relation.nonzero_inputs
    output_axes = []
    bindings = {binding.axis: binding for binding in plan.bindings}
    splices = plan.result_input_splices

    # Every referenced input must carry a scalar self-product in this family even when another factor is zero.
    for axis in relation.axes:
        binding = bindings.get(axis)
        if binding is None:
            output_axes.append(splices[axis][0])
            continue

        operand = binding.operand_gatype
        if operand.arity == 0:
            result = operand.self_products.result(relation.self_product)
            if result is None:
                return None
            if relation.nonzero_inputs and result not in (ProductResult.ONE, ProductResult.NONZERO):
                return None
            factor = multiply_product_results(factor, result)
            continue

        inner = operand.self_products.relation(relation.self_product)
        if inner is None:
            return None
        if relation.nonzero_inputs and inner.factor not in (ProductResult.ONE, ProductResult.NONZERO):
            return None
        nonzero_inputs = nonzero_inputs or inner.nonzero_inputs
        factor = multiply_product_results(factor, inner.factor)
        output_axes.extend(splices[axis][inner_axis] for inner_axis in inner.axes)

    if output_axes:
        return ProductRelation(relation.self_product, factor, output_axes, nonzero_inputs)
    if len(plan.result_subspaces) != 1:
        return None
    return ProductFact(relation.self_product, factor)


def _bind_closure(plan: BindingPlan, relation: ClosureRelation, fact: Trait) -> Trait | None:
    output_axes = []
    bindings = {binding.axis: binding for binding in plan.bindings}
    splices = plan.result_input_splices
    for axis in relation.axes:
        binding = bindings.get(axis)
        if binding is None:
            output_axes.append(splices[axis][0])
            continue
        operand = binding.operand_gatype
        if operand.arity == 0:
            if not operand.entails(fact):
                return None
            continue
        inner = operand.self_products.closure(type(relation))
        if inner is None:
            return None
        output_axes.extend(splices[axis][inner_axis] for inner_axis in inner.axes)

    if output_axes:
        return type(relation)(output_axes)
    if len(plan.result_subspaces) == 1:
        return fact
    return None


_CLOSURES = {VersorProduct: Versor, BladeProduct: Blade}
