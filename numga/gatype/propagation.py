"""Explicit mathematical laws and shared substitution for Extensor facts."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .gatype import GAType
from .traits import (
    EMPTY_TRAITS,
    CliffordConjugateProduct,
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
    """Multiply scalar categories without assuming reality or positivity."""

    if left is ProductResult.ONE:
        return right
    if right is ProductResult.ONE:
        return left
    if left is ProductResult.ZERO or right is ProductResult.ZERO:
        return ProductResult.ZERO
    if left is ProductResult.NONZERO and right is ProductResult.NONZERO:
        return ProductResult.NONZERO
    return ProductResult.SCALAR


def no_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """An operation with no known law attaches no traits."""
    return EMPTY_TRAITS


def geometric_product_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """The reverse and the Clifford conjugate satisfy T(ab) = T(b)T(a), so the geometric product
    relates its result's self-products to its operands'; a product of versors is a versor. Being an
    involution alone is not enough, and no other self-product registers this theorem."""
    return TraitSet((
        ProductRelation(ReverseProduct, ProductResult.ONE, (0, 1)),
        ProductRelation(CliffordConjugateProduct, ProductResult.ONE, (0, 1)),
        VersorProduct(slots=(0, 1)),
    ))


def transform_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """A versor's action carries the sandwich laws, applied when the same versor binds both sides."""
    return TraitSet((Sandwich,))


def negation_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """Negation and grade involution leave quadratic products unchanged; grade involution is an
    automorphism commuting with these transforms and fixing scalars. Neither preserves arbitrary user
    traits or the identity map."""
    return TraitSet(
        trait for trait in operands[0].effective_traits
        if (
            isinstance(trait, (ProductFact, ProductRelation))
            and trait.self_product in (ReverseProduct, CliffordConjugateProduct, GradeInvolutionProduct)
        )
        or isinstance(trait, VersorProduct)
        or trait in (Versor, CoefficientOrthogonal)
    )


def transpose_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """A transpose keeps coefficient orthogonality."""
    return TraitSet((CoefficientOrthogonal,)) if operands[0].entails(CoefficientOrthogonal) else EMPTY_TRAITS


def coefficient_orthogonality_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """A sign change of output coordinates keeps coefficient orthogonality."""
    return TraitSet((CoefficientOrthogonal,)) if operands[0].entails(CoefficientOrthogonal) else EMPTY_TRAITS


def shared_laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
    """Values set side by side each keep their facts, so the collection has the facts they share. Facts
    about whole values survive a zero embedding; facts about maps, such as orthogonality, need their
    original codomain."""
    if operands[0].arity and any(gatype.output_subspace != axes[0] for gatype in operands):
        return EMPTY_TRAITS
    shared = set(operands[0].effective_traits.closure)
    for gatype in operands[1:]:
        shared.intersection_update(gatype.effective_traits.closure)
    return TraitSet(shared)


def conjugation_laws(family: SelfProduct):
    """The laws of the reverse or the Clifford conjugate, the transform of the self-product `family`.

    A one-sided scalar or zero self-product cannot in general be swapped; a nonzero scalar gives a
    genuine inverse in finite dimensions. These transforms take products of vectors to products of
    vectors, independently of self-product inference.
    """

    def laws(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
        operand = operands[0]
        inferred: list[Trait] = []
        # Any output-coordinate sign change preserves coefficient orthogonality.
        if operand.entails(CoefficientOrthogonal):
            inferred.append(CoefficientOrthogonal)
        for result in (ProductResult.ONE, ProductResult.NONZERO):
            fact = ProductFact(family, result)
            if operand.entails(fact):
                inferred.append(fact)
                break
        if operand.entails(Versor):
            inferred.append(Versor)
        relation = operand.self_products.relation(family)
        if relation is not None and relation.factor in (ProductResult.ONE, ProductResult.NONZERO):
            inferred.append(ProductRelation(family, relation.factor, relation.slots, nonzero_inputs=True))
        versor_relation = operand.self_products.versor_relation
        if versor_relation is not None:
            inferred.append(versor_relation)
        return TraitSet(inferred)

    return laws


def unless_identity(transform: str, laws):
    """A grade transform's laws, except where the transform is the identity on its operand's grades:
    there the result is the operand, with all its traits."""

    def guarded(operands: tuple[GAType, ...], axes: tuple["SubSpace", ...]) -> TraitSet:
        if operands[0].grade_transform_is_identity(transform):
            return operands[0].traits
        return laws(operands, axes)

    return guarded



def _transform_versor(plan: "BindingPlan") -> GAType | None:
    """The same certified versor occurs on both sides of this binding."""

    if not plan.target_gatype.entails(Sandwich):
        return None
    if not any(0 in group and 2 in group for group in plan.equality_groups):
        return None
    operand = next(binding.operand_gatype for binding in plan.bindings if binding.slot == 0)
    # The inferred self-product relations divide by the versor's reverse product.
    return operand if operand.entails(Versor) and operand.entails(ReverseProductNonzero) else None


def _transform_passenger(plan: "BindingPlan") -> SubSpace:
    return next(
        (binding.operand_gatype.output_subspace for binding in plan.bindings if binding.slot == 1),
        plan.target_gatype.input_subspaces[1],
    )


@lru_cache(maxsize=None)
def bind_subspaces(plan: "BindingPlan") -> tuple[SubSpace, ...]:
    """Apply certified grade preservation before constructing the kernel."""

    if not plan.target_gatype.entails(Sandwich):
        return plan.result_subspaces
    if not any(0 in group and 2 in group for group in plan.equality_groups):
        return plan.result_subspaces

    versor = next(binding.operand_gatype for binding in plan.bindings if binding.slot == 0)
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


def bind_traits(plan: "BindingPlan") -> TraitSet:
    """Specialize each declared relation independently through one binding."""

    if plan.target_gatype.entails(Identity) and plan.bindings:
        operand = plan.bindings[0].operand_gatype
        if plan.result_subspaces[0].same_support(operand.output_subspace):
            return operand.traits
        # Embedding changes the coefficient layout, not the output value.
        # Do not copy layout-specific claims such as matrix orthogonality.
        return TraitSet(
            trait for trait in operand.effective_traits
            if isinstance(trait, (ProductFact, ProductRelation, VersorProduct))
            or trait == Versor
        )

    inferred: list[Trait] = []
    for relation in plan.target_gatype.traits:
        if isinstance(relation, ProductRelation):
            result = _bind_self_product(plan, relation)
            if result is not None:
                inferred.append(result)

    relation = plan.target_gatype.self_products.versor_relation
    if relation is not None:
        result = _bind_versor_product(plan, relation)
        if result is not None:
            inferred.append(result)

    versor = _transform_versor(plan)
    if versor is not None:
        factor = (
            ProductResult.ONE if versor.entails(ReverseProductOne)
            else ProductResult.NONZERO
        )
        # For y = m*x*reverse(m), y*reverse(y) =
        # (m*reverse(m))**2 * (x*reverse(x)), whenever x's product is scalar.
        result = _bind_self_product(plan, ProductRelation(ReverseProduct, factor, (1,)))
        if result is not None:
            inferred.append(result)
        result = _bind_versor_product(plan, VersorProduct((1,)))
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
            and not any(binding.slot == 1 for binding in plan.bindings)
        ):
            # In a Euclidean algebra the reverse scalar product is exactly
            # the coefficient dot product. This is not true of PGA/Lorentz.
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


def _bind_self_product(plan: "BindingPlan", relation: ProductRelation) -> Trait | None:
    """Substitute a declared relation, without rediscovering its theorem."""

    factor = relation.factor
    nonzero_inputs = relation.nonzero_inputs
    result_slots: list[int] = []
    bindings = {binding.slot: binding for binding in plan.bindings}
    splices = plan.result_input_splices

    # A zero factor does not remove the scalar guard on other inputs. Every
    # referenced input still needs a scalar self-product in THIS family.
    for slot in relation.slots:
        binding = bindings.get(slot)
        if binding is None:
            result_slots.append(splices[slot][0])
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
        result_slots.extend(splices[slot][inner_slot] for inner_slot in inner.slots)

    if result_slots:
        return ProductRelation(relation.self_product, factor, result_slots, nonzero_inputs)
    if len(plan.result_subspaces) != 1:
        # No current law becomes independent of remaining input slots.
        return None
    return ProductFact(relation.self_product, factor)


def _bind_versor_product(plan: "BindingPlan", relation: VersorProduct) -> Trait | None:
    """Substitute the closure of versors, products of vectors, under geometric product."""

    result_slots: list[int] = []
    bindings = {binding.slot: binding for binding in plan.bindings}
    splices = plan.result_input_splices
    for slot in relation.slots:
        binding = bindings.get(slot)
        if binding is None:
            result_slots.append(splices[slot][0])
            continue
        operand = binding.operand_gatype
        if operand.arity == 0:
            if not operand.entails(Versor):
                return None
            continue
        inner = operand.self_products.versor_relation
        if inner is None:
            return None
        result_slots.extend(splices[slot][inner_slot] for inner_slot in inner.slots)

    if result_slots:
        return VersorProduct(result_slots)
    if len(plan.result_subspaces) == 1:
        return Versor
    return None
