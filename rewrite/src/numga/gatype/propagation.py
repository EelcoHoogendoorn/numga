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


# These transforms satisfy T(ab)=T(b)T(a). Merely being an involution is
# insufficient. A new SelfProduct descriptor does not register a theorem.
# Coefficient conjugation/adjoints require a conjugate-linear execution
# contract and are deliberately not installed by this first pass.
_MULTIPLICATIVE_SELF_PRODUCTS = {
    "geometric_product": (ReverseProduct, CliffordConjugateProduct),
}


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


def self_product_result(gatype: GAType, self_product: SelfProduct) -> ProductResult | None:
    """Return the strongest result in exactly this self-product family."""

    for result in (
        ProductResult.ONE, ProductResult.NONZERO,
        ProductResult.ZERO, ProductResult.SCALAR,
    ):
        if gatype.entails(ProductFact(self_product, result)):
            return result
    return None


def self_product_relation(
    gatype: GAType, self_product: SelfProduct,
) -> ProductRelation | None:
    """Look up a relation by family; different families never substitute."""

    return next((
        trait for trait in gatype.traits
        if isinstance(trait, ProductRelation) and trait.self_product == self_product
    ), None)


def versor_product_relation(gatype: GAType) -> VersorProduct | None:
    return next((t for t in gatype.traits if isinstance(t, VersorProduct)), None)


def operation_traits(
    operation: object,
    operand_gatypes: tuple[GAType, ...],
    result_subspaces: tuple["SubSpace", ...],
) -> TraitSet:
    """Attach only the laws explicitly known for this operation."""

    families = _MULTIPLICATIVE_SELF_PRODUCTS.get(operation, ())
    if families:
        return TraitSet((
            *(ProductRelation(family, ProductResult.ONE, (0, 1)) for family in families),
            VersorProduct(slots=(0, 1)),
        ))

    if operation == "sandwich":
        return TraitSet((Sandwich,))

    if operation in (
        "reverse", "clifford_conjugate", "involute",
        "scalar_negation", "pseudoscalar_negation",
    ) and operand_gatypes[0].grade_transform_is_identity(operation):
        return operand_gatypes[0].traits

    if operation in ("negative", "involute"):
        # Negation leaves quadratic products unchanged. Grade involution is an
        # automorphism commuting with these transforms and fixing scalars.
        # Neither preserves arbitrary user traits or the identity map.
        return TraitSet(
            trait for trait in operand_gatypes[0].effective_traits
            if (
                isinstance(trait, (ProductFact, ProductRelation))
                and trait.self_product in (
                    ReverseProduct, CliffordConjugateProduct, GradeInvolutionProduct,
                )
            )
            or isinstance(trait, VersorProduct)
            or trait in (Versor, CoefficientOrthogonal)
        )

    if operation == "transpose" and len(operand_gatypes) == 1:
        if operand_gatypes[0].entails(CoefficientOrthogonal):
            return TraitSet((CoefficientOrthogonal,))

    if operation in ("reverse", "clifford_conjugate") and len(operand_gatypes) == 1:
        operand = operand_gatypes[0]
        # A one-sided scalar/zero self-product cannot in general be swapped.
        # A nonzero scalar gives a genuine inverse in finite dimensions.
        family = SelfProduct(operation)
        inferred: list[Trait] = []
        # Any output-coordinate sign change preserves coefficient orthogonality.
        if operand.entails(CoefficientOrthogonal):
            inferred.append(CoefficientOrthogonal)
        for result in (ProductResult.ONE, ProductResult.NONZERO):
            fact = ProductFact(family, result)
            if operand.entails(fact):
                inferred.append(fact)
                break
        # These transforms take products of invertible vectors to products
        # of invertible vectors, independently of self-product inference.
        if operand.entails(Versor):
            inferred.append(Versor)
        relation = self_product_relation(operand, family)
        if relation is not None and relation.factor in (ProductResult.ONE, ProductResult.NONZERO):
            inferred.append(ProductRelation(
                family, relation.factor, relation.slots, nonzero_inputs=True,
            ))
        versor_relation = versor_product_relation(operand)
        if versor_relation is not None:
            inferred.append(versor_relation)
        return TraitSet(inferred)

    if operation in ("scalar_negation", "pseudoscalar_negation"):
        if operand_gatypes[0].entails(CoefficientOrthogonal):
            return TraitSet((CoefficientOrthogonal,))

    return EMPTY_TRAITS


def _sandwicher(plan: "BindingPlan") -> GAType | None:
    """The same certified versor occurs on both sides of this binding."""

    if not plan.target_gatype.entails(Sandwich):
        return None
    if not any(0 in group and 2 in group for group in plan.equality_groups):
        return None
    operand = next(binding.operand_gatype for binding in plan.bindings if binding.slot == 0)
    return operand if operand.entails(Versor) else None


def _sandwich_passenger(plan: "BindingPlan") -> SubSpace:
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

    sandwicher = next(binding.operand_gatype for binding in plan.bindings if binding.slot == 0)
    passenger = _sandwich_passenger(plan)
    algebra = plan.target_gatype.algebra

    if not sandwicher.entails(Versor):
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

    relation = versor_product_relation(plan.target_gatype)
    if relation is not None:
        result = _bind_versor_product(plan, relation)
        if result is not None:
            inferred.append(result)

    sandwicher = _sandwicher(plan)
    if sandwicher is not None:
        factor = (
            ProductResult.ONE if sandwicher.entails(ReverseProductOne)
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
        passenger = _sandwich_passenger(plan)
        if (
            factor is ProductResult.ONE
            and len(axes) == 2
            and axes[0].same_support(passenger)
            and axes[1].same_support(passenger)
            and all(square == 1 for square in sandwicher.algebra.signature)
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
            result = self_product_result(operand, relation.self_product)
            if result is None:
                return None
            if relation.nonzero_inputs and result not in (ProductResult.ONE, ProductResult.NONZERO):
                return None
            factor = multiply_product_results(factor, result)
            continue

        inner = self_product_relation(operand, relation.self_product)
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
    """Substitute the separate closure of versors under geometric product."""

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
        inner = versor_product_relation(operand)
        if inner is None:
            return None
        result_slots.extend(splices[slot][inner_slot] for inner_slot in inner.slots)

    if result_slots:
        return VersorProduct(result_slots)
    if len(plan.result_subspaces) == 1:
        return Versor
    return None
