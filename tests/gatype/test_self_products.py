"""Self-product facts stay local to their explicitly named product family."""

import pickle
from fractions import Fraction

import numpy as np
import pytest

from numga import (
    Algebra,
    CliffordConjugateProduct,
    CliffordConjugateProductNonzero,
    CliffordConjugateProductOne,
    CliffordConjugateProductScalar,
    CliffordConjugateProductZero,
    GATypeDispatch,
    GradeInvolutionProduct,
    ProductFact,
    ProductRelation,
    ProductResult,
    ReverseProduct,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    SelfProduct,
    Trait,
    TraitSet,
)


class _AliasFact(Trait):
    """Two names for one fact, with all state in immutable parameters."""

    __slots__ = ()

    def __init__(self, label):
        super().__init__(f"Alias{label}", valid_arities=(0,), parameters=(label,))

    def implied_traits(self):
        return (_AliasFact(1 - self._parameters[0]),)


def _relations(value):
    return {
        trait.self_product: trait
        for trait in value.gatype.traits
        if isinstance(trait, ProductRelation)
    }


def _value(algebra, traits, coefficients):
    gatype = algebra.gatype(algebra.subspace.full(), traits)
    return algebra.exact.multivector(gatype, coefficients)


def test_product_family_is_immutable_metadata_not_a_norm_alias():
    reverse = SelfProduct("reverse")
    projected = SelfProduct("reverse", product="scalar_product")

    assert reverse == ReverseProduct
    assert reverse != CliffordConjugateProduct
    assert reverse != projected
    assert ProductFact(reverse, ProductResult.ONE) == ReverseProductOne
    assert pickle.loads(pickle.dumps(reverse)) == reverse
    with pytest.raises(AttributeError):
        reverse.transform = "involute"

    # Knowing only the scalar projection is one is not knowing the whole
    # self-product is the scalar one: nonscalar components may remain.
    projected_one = ProductFact(projected, ProductResult.ONE)
    assert not TraitSet((projected_one,)).entails(ReverseProductOne)
    assert pickle.loads(pickle.dumps(projected_one)) == projected_one
    with pytest.raises(AttributeError, match="immutable"):
        projected_one.result = ProductResult.ZERO


def test_fact_implication_and_contradictions_are_family_local():
    facts = TraitSet((
        ReverseProductOne, ReverseProductNonzero, ReverseProductScalar,
        CliffordConjugateProductZero, CliffordConjugateProductScalar,
    ))

    assert facts == TraitSet((ReverseProductOne, CliffordConjugateProductZero))
    assert facts.entails(ReverseProductNonzero)
    assert facts.entails(CliffordConjugateProductScalar)
    assert not facts.entails(CliffordConjugateProductNonzero)
    with pytest.raises(ValueError, match="contradicts"):
        TraitSet((ReverseProductOne, ReverseProductZero))
    with pytest.raises(ValueError, match="contradicts"):
        TraitSet((CliffordConjugateProductOne, CliffordConjugateProductZero))


def test_custom_implication_aliases_canonicalize_without_erasing_the_fact():
    left, right = _AliasFact(0), _AliasFact(1)

    canonical = TraitSet((left, right))

    assert canonical == TraitSet((left,)) == TraitSet((right,))
    assert len(canonical) == 1
    assert canonical.entails(left)
    assert canonical.entails(right)


def test_custom_trait_pickle_preserves_its_parameters_and_implication_hook():
    original = _AliasFact(1)

    restored = pickle.loads(pickle.dumps(original))

    assert type(restored) is _AliasFact
    assert restored == original
    assert hash(restored) == hash(original)
    assert restored.implied_traits() == (_AliasFact(0),)
    assert TraitSet((restored,)).entails(_AliasFact(0))


def test_relations_with_the_same_slots_remain_independent_between_families():
    reverse = ProductRelation(ReverseProduct, ProductResult.ONE, (1, 0, 1))
    conjugate = ProductRelation(
        CliffordConjugateProduct, ProductResult.ZERO, (0, 1, 1),
    )
    weaker_reverse = ProductRelation(
        ReverseProduct, ProductResult.SCALAR, (0, 1, 1),
    )

    facts = TraitSet((reverse, conjugate, weaker_reverse))

    assert reverse.slots == (0, 1, 1)
    assert facts == TraitSet((reverse, conjugate))
    assert pickle.loads(pickle.dumps(facts)) == facts
    with pytest.raises(AttributeError, match="immutable"):
        reverse.factor = ProductResult.ZERO
    with pytest.raises(AttributeError, match="immutable"):
        reverse.slots = (0,)
    with pytest.raises(ValueError, match="contradicts"):
        TraitSet((
            reverse,
            ProductRelation(ReverseProduct, ProductResult.ZERO, (0, 1, 1)),
        ))


def test_composition_and_binding_carry_both_registered_product_families():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    # x reverse(x) = 1; x conjugate(x) = -1, hence nonzero but not one.
    x = _value(
        algebra, (ReverseProductOne, CliffordConjugateProductNonzero),
        [0, 1, 0, 0],
    )
    ternary = full * full * full

    assert _relations(ternary) == {
        family: ProductRelation(family, ProductResult.ONE, (0, 1, 2))
        for family in (ReverseProduct, CliffordConjugateProduct)
    }
    atomic = ternary.bind({0: x, 2: x})
    sequential = ternary.bind({0: x}).bind({1: x})

    assert atomic.gatype is sequential.gatype
    assert _relations(atomic) == {
        ReverseProduct: ProductRelation(ReverseProduct, ProductResult.ONE, (0,)),
        CliffordConjugateProduct: ProductRelation(
            CliffordConjugateProduct, ProductResult.NONZERO, (0,),
        ),
    }
    result = atomic(x)
    assert result.gatype.entails(ReverseProductOne)
    assert result.gatype.entails(CliffordConjugateProductNonzero)
    assert not result.gatype.entails(CliffordConjugateProductOne)
    np.testing.assert_array_equal(result.kernel.values, [0, 1, 0, 0])


def test_user_family_can_dispatch_without_inventing_a_product_law():
    algebra = Algebra("x+y+")
    custom = SelfProduct("user_transform")
    custom_one = ProductFact(custom, ProductResult.ONE)
    value = _value(algebra, (custom_one,), [1, 0, 0, 0])
    dispatch = GATypeDispatch("custom_operation", None, 1)

    @dispatch.register(custom_one)
    def implementation(operand):
        return operand

    assert dispatch.resolve(value.gatype) is implementation
    assert dispatch(value) is value
    assert custom not in _relations(algebra.gatype.full() * algebra.gatype.full())
    assert not (value * value).gatype.entails(custom_one)


def test_grade_involution_has_no_anti_automorphism_multiplicativity_rule():
    algebra = Algebra("x+y-")
    involute_one = ProductFact(GradeInvolutionProduct, ProductResult.ONE)
    # Both negative-unit vectors satisfy a involute(a) = 1.
    a = _value(algebra, (involute_one,), [0, 0, 1, 0])
    b = _value(algebra, (involute_one,), [0, Fraction(3, 4), Fraction(5, 4), 0])

    product = a * b

    assert GradeInvolutionProduct not in _relations(
        algebra.gatype.full() * algebra.gatype.full(),
    )
    assert not product.gatype.entails(involute_one)
    # The product is even, so involute(product) = product. Its self-product
    # has a bivector part: neither scalar nor one is a sound inferred fact.
    np.testing.assert_array_equal(
        (product * product).kernel.values,
        [Fraction(17, 8), 0, 0, Fraction(15, 8)],
    )
    assert not product.gatype.entails(
        ProductFact(GradeInvolutionProduct, ProductResult.SCALAR),
    )


def test_zero_factor_keeps_a_separate_scalar_guard_for_each_family():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    zero = _value(
        algebra, (ReverseProductZero, CliffordConjugateProductZero),
        [0, 0, 0, 0],
    )
    conjugate_only = _value(algebra, (CliffordConjugateProductOne,), [1, 0, 0, 0])
    partial = (full * full).bind({0: zero})

    assert _relations(partial) == {
        family: ProductRelation(family, ProductResult.ZERO, (0,))
        for family in (ReverseProduct, CliffordConjugateProduct)
    }
    result = partial(conjugate_only)

    assert result.gatype.entails(CliffordConjugateProductZero)
    assert not result.gatype.entails(ReverseProductScalar)
    np.testing.assert_array_equal(result.kernel.values, [0, 0, 0, 0])
