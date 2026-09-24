from __future__ import annotations

import pickle

import pytest

from numga import (
    Algebra,
    EMPTY_TRAITS,
    GATypeDispatch,
    GATypePattern,
    ProductResult,
    ProductRelation,
    ReverseProduct,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    TraitSet,
    Versor,
)


def _relation(gatype):
    relations = tuple(
        trait
        for trait in gatype.traits
        if isinstance(trait, ProductRelation) and trait.self_product == ReverseProduct
    )
    assert len(relations) == 1
    return relations[0]


def _typed_full(algebra, trait, coefficients):
    gatype = algebra.gatype(algebra.subspace.full(), (trait,))
    return algebra.exact.multivector(gatype, coefficients)


def test_reverse_product_relation_is_canonical_immutable_metadata():
    relation = ProductRelation(ReverseProduct, ProductResult.NONZERO, slots=(1, 0, 1))

    assert relation.slots == (0, 1, 1)
    assert relation == ProductRelation(
        ReverseProduct,
        ProductResult.NONZERO,
        slots=(0, 1, 1),
    )
    assert pickle.loads(pickle.dumps(relation)) == relation
    with pytest.raises(AttributeError, match="immutable"):
        relation.factor = ProductResult.ONE
    with pytest.raises(ValueError, match="at least one"):
        ProductRelation(ReverseProduct, ProductResult.ONE, slots=())

    weaker = ProductRelation(ReverseProduct, ProductResult.SCALAR, slots=(0, 1, 1))
    assert TraitSet((relation, weaker)) == TraitSet((relation,))
    with pytest.raises(ValueError, match="contradicts"):
        TraitSet(
            (
                ProductRelation(ReverseProduct, ProductResult.ONE, slots=(0,)),
                ProductRelation(ReverseProduct, ProductResult.ZERO, slots=(0,)),
            )
        )


def test_reverse_product_relation_validates_slots_against_whole_gatype_arity():
    algebra = Algebra("x+y+")
    full = algebra.subspace.full()

    relation = ProductRelation(ReverseProduct, ProductResult.ONE, slots=(0, 1))
    gatype = algebra.gatype((full, full, full), (relation,))

    assert gatype.traits == TraitSet((relation,))
    with pytest.raises(ValueError, match="outside arity"):
        algebra.gatype((full, full), (relation,))

    dispatch = GATypeDispatch("relation", None, 1)
    with pytest.raises(TypeError, match="explicit GATypePattern arity"):
        dispatch.register(relation)

    @dispatch.register(GATypePattern.nary(2, relation))
    def implementation(_value):
        return "matched"

    assert dispatch.resolve(gatype) is implementation


def test_relation_factor_lattice_participates_in_refinement_and_dispatch():
    algebra = Algebra("x+y+")
    full = algebra.subspace.full()
    axes = (full, full, full)
    one_relation = ProductRelation(ReverseProduct, ProductResult.ONE, slots=(0, 1))
    nonzero_relation = ProductRelation(
        ReverseProduct,
        ProductResult.NONZERO,
        slots=(0, 1),
    )
    scalar_relation = ProductRelation(ReverseProduct, ProductResult.SCALAR, slots=(0, 1))
    zero_relation = ProductRelation(ReverseProduct, ProductResult.ZERO, slots=(0, 1))

    one = algebra.gatype(axes, (one_relation,))
    nonzero = algebra.gatype(axes, (nonzero_relation,))
    scalar = algebra.gatype(axes, (scalar_relation,))
    zero = algebra.gatype(axes, (zero_relation,))

    assert one < nonzero < scalar
    assert zero < scalar
    assert one.overlaps(nonzero)
    assert not one.overlaps(zero)
    assert algebra.gatype(
        axes,
        (one_relation, nonzero_relation, scalar_relation),
    ) is one

    dispatch = GATypeDispatch("relation", None, 1)

    @dispatch.register(GATypePattern.nary(2, one_relation))
    def specific(_value):
        return "specific"

    @dispatch.register(GATypePattern.nary(2, nonzero_relation))
    def general(_value):
        return "general"

    assert dispatch.resolve(one) is specific
    assert dispatch.resolve(nonzero) is general


def test_geometric_product_declares_conditional_reverse_product_relation():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()

    product = full * full

    assert product.arity == 2
    assert _relation(product.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.ONE,
        slots=(0, 1),
    )
    # This is a relation, not an unconditional claim about every output.
    assert not product.gatype.entails(ReverseProductScalar)


def test_partial_binding_accumulates_known_reverse_product_factor():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    nonzero = _typed_full(
        algebra,
        ReverseProductNonzero,
        [2, 0, 0, 0],
    )

    partial = (full * full).bind({0: nonzero})

    assert _relation(partial.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.NONZERO,
        slots=(0,),
    )


@pytest.mark.parametrize(
    ("left_fact", "left_coefficients", "right_fact", "right_coefficients", "expected"),
    (
        (
            ReverseProductOne,
            [1, 0, 0, 0],
            ReverseProductOne,
            [1, 0, 0, 0],
            ReverseProductOne,
        ),
        (
            ReverseProductOne,
            [1, 0, 0, 0],
            ReverseProductNonzero,
            [2, 0, 0, 0],
            ReverseProductNonzero,
        ),
        (
            ReverseProductNonzero,
            [2, 0, 0, 0],
            ReverseProductScalar,
            [3, 0, 0, 0],
            ReverseProductScalar,
        ),
        (
            ReverseProductZero,
            [0, 0, 0, 0],
            ReverseProductScalar,
            [3, 0, 0, 0],
            ReverseProductZero,
        ),
    ),
)
def test_full_binding_collapses_relation_to_a_value_fact(
    left_fact,
    left_coefficients,
    right_fact,
    right_coefficients,
    expected,
):
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    left = _typed_full(algebra, left_fact, left_coefficients)
    right = _typed_full(algebra, right_fact, right_coefficients)

    result = (full * full)(left, right)

    assert result.gatype.traits == TraitSet((expected,))


def test_zero_factor_does_not_erase_the_remaining_scalar_guard():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    zero = _typed_full(algebra, ReverseProductZero, [0, 0, 0, 0])
    one = _typed_full(algebra, ReverseProductOne, [1, 0, 0, 0])
    unknown = algebra.exact.multivector(full, [1, 1, 0, 0])

    partial = (full * full).bind({0: zero})

    assert _relation(partial.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.ZERO,
        slots=(0,),
    )
    assert partial(one).gatype.traits == TraitSet((ReverseProductZero,))
    assert partial(unknown).gatype.traits is EMPTY_TRAITS


def test_nested_product_substitutes_and_remaps_relations():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()

    ternary = full * full * full

    assert ternary.arity == 3
    assert _relation(ternary.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.ONE,
        slots=(0, 1, 2),
    )

    inner = full * full
    repeated = (full * full).bind({0: inner, 1: inner})
    assert repeated.arity == 4
    assert _relation(repeated.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.ONE,
        slots=(0, 1, 2, 3),
    )


def test_atomic_and_sequential_binding_share_the_same_flyweight_type():
    algebra = Algebra("x+y+")
    full = algebra.gatype.full()
    one = _typed_full(algebra, ReverseProductOne, [1, 0, 0, 0])
    nonzero = _typed_full(
        algebra,
        ReverseProductNonzero,
        [2, 0, 0, 0],
    )
    ternary = full * full * full

    atomic = ternary.bind({0: nonzero, 2: one})
    sequential = ternary.bind({0: nonzero}).bind({1: one})

    assert atomic.gatype is sequential.gatype
    assert _relation(atomic.gatype) == ProductRelation(
        ReverseProduct,
        ProductResult.NONZERO,
        slots=(0,),
    )


def test_reverse_preserves_only_sound_one_sided_reverse_product_facts():
    algebra = Algebra("x+y-")
    coefficients = [-1, -1, -1, -1]

    unit = _typed_full(algebra, ReverseProductOne, coefficients)
    nonzero = _typed_full(algebra, ReverseProductNonzero, coefficients)
    versor = _typed_full(algebra, Versor, coefficients)
    scalar = _typed_full(algebra, ReverseProductScalar, coefficients)
    zero = _typed_full(algebra, ReverseProductZero, coefficients)

    assert unit.reverse().gatype.entails(ReverseProductOne)
    assert nonzero.reverse().gatype.entails(ReverseProductNonzero)
    assert versor.reverse().gatype.entails(Versor)
    assert not scalar.reverse().gatype.entails(ReverseProductScalar)
    assert not zero.reverse().gatype.entails(ReverseProductScalar)

    # Concrete Cl(1, 1) witness: x*~x is zero, while ~x*x is not scalar.
    reversed_zero = zero.reverse()
    assert (zero * reversed_zero).kernel.values.tolist() == [
        0,
        0,
        0,
        0,
    ]
    assert (reversed_zero * zero).kernel.values.tolist() == [
        0,
        4,
        4,
        0,
    ]

    # Reversing an open expression must retain the same nonzero premise.
    full = algebra.gatype.full()
    one = _typed_full(algebra, ReverseProductOne, [1, 0, 0, 0])
    reverse_product = (full * full).reverse()
    partial = reverse_product.bind({1: one})
    assert partial(one).gatype.entails(ReverseProductOne)
    assert not partial(zero).gatype.entails(ReverseProductScalar)
