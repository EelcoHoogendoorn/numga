from __future__ import annotations

import warnings

import pytest

from numga import Algebra, GATypeError
from numga.gatype import (
    AmbiguousGATypeDispatchWarning,
    GATypeDispatch,
    GATypePattern,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
)


def test_generic_trait_patterns_dispatch_across_algebras():
    dispatch = GATypeDispatch("portable", None, 1, takes="a value")

    @dispatch.register(ReverseProductOne)
    def unit(value):
        return value

    for algebra in (Algebra("x+y+"), Algebra("a+b+c+")):
        assert dispatch.resolve(algebra.gatype.rotor()) is unit

    unrelated = Algebra("u+v+")
    with pytest.raises(GATypeError):
        dispatch.resolve(unrelated.gatype.even())


def test_concrete_pattern_specializes_generic_pattern_for_its_algebra():
    local = Algebra("x+y+")
    foreign = Algebra("x+y+")
    dispatch = GATypeDispatch("mixed", None, 1, takes="a value")

    @dispatch.register(local.gatype.even())
    def local_even(value):
        return value

    @dispatch.register(GATypePattern(arity=0))
    def generic_value(value):
        return value

    assert dispatch.resolve(local.gatype.rotor()) is local_even
    assert dispatch.resolve(foreign.gatype.rotor()) is generic_value


def test_dispatch_matches_complete_positive_arity_gatypes():
    algebra = Algebra("x+y+z+")
    gatypes = algebra.gatype
    dispatch = GATypeDispatch("factor", algebra, 1, takes="a value")
    pattern = gatypes((gatypes.full(), gatypes.vector(), gatypes.full()))

    @dispatch.register(pattern)
    def implementation(value):
        return value

    actual = gatypes((gatypes.vector(), gatypes.vector(), gatypes.scalar()))
    wrong_axis = gatypes((gatypes.vector(), gatypes.bivector(), gatypes.scalar()))
    wrong_arity = gatypes((gatypes.vector(), gatypes.vector()))

    assert dispatch.resolve(actual) is implementation
    with pytest.raises(GATypeError):
        dispatch.resolve(wrong_axis)
    with pytest.raises(GATypeError):
        dispatch.resolve(wrong_arity)


def test_dispatch_uses_entailed_and_structurally_implied_facts():
    algebra = Algebra("x+y+")
    gatypes = algebra.gatype

    nonzero_dispatch = GATypeDispatch("nonzero", algebra, 1, takes="a value")

    @nonzero_dispatch.register(
        gatypes(gatypes.full(), (ReverseProductNonzero,))
    )
    def nonzero(value):
        return value

    assert nonzero_dispatch.resolve(
        gatypes(gatypes.even(), (ReverseProductOne,))
    ) is nonzero

    scalar_dispatch = GATypeDispatch("scalar_product", algebra, 1, takes="a value")

    @scalar_dispatch.register(
        gatypes(gatypes.full(), (ReverseProductScalar,))
    )
    def scalar_product(value):
        return value

    assert scalar_dispatch.resolve(gatypes.scalar()) is scalar_product
    assert gatypes.scalar() is gatypes(
        gatypes.scalar(),
        (ReverseProductScalar,),
    )


def test_registration_requires_specific_to_general_order():
    algebra = Algebra("x+y+")
    gatypes = algebra.gatype
    specific = gatypes.even()
    general = gatypes.full()

    valid = GATypeDispatch("valid", algebra, 1, takes="a value")

    @valid.register(specific)
    def specialized(value):
        return value

    @valid.register(general)
    def fallback(value):
        return value

    assert valid.resolve(specific) is specialized
    assert valid.resolution_cache_size == 1
    # Every registry mutation invalidates exact-type resolutions.
    disjoint = gatypes((general, general))

    @valid.register(disjoint)
    def unary_map(value):
        return value

    assert valid.resolution_cache_size == 0
    assert valid.resolve(specific) is specialized
    assert valid.resolve(general) is fallback

    invalid = GATypeDispatch("invalid", algebra, 1, takes="a value")
    invalid.register(general)(fallback)
    with pytest.raises(ValueError, match="specialization.*appears after"):
        invalid.register(specific)
    with pytest.raises(ValueError, match="duplicate"):
        valid.register(specific)


def test_incomparable_overlap_warns_and_uses_declaration_order():
    algebra = Algebra("x+y+")
    gatypes = algebra.gatype
    even = gatypes.even()
    unit = gatypes(gatypes.full(), (ReverseProductOne,))
    actual = gatypes(gatypes.even(), (ReverseProductOne,))
    dispatch = GATypeDispatch("ambiguous", algebra, 1, takes="a value")

    @dispatch.register(even)
    def structural(value):
        return value

    with pytest.warns(
        AmbiguousGATypeDispatchWarning,
        match="declaration order is the tie-breaker",
    ):

        @dispatch.register(unit)
        def factual(value):
            return value

    assert dispatch.resolve(actual) is structural

    acknowledged = GATypeDispatch("acknowledged", algebra, 1, takes="a value")
    acknowledged.register(even)(structural)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        acknowledged.register(unit, precedence="declaration")(factual)
    assert not caught


def test_multiple_dispatch_uses_product_order_specificity():
    algebra = Algebra("x+y+")
    gatypes = algebra.gatype
    dispatch = GATypeDispatch("binary", algebra, 2, takes="a value")
    generic_map = gatypes((gatypes.full(), gatypes.full()))
    vector_map = gatypes((gatypes.vector(), gatypes.vector()))
    generic_value = gatypes.full()
    vector_value = gatypes.vector()

    @dispatch.register(vector_map, vector_value)
    def intersection(left, right):
        return left, right

    @dispatch.register(vector_map, generic_value)
    def left_specific(left, right):
        return left, right

    @dispatch.register(
        generic_map,
        vector_value,
        precedence="declaration",
    )
    def right_specific(left, right):
        return left, right

    @dispatch.register(generic_map, generic_value)
    def fallback(left, right):
        return left, right

    assert dispatch.resolve(vector_map, vector_value) is intersection
    assert dispatch.resolve(vector_map, generic_value) is left_specific
    assert dispatch.resolve(generic_map, vector_value) is right_specific
    assert dispatch.resolve(generic_map, generic_value) is fallback
