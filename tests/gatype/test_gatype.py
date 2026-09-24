from __future__ import annotations


import pytest


from numga.gatype import (
    ROTOR_TRAITS,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    TraitSet,
    Versor,
)


def test_trait_sets_canonicalize_implications_and_reject_contradictions() -> None:
    one = TraitSet(
        (ReverseProductOne, ReverseProductNonzero, ReverseProductScalar)
    )

    assert one == TraitSet((ReverseProductOne,))
    assert one.traits == (ReverseProductOne,)
    assert one.entails(ReverseProductOne)
    assert one.entails(ReverseProductNonzero)
    assert one.entails(ReverseProductScalar)
    assert not one.entails(ReverseProductZero)
    assert ROTOR_TRAITS.entails(ReverseProductNonzero)
    assert ROTOR_TRAITS.entails(ReverseProductScalar)

    with pytest.raises(ValueError, match="contradicts"):
        TraitSet((ReverseProductZero, ReverseProductOne))


def test_gatype_refinement_combines_support_and_known_facts() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    even = algebra.subspace.even()
    full = algebra.subspace.full()
    vector = algebra.subspace.vector()

    rotor = algebra.gatype.rotor()
    generic_versor = algebra.gatype(full, (Versor,))
    generic_value = algebra.gatype.full()
    generic_even = algebra.gatype.even()
    unit_value = algebra.gatype(full, (ReverseProductOne,))

    assert rotor < generic_versor < generic_value
    assert rotor < generic_even < generic_value
    assert rotor.entails(ReverseProductNonzero)
    assert rotor.entails(ReverseProductScalar)
    assert not generic_even.entails(Versor)

    assert not generic_even.refines(unit_value)
    assert not unit_value.refines(generic_even)
    assert generic_even.overlaps(unit_value)
    assert not algebra.gatype(vector).overlaps(generic_even)

    redundant = algebra.gatype(
        even,
        (ReverseProductOne, ReverseProductNonzero, ReverseProductScalar),
    )
    assert redundant is algebra.gatype(even, (ReverseProductOne,))


def test_refinement_comparisons_lift_subspaces_to_plain_nullary_gatypes() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    bivector = algebra.subspace.bivector()
    plain = algebra.gatype.bivector()
    blade = algebra.gatype.from_masks((3,))
    refined = algebra.gatype(bivector, (ReverseProductOne,))

    # The lift adds neither support restrictions nor certified traits.
    assert plain <= bivector
    assert plain >= bivector
    assert bivector <= plain
    assert bivector >= plain
    assert not plain < bivector
    assert not plain > bivector

    assert blade < bivector
    assert bivector > blade
    assert refined < bivector
    assert bivector > refined
    assert refined.refines(bivector)
    assert refined.strictly_refines(bivector)
    assert refined.overlaps(bivector)


def test_subspace_comparison_lift_preserves_algebra_and_arity_boundaries() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+")
    foreign = Algebra("x+y+")
    vector = algebra.gatype.vector()
    foreign_vector = foreign.subspace.vector()
    vector_map = algebra.gatype(
        (algebra.subspace.vector(), algebra.subspace.vector())
    )

    assert not vector.refines(foreign_vector)
    assert not vector.overlaps(foreign_vector)
    assert not vector <= foreign_vector
    assert not vector >= foreign_vector
    assert not vector_map.refines(algebra.subspace.vector())
    assert not vector_map.overlaps(algebra.subspace.vector())
    assert not vector_map <= algebra.subspace.vector()
    assert not vector_map >= algebra.subspace.vector()


