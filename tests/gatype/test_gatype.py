from __future__ import annotations

import pickle

import pytest

from numga import Algebra

from numga.gatype import (
    CoefficientOrthogonal,
    EMPTY_TRAITS,
    ROTOR_TRAITS,
    GAType,
    GATypeFactory,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    ReverseProductZero,
    TraitSet,
    Versor,
)
from numga.subspace import SubSpaceFactory


def test_trait_stub_stores_only_explicit_typed_facts() -> None:
    assert TraitSet() == EMPTY_TRAITS
    assert TraitSet(()) == EMPTY_TRAITS
    assert not EMPTY_TRAITS
    assert tuple(EMPTY_TRAITS) == ()
    assert TraitSet((Versor, Versor, ReverseProductOne)) == ROTOR_TRAITS
    assert Versor in ROTOR_TRAITS
    assert ReverseProductOne in ROTOR_TRAITS
    assert pickle.loads(pickle.dumps(ROTOR_TRAITS)) == ROTOR_TRAITS
    with pytest.raises(TypeError, match="Trait values"):
        TraitSet(("unit",))
    with pytest.raises(AttributeError, match="immutable"):
        EMPTY_TRAITS.traits = ()


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


def test_gatype_factory_forwards_subspace_constructors_as_plain_types() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    subspaces = algebra.subspace
    gatypes = algebra.gatype

    for name in (
        "empty",
        "scalar",
        "vector",
        "full",
        "even",
        "multivector",
        "even_grade",
    ):
        actual = getattr(gatypes, name)()
        expected_subspace = getattr(subspaces, name)()
        assert actual is gatypes(expected_subspace)
        assert actual.subspaces == (expected_subspace,)
        assert actual.traits is EMPTY_TRAITS

    assert gatypes.k_vector(2) is gatypes(subspaces.k_vector(2))
    assert gatypes.from_masks((0, 3)) is gatypes(subspaces.from_masks((0, 3)))
    assert "even" in dir(gatypes)


def test_rotor_is_an_explicit_refinement_of_plain_even_support() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+z+")
    even = algebra.gatype.even()
    rotor = algebra.gatype.rotor()

    assert even.output_subspace is algebra.subspace.even()
    assert even.traits is EMPTY_TRAITS
    assert rotor is algebra.gatype.rotor()
    assert rotor is algebra.gatype(even.output_subspace, ROTOR_TRAITS)
    assert rotor.output_subspace is even.output_subspace
    assert rotor.traits == ROTOR_TRAITS
    assert rotor != even


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


def test_direct_gatype_can_compare_with_a_subspace_without_a_bound_factory() -> None:
    algebra = Algebra((3, 0, 0))
    vector = SubSpaceFactory(algebra).vector()
    gatype = GAType((vector,))

    assert gatype <= vector
    assert gatype >= vector


def test_stub_value_traits_are_rejected_on_positive_arity_types() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+")
    even = algebra.subspace.even()

    with pytest.raises(ValueError, match="arity 0"):
        algebra.gatype((even, even), (Versor,))


def test_coefficient_orthogonal_requires_a_square_unary_carrier() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+")
    scalar = algebra.subspace.scalar()
    vector = algebra.subspace.vector()

    orthogonal = algebra.gatype(
        (vector, vector),
        (CoefficientOrthogonal,),
    )
    assert orthogonal.entails(CoefficientOrthogonal)
    with pytest.raises(ValueError, match="equally sized"):
        algebra.gatype(
            (scalar, vector),
            (CoefficientOrthogonal,),
        )


def test_lift_accepts_exactly_one_local_subspace() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+")
    other = Algebra("x+y+")
    vector = algebra.subspace.vector()

    assert algebra.gatype.lift(vector) is algebra.gatype.vector()
    with pytest.raises(TypeError, match="one SubSpace"):
        algebra.gatype.lift((vector, vector))
    with pytest.raises(ValueError, match="another algebra"):
        algebra.gatype.lift(other.subspace.vector())
    with pytest.raises(AttributeError):
        getattr(algebra.gatype, "not_a_constructor")


def test_gatype_is_an_output_first_flyweight() -> None:
    algebra = Algebra((3, 0, 0))
    subspace = SubSpaceFactory(algebra)
    gatypes = GATypeFactory(algebra)
    output = subspace.vector()
    left = subspace.scalar()
    right = subspace.k_vector(2)

    gatype = gatypes((output, left, right))

    assert gatype is gatypes([output, left, right], TraitSet())
    assert gatype.subspaces == (output, left, right)
    assert gatype.axes is gatype.subspaces
    assert gatype.algebra is algebra
    assert gatype.output_subspace is output
    assert gatype.input_subspaces == (left, right)
    assert gatype.arity == 2
    assert gatype.traits is EMPTY_TRAITS
    assert gatype.representation_key == (gatype.subspaces, EMPTY_TRAITS)


def test_direct_gatype_construction_is_structurally_correct_but_not_interned() -> None:
    algebra = Algebra((3, 0, 0))
    output = SubSpaceFactory(algebra).vector()

    left = GAType((output,))
    right = GAType((output,))

    assert left == right
    assert hash(left) == hash(right)
    assert left is not right


def test_algebra_owns_the_canonical_gatype_factory() -> None:
    from numga.algebra import Algebra

    algebra = Algebra("x+y+")
    even = algebra.subspace.even()

    left = algebra.gatype((even,))
    right = algebra.gatype([even], TraitSet())

    assert left is right
    assert left == GAType((even,))
    assert algebra.gatype.flyweight_count == 1


def test_nullary_gatype_is_not_a_special_case() -> None:
    subspace = SubSpaceFactory(Algebra((3, 0, 0))).full()
    gatype = GAType((subspace,))

    assert gatype.output_subspace is subspace
    assert gatype.input_subspaces == ()
    assert gatype.arity == 0


def test_gatype_rejects_empty_or_non_subspace_axes() -> None:
    with pytest.raises(ValueError, match="at least an output"):
        GAType(())
    with pytest.raises(TypeError, match="every GAType axis"):
        GAType((object(),))


def test_gatype_rejects_axes_from_different_algebra_objects() -> None:
    left = SubSpaceFactory(Algebra((3, 0, 0))).vector()
    right = SubSpaceFactory(Algebra((3, 0, 0))).vector()

    with pytest.raises(ValueError, match="same algebra"):
        GAType((left, right))


def test_gatype_is_immutable() -> None:
    gatype = GAType((SubSpaceFactory(Algebra((3, 0, 0))).scalar(),))
    with pytest.raises(AttributeError, match="immutable"):
        gatype.subspaces = ()
