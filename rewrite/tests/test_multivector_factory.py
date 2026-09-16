import numpy as np
import pytest

from numga import (
    Algebra,
    Extensor,
    MultivectorFactory,
    NumpyContext,
    ROTOR_TRAITS,
    ReverseProductOne,
    ReverseProductZero,
    Versor,
)


def test_named_constructor_builds_batched_nullary_extensor_in_its_context():
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra)

    vectors = context.multivector.vector([[1, 2, 3], [4, 5, 6]])

    assert context.multivector is context.multivector
    assert isinstance(context.multivector, MultivectorFactory)
    assert isinstance(vectors, Extensor)
    assert vectors.context is context
    assert vectors.gatype is algebra.gatype.vector()
    assert vectors.arity == 0
    assert vectors.shape == (2,)
    np.testing.assert_array_equal(vectors.kernel, [[1, 2, 3], [4, 5, 6]])


def test_exact_context_exposes_the_same_nullary_construction_namespace():
    algebra = Algebra("x+y+")

    vector = algebra.exact.multivector.vector([1, 2])

    assert vector.context is algebra.exact
    assert vector.gatype is algebra.gatype.vector()
    assert vector.kernel.to_object_array().tolist() == [1, 2]


def test_named_constructors_distinguish_plain_support_from_semantic_type():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)

    even = context.multivector.even([1, 0])
    rotor = context.multivector.rotor([1, 0])

    assert even.output_subspace is rotor.output_subspace
    assert even.gatype is algebra.gatype.even()
    assert not even.gatype.traits
    assert rotor.gatype is algebra.gatype.rotor()
    assert rotor.gatype.traits == ROTOR_TRAITS


def test_omitted_coefficients_are_the_projected_multiplicative_unit():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)

    scalar = context.multivector.scalar()
    vector = context.multivector.vector()
    even = context.multivector.even()
    rotor = context.multivector.rotor()
    by_grade = context.multivector.k_vector(1)
    by_masks = context.multivector.from_masks((0, 3))

    np.testing.assert_array_equal(scalar.kernel, [1])
    np.testing.assert_array_equal(vector.kernel, [0, 0])
    np.testing.assert_array_equal(even.kernel, [1, 0])
    np.testing.assert_array_equal(rotor.kernel, [1, 0])
    np.testing.assert_array_equal(by_grade.kernel, [0, 0])
    np.testing.assert_array_equal(by_masks.kernel, [1, 0])
    assert rotor.gatype is algebra.gatype.rotor()
    assert even.gatype is algebra.gatype.rotor()
    assert scalar.gatype.entails(Versor)
    assert scalar.gatype.entails(ReverseProductOne)
    assert vector.gatype.entails(ReverseProductZero)
    assert by_grade.gatype.entails(ReverseProductZero)


def test_omitted_coefficients_reject_contradicting_asserted_traits():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    unit_claimed_zero = algebra.gatype(
        algebra.subspace.even(),
        (ReverseProductZero,),
    )
    zero_claimed_unit = algebra.gatype(
        algebra.subspace.vector(),
        (ReverseProductOne,),
    )

    with pytest.raises(ValueError, match="omitted coefficients.*contradict"):
        context.multivector(unit_claimed_zero)
    with pytest.raises(ValueError, match="omitted coefficients.*contradict"):
        context.multivector(zero_claimed_unit)


def test_coefficients_are_not_checked_against_asserted_traits():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)

    asserted_rotor = context.multivector.rotor([0, 0])

    assert asserted_rotor.gatype is algebra.gatype.rotor()
    np.testing.assert_array_equal(asserted_rotor.kernel, [0, 0])


def test_generic_and_parameterized_construction_preserve_complete_type():
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra)
    bivector = algebra.subspace.bivector()
    refined = algebra.gatype.rotor()

    structural = context.multivector(bivector, [1, 2, 3])
    semantic = context.multivector(refined, [1, 0, 0, 0])
    by_grade = context.multivector.k_vector(2, [4, 5, 6])
    by_masks = context.multivector.from_masks((1, 4), [7, 8])

    assert structural.gatype is algebra.gatype.bivector()
    assert semantic.gatype is refined
    assert by_grade.gatype is algebra.gatype.k_vector(2)
    assert by_masks.gatype is algebra.gatype.from_masks((1, 4))


def test_multivector_namespace_rejects_positive_arity_only():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    vector = algebra.subspace.vector()
    unary_type = algebra.gatype((vector, vector))
    coefficients = [[1, 0], [0, 1]]

    with pytest.raises(ValueError, match="arity-0 GAType"):
        context.multivector(unary_type, coefficients)

    unary = context.extensor(unary_type, coefficients)
    assert unary.gatype is unary_type
    assert unary.arity == 1


def test_multivector_namespace_constructs_basis_blades_and_rejects_unknown():
    algebra = Algebra("x+y+")
    namespace = NumpyContext(algebra).multivector

    assert {"vector", "k_vector", "from_masks", "even", "rotor"} <= set(
        dir(namespace)
    )
    xy = getattr(namespace, "xy")
    assert xy.subspace == algebra.subspace.from_blades("xy")
    np.testing.assert_array_equal(xy.kernel, [1.0])

    with pytest.raises(AttributeError):
        getattr(namespace, "z")
    with pytest.raises(AttributeError):
        getattr(namespace, "nonexistent")
