import numpy as np
import pytest

from numga import (
    Algebra,
    NumpyContext,
    ROTOR_TRAITS,
    ReverseProductOne,
    ReverseProductZero,
    Versor,
)


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
