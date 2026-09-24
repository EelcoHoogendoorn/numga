"""Inverses from useful geometric expressions, including the exact boundary."""


import numpy as np
import pytest

from numga import Algebra, NumpyContext, ReverseProductNonzero


def test_statically_empty_inverse_raises():
    algebra = Algebra((5, 0, 0))
    empty = NumpyContext(algebra).multivector.empty(np.empty((2, 0)))
    with pytest.raises(ZeroDivisionError, match="statically null"):
        empty.inverse()


def test_scalar_reciprocal_does_not_square_large_coefficients():
    algebra = Algebra("x+y+")
    scalar = NumpyContext(algebra).multivector.scalar([[1e300], [-1e300]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        inverse = scalar.inverse()
    np.testing.assert_allclose(inverse.kernel, [[1e-300], [-1e-300]], atol=0)
    assert inverse.gatype.entails(ReverseProductNonzero)


@pytest.mark.parametrize("signature", ["x+y+z+w0", "x+y+z+w0v+"])
def test_general_inverse_of_a_mixed_multivector_is_two_sided(signature):
    algebra = Algebra(signature)
    coefficients = [
        3, .2, -.3, .1, .4, .1, -.1, .2, -.2, .3, -.4, .1, .2, .1, -.3, .2
    ] + [.1] * (algebra.blade_count - 16)
    value = NumpyContext(algebra).multivector.full(coefficients)
    inverse = value.inverse()
    for product in (value * inverse, inverse * value):
        np.testing.assert_allclose(
            product.kernel, [1] + [0] * (algebra.blade_count - 1), atol=1e-13
        )


def test_non_square_and_singular_maps_fail_without_pseudoinverse():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    scalar, vector = algebra.subspace.scalar(), algebra.subspace.vector()
    nonsquare = context.extensor(algebra.gatype((scalar, vector)), [[1, 2]])
    with pytest.raises(LookupError, match="no .* implementation"):
        nonsquare.inverse()
    singular = context.extensor(algebra.gatype((vector, vector)), [[1, 2], [2, 4]])
    with pytest.raises(np.linalg.LinAlgError):
        singular.inverse()


@pytest.mark.parametrize("grade", [2, 3])
def test_5d_grade_inverses_keep_their_narrow_carriers(grade):
    algebra = Algebra((5, 0, 0))
    space = algebra.subspace.k_vector(grade)
    value = algebra.exact.extensor(space, [3, 1, -2, 4, 0, 1, -1, 2, -3, 1])
    inverse = value.inverse()
    assert inverse.subspace is space
    for product in (value * inverse, inverse * value):
        expected = [1] + [0] * (len(product.subspace) - 1)
        np.testing.assert_allclose(product.kernel.values, expected, atol=1e-14)
