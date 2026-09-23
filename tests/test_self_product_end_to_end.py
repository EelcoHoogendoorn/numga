"""Useful self-product guarantees beyond unit reverse products."""

import numpy as np

from numga import (
    Algebra,
    CliffordConjugateProductOne,
    NumpyContext,
    ReverseProductOne,
    Versor,
)


def test_paravector_composition_keeps_its_conjugate_inverse():
    algebra = Algebra("x+y-")
    mv = NumpyContext(algebra).multivector
    spaces = algebra.subspace
    unit_paravector = algebra.gatype(
        spaces.scalar() + spaces.vector(),
        (CliffordConjugateProductOne,),
    )

    # Trusted inputs with a * clifford_conjugate(a) == 1. These mix even and
    # odd grades: they are not versors, nor are their reverse products one.
    a = mv(unit_paravector, [5 / 4, 3 / 4, 0])
    b = mv(unit_paravector, [3 / 5, 0, 4 / 5])

    left_multiply = a * unit_paravector
    product = left_multiply(b)
    inverse = product.inverse()

    assert product.gatype.entails(CliffordConjugateProductOne)
    assert inverse.gatype is product.gatype
    assert not product.gatype.entails(ReverseProductOne)
    assert not product.gatype.entails(Versor)
    np.testing.assert_allclose(
        product.kernel, [3 / 4, 9 / 20, 1, 3 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        inverse.kernel, [3 / 4, -9 / 20, -1, -3 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    for identity in (product * inverse, inverse * product):
        np.testing.assert_allclose(
            (identity - mv.scalar([1])).kernel, 0,
            rtol=1e-14, atol=1e-14, equal_nan=False,
        )


def test_clifford_conjugation_is_complex_linear_on_batched_and_open_extensors():
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra, dtype=np.complex128)
    mv = context.multivector
    coefficients = np.arange(1, 17).reshape(2, 8) * (1 + 2j)
    value = mv.full(coefficients)

    conjugated = value.clifford_conjugate()

    # Grade 0 and 3 are fixed; grades 1 and 2 change sign. Imaginary scalar
    # components are untouched: this is not coefficient conjugation.
    signs = np.asarray([1, -1, -1, -1, -1, -1, -1, 1])
    np.testing.assert_array_equal(conjugated.kernel, coefficients * signs)
    np.testing.assert_array_equal(conjugated.clifford_conjugate().kernel, coefficients)

    vector = algebra.subspace.vector()
    product = vector * vector
    a = mv.vector([[1 + 2j, 2 - 1j, 0], [0, 1j, 2]])
    b = mv.vector([0, 2 + 3j, 1 - 1j])
    conjugate_first = product.clifford_conjugate()(a, b)
    bind_first = product(a, b).clifford_conjugate()
    reversed_product = b.clifford_conjugate() * a.clifford_conjugate()

    for result in (bind_first, reversed_product):
        np.testing.assert_allclose(
            conjugate_first.kernel, result.kernel,
            rtol=1e-14, atol=1e-14, equal_nan=False,
        )


def test_complex_rotor_product_uses_reversal_not_hermitian_conjugation():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra, dtype=np.complex128).multivector
    # (5/4)**2 + (3j/4)**2 == 1. This is a unit reverse product, not a
    # promise about the sum of squared absolute coefficient magnitudes.
    r = mv.rotor([5 / 4, 3j / 4])

    product = r * r
    inverse = product.inverse()

    assert product.gatype <= algebra.gatype.rotor()
    assert inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        inverse.kernel, [17 / 8, -15j / 8],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        (product * inverse - mv.scalar([1])).kernel, 0,
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
