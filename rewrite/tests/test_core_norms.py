"""Explicit measurements establish useful facts without overriding intent."""

from fractions import Fraction

import numpy as np
import pytest

from numga import Algebra, Extensor, NumpyContext, ReverseProductNonzero, ReverseProductOne, Versor
from numga.extensions import roots


@pytest.mark.parametrize(
    "constructor, coefficients, expected",
    [
        ("scalar", [-3], [-1]),
        ("vector", [[3, 4], [5, 12]], [[3 / 5, 4 / 5], [5 / 13, 12 / 13]]),
        ("even", [[3, 4], [5, 12]], [[3 / 5, 4 / 5], [5 / 13, 12 / 13]]),
    ],
)
def test_normalized_values_are_ready_for_the_reverse_inverse(constructor, coefficients, expected):
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    value = getattr(mv, constructor)(coefficients)

    unit = value.normalized()

    assert unit.gatype.entails(ReverseProductOne)
    assert unit.gatype.entails(Versor)
    np.testing.assert_allclose(unit.kernel, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(unit.inverse().kernel, unit.reverse().kernel, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(value.kernel, coefficients)


def test_explicit_unit_measurements_and_normalization_compute_drift():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    scale = 1 + 1e-6
    r = mv.rotor(scale * np.asarray([3 / 5, 4 / 5]))

    np.testing.assert_allclose(r.norm_squared().kernel, [scale**2], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(r.norm().kernel, [scale], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(r.normalized().kernel, [3 / 5, 4 / 5], atol=1e-14, rtol=1e-14)


def test_scaled_rotor_square_root_keeps_scale_batch_and_narrow_support():
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    xy = algebra.subspace.from_masks((algebra.parse_blade("xy").mask,))
    plane = algebra.subspace.scalar() + xy
    angles = np.asarray([0, 0.2, -0.4])
    scales = np.asarray([0.25, 4, 9])
    unit = mv(plane, np.column_stack((np.cos(angles), np.sin(angles)))).with_traits(
        Versor, ReverseProductOne,
    )
    scaled = (mv.scalar(scales[:, None]) * unit).with_traits(Versor)

    assert Extensor.square_root._dispatch.resolve(unit.gatype) is roots.rotor_square_root
    assert Extensor.square_root._dispatch.resolve(scaled.gatype) is roots.scaled_rotor_square_root

    root = scaled.square_root()

    assert root.shape == (3,)
    assert root.subspace is plane
    assert root.gatype.entails((Versor, ReverseProductNonzero))
    assert not root.gatype.entails(ReverseProductOne)
    np.testing.assert_allclose((root * root).kernel, scaled.kernel, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        root.kernel,
        np.sqrt(scales)[:, None] * np.column_stack((np.cos(angles / 2), np.sin(angles / 2))),
        atol=1e-14, rtol=1e-14,
    )


def test_indefinite_measurements_remain_signed_and_real_normalization_has_a_domain():
    algebra = Algebra("x+y-")
    vector = NumpyContext(algebra).multivector.vector([0, 2])

    np.testing.assert_array_equal(vector.norm_squared().kernel, [-4])
    with np.errstate(invalid="ignore"):
        assert np.isnan(vector.norm().kernel).all()
        assert np.isnan(vector.normalized().kernel).all()

    null_algebra = Algebra("w0")
    ideal = NumpyContext(null_algebra).multivector.vector([2])
    np.testing.assert_array_equal(ideal.norm_squared().kernel, [0])
    with pytest.raises(ValueError, match="structurally zero reverse product"):
        ideal.normalized()


def test_scalar_norm_keeps_the_original_absolute_value_operation():
    algebra = Algebra("x+")
    scalar = NumpyContext(algebra, dtype=np.complex128).multivector.scalar([1j])

    np.testing.assert_array_equal(scalar.norm_squared().kernel, [-1])
    np.testing.assert_array_equal(scalar.norm().kernel, [1])
    np.testing.assert_allclose(scalar.normalized().kernel, [1], atol=1e-14, rtol=1e-14)


def test_pga_even_normalization_corrects_the_full_study_product():
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    value = mv.even([2, 1, 0, 0, 0, 0, 0, 0.5])

    squared = value.norm_squared()
    root = value.norm()
    unit = value.normalized()

    assert squared.subspace is algebra.subspace.scalar() + algebra.subspace.pseudoscalar()
    np.testing.assert_allclose(squared.kernel, [5, 2], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(squared.study_norm_squared().kernel, [25], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(squared.study_norm().kernel, [5], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((root * root - squared).kernel, 0, atol=1e-14, rtol=1e-14)
    inverse_root = squared.inverse_square_root()
    np.testing.assert_allclose((inverse_root * squared * inverse_root - 1).kernel, 0, atol=1e-14, rtol=1e-14)
    assert unit.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(unit.norm_squared().kernel, [1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((unit * unit.inverse() - mv.scalar([1])).kernel, 0, atol=1e-14, rtol=1e-14)


def test_explicit_pga_rotor_normalization_repairs_nonscalar_drift():
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    drift = 1e-6
    r = mv.rotor([1, 0, 0, 0, 0, 0, 0, drift])

    # The public measurement trusts its scalar-result carrier. Normalization
    # must nevertheless reconstruct the full product to repair its defect.
    np.testing.assert_allclose(r.norm_squared().kernel, [1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        (r * r.reverse()).kernel,
        [1, 0, 0, 0, 0, 0, 0, 2 * drift], atol=1e-14, rtol=1e-14,
    )
    repaired = r.normalized()

    assert repaired.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(repaired.kernel, [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(r.kernel, [1, 0, 0, 0, 0, 0, 0, drift])


@pytest.mark.parametrize("signature", ["x+y+z+w+", "x+y+z+w+v+"])
def test_nonnilpotent_study_root_squares_back_and_normalizes(signature):
    algebra = Algebra(signature)
    mv = NumpyContext(algebra).multivector
    # In both dimensions this is 2 + xy + 0.5 xyzw. Its reverse product
    # is 5.25 + 2 xyzw, whose nonscalar part squares to the scalar 4.
    coefficients = np.zeros(len(algebra.subspace.even()))
    for blade, coefficient in (("", 2), ("xy", 1), ("xyzw", 0.5)):
        mask = algebra.parse_blade(blade).mask if blade else 0
        coefficients[algebra.subspace.even().masks.index(mask)] = coefficient
    value = mv.even(coefficients)

    squared = value.norm_squared()
    root = value.norm()
    unit = value.normalized()

    np.testing.assert_allclose(squared.kernel[0], 5.25, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(squared.study_norm_squared().kernel, [5.25**2 - 4], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((root * root - squared).kernel, 0, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((unit * unit.reverse() - mv.scalar([1])).kernel, 0, atol=1e-14, rtol=1e-14)
    assert unit.gatype.entails(ReverseProductOne)
    assert unit.gatype.entails(Versor)


def test_exact_reverse_product_is_measured_without_a_transcendental_backend():
    algebra = Algebra("x+y+")
    vector = algebra.exact.multivector.vector([Fraction(3, 5), Fraction(4, 5)])

    assert vector.norm_squared().kernel.to_object_array().tolist() == [Fraction(1)]
    with pytest.raises(TypeError, match="sqrt"):
        vector.norm()


def test_wide_known_versor_can_still_request_explicit_renormalization():
    algebra = Algebra((6, 0, 0))
    mv = NumpyContext(algebra).multivector
    coefficients = np.zeros(len(algebra.subspace.even()))
    coefficients[0] = 1.01
    value = mv.rotor(coefficients)

    repaired = value.normalized()

    assert repaired.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(repaired.kernel, coefficients / 1.01, atol=1e-14)
