"""Explicit measurements establish useful facts without overriding intent."""


import numpy as np
import pytest

from numga import Orientation
from numga.gatype.errors import GATypeError
from numga import Algebra, NumpyContext, ReverseProductNonzero, ReverseProductOne, Versor


def test_scaled_rotor_square_root_keeps_scale_batch_and_narrow_support():
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    xy = algebra.subspace.from_masks((algebra.parse_blade("xy").mask,))
    plane = algebra.subspace.scalar() + xy
    angles = np.asarray([0, 0.2, -0.4])
    scales = np.asarray([0.25, 4, 9])
    unit = mv(algebra.gatype(plane, orientation=Orientation.extrinsic), np.column_stack((np.cos(angles), np.sin(angles)))).with_traits(
        Versor, ReverseProductOne,
    )
    scaled = (mv.scalar(scales[:, None]) * unit).with_traits(Versor)


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
    np.testing.assert_allclose((root * root - squared).kernel, 0, atol=1e-13, rtol=1e-13)
    inverse_root = squared.inverse_square_root()
    np.testing.assert_allclose((inverse_root * squared * inverse_root - 1).kernel, 0, atol=1e-13, rtol=1e-13)
    assert unit.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(unit.norm_squared().kernel, [1], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose((unit * unit.inverse() - mv.scalar([1])).kernel, 0, atol=1e-13, rtol=1e-13)


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
    # In both dimensions this is 2 + mv.xy + 0.5 * mv.xyzw. Its reverse product
    # is 5.25 + 2 * mv.xyzw, whose nonscalar part squares to the scalar 4.
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
    np.testing.assert_allclose((root * root - squared).kernel, 0, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose((unit * unit.reverse() - mv.scalar([1])).kernel, 0, atol=1e-13, rtol=1e-13)
    assert unit.gatype.entails(ReverseProductOne)
    assert unit.gatype.entails(Versor)


def test_wide_known_versor_can_still_request_explicit_renormalization():
    algebra = Algebra((6, 0, 0))
    mv = NumpyContext(algebra).multivector
    coefficients = np.zeros(len(algebra.subspace.even()))
    coefficients[0] = 1.01
    value = mv.rotor(coefficients)

    repaired = value.normalized()

    assert repaired.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(repaired.kernel, coefficients / 1.01, atol=1e-14)


def test_scalar_norm_squared_is_the_scalar_part_of_the_reverse_product_for_values_and_forms():
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    Bivector = algebra.gatype.bivector()
    # A screw: the reverse product has a pseudoscalar part, which the scalar norm leaves out.
    bivectors = mv.bivector(np.random.default_rng(3).normal(size=(4, 6)))
    product = bivectors * bivectors.reverse()

    squared = bivectors.scalar_norm_squared()

    assert squared.subspace is algebra.subspace.scalar()
    np.testing.assert_allclose(squared.kernel, product.select[0].kernel, atol=1e-13)
    # The bilinear form evaluated on a bivector twice gives its squared norm.
    form = (mv.rotor() >> Bivector).scalar_norm_squared()
    np.testing.assert_allclose(form(bivectors, bivectors).kernel, squared.kernel, atol=1e-13)


@pytest.mark.parametrize("signature, grade", [("x+y+z+", 2), ("x+y+z+w+", 3), ("x+y+z+w+", 2), ("x+y+z+w+v+", 4)])
def test_dot_is_the_gram_determinant_of_the_factors_on_every_grade(signature, grade):
    """For blades `a1 ^ ... ^ ak` and `b1 ^ ... ^ bk` the dot product is det(ai | bj): the vector metric
    extended to every grade, positive on a blade with itself, as a value and as a form."""
    algebra = Algebra(signature)
    mv = NumpyContext(algebra).multivector
    rng = np.random.default_rng(grade)
    left, right = rng.normal(size=(2, grade, algebra.dimension))
    a, b = mv.vector(left[0]), mv.vector(right[0])
    for i in range(1, grade):
        a, b = a ^ mv.vector(left[i]), b ^ mv.vector(right[i])

    np.testing.assert_allclose(a.dot(b).to_array(), np.linalg.det(left @ right.T), rtol=1e-10)
    np.testing.assert_allclose(a.dot(a).to_array(), a.scalar_norm_squared().to_array(), rtol=1e-12)
    assert a.dot(a).to_array() > 0
    Blade = algebra.gatype.oriented((a.subspace,), (a.gatype.orientation,))
    np.testing.assert_allclose(Blade.dot(Blade)(a, b).to_array(), a.dot(b).to_array(), rtol=1e-12)


@pytest.mark.parametrize("signature, blade, scale, reverse_product, dot", [
    ("x+y+z+t-", "x", 3, -9, 9),
    ("x+y+z+t-", "t", 1, 1, -1),
    ("x+y+z+", "x", 3, 9, 9),
    ("x+y+z+w0", "x", 3, 0, 9),
    ("x+y+z+w0", "w", 1, 1, 0),
])
def test_a_point_measured_by_its_reverse_product_and_by_its_dot(signature, blade, scale, reverse_product, dot):
    """A point is an antivector, the dual of the plane it lies in, and intrinsic: its reverse product measures it by
    the metric on antivectors, which in a spacetime is opposite to the metric on vectors and in a projective algebra
    reads only its weight, while its dot reads it through its dual, the vector."""
    mv = NumpyContext(Algebra(signature)).multivector
    point = (scale * getattr(mv, blade)).dual()

    assert point.gatype.orientation is Orientation.intrinsic
    np.testing.assert_allclose(point.scalar_norm_squared().to_array(), reverse_product)
    np.testing.assert_allclose(point.dot(point).to_array(), dot)
    with np.errstate(invalid="ignore"):
        np.testing.assert_allclose(point.antinorm().to_array(), np.sqrt(dot))


@pytest.mark.parametrize("signature", ["x+y+z+t-", "x+y+z+", "x+y+z+w0", "x+y+w0", "x+y-z-"])
def test_the_dot_of_intrinsic_values_is_the_dot_of_their_duals_as_a_value_and_as_a_form(signature):
    algebra = Algebra(signature)
    mv = NumpyContext(algebra).multivector
    Multivector = algebra.gatype.multivector()
    Intrinsic = Multivector.derive.flipped_orientation
    a, b = mv(Intrinsic, np.random.default_rng(5).normal(size=(2, 4, len(Intrinsic.output_subspace))))

    np.testing.assert_allclose(a.dot(b).to_array(), a.dual().dot(b.dual()).to_array(), atol=1e-13)
    np.testing.assert_allclose(a.dot(b).to_array(), b.dot(a).to_array(), atol=1e-13)
    np.testing.assert_allclose(Intrinsic.dot(Intrinsic)(a, b).to_array(), a.dot(b).to_array(), atol=1e-13)


def test_an_extrinsic_value_and_an_intrinsic_one_have_no_dot():
    mv = NumpyContext(Algebra("x+y+z+")).multivector
    plane, point = mv.x, mv.y.dual()

    with pytest.raises(GATypeError):
        plane.dot(point)
    np.testing.assert_allclose((plane & point).to_array(), 0.0, atol=1e-15)


def test_in_the_algebra_of_space_both_readings_agree_on_antivectors():
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    Antivector = algebra.gatype.antivector()
    Flipped = Antivector.derive.flipped_orientation
    coefficients = np.random.default_rng(6).normal(size=(2, 5, 3))
    a, b = mv(Antivector, coefficients)
    extrinsic_a, extrinsic_b = mv(Flipped, coefficients)

    np.testing.assert_allclose(a.dot(b).to_array(), extrinsic_a.dot(extrinsic_b).to_array(), atol=1e-13)
    np.testing.assert_allclose(Antivector.dot(Antivector)(a, b).to_array(), a.dot(b).to_array(), atol=1e-13)
    np.testing.assert_allclose(a.antinorm().to_array(), a.norm().to_array(), atol=1e-13)


@pytest.mark.parametrize(
    "signature, grade", [("x+y+z+t-", grade) for grade in range(5)] + [("x+y+t-", grade) for grade in range(4)],
)
def test_in_a_spacetime_the_two_readings_differ_by_the_antiscalar_reverse_product_on_every_grade(signature, grade):
    algebra = Algebra(signature)
    mv = NumpyContext(algebra).multivector
    KVector = algebra.gatype.k_vector(grade)
    Flipped = KVector.derive.flipped_orientation
    coefficients = np.random.default_rng(grade).normal(size=(2, 4, len(KVector.output_subspace)))
    a, b = mv(KVector, coefficients)
    flipped_a, flipped_b = mv(Flipped, coefficients)
    antiscalar = mv.antiscalar([1])

    np.testing.assert_allclose(antiscalar.scalar_norm_squared().to_array(), -1)
    np.testing.assert_allclose(
        flipped_a.dot(flipped_b).to_array(), (antiscalar.scalar_norm_squared() * a.dot(b)).to_array(), atol=1e-13,
    )


def test_a_projective_point_has_its_weight_as_norm_and_its_bulk_as_antinorm():
    """A point of weight w at position p is the dual of the plane `w * (p + mv.w)`: its norm is |w|, its antinorm the
    bulk |w| |p|, and their quotient the distance from the origin."""
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    rng = np.random.default_rng(7)
    positions, weights = rng.normal(size=(6, 3)), rng.uniform(0.5, 2, size=6) * rng.choice([-1, 1], size=6)
    points = (mv.vector(np.column_stack((positions, np.ones(6)))) * weights).dual()

    np.testing.assert_allclose(points.norm().to_array(), np.abs(weights), atol=1e-13)
    np.testing.assert_allclose(points.antinorm().to_array(), np.abs(weights) * np.linalg.norm(positions, axis=-1), atol=1e-13)
    np.testing.assert_allclose(
        (points.antinorm() / points.norm()).to_array(), np.linalg.norm(positions, axis=-1), atol=1e-13,
    )


def test_the_dot_of_an_event_is_its_interval_and_the_antinorm_of_a_spatial_point_its_distance():
    algebra = Algebra("x+y+z+t-")
    mv = NumpyContext(algebra).multivector
    rng = np.random.default_rng(8)
    positions, times = rng.normal(size=(5, 3)), rng.normal(size=5)
    events = mv.vector(np.column_stack((positions, times))).dual()
    spatial = mv.vector(np.column_stack((positions, np.zeros(5)))).dual()

    np.testing.assert_allclose(events.dot(events).to_array(), (positions ** 2).sum(-1) - times ** 2, atol=1e-13)
    np.testing.assert_allclose(spatial.antinorm().to_array(), np.linalg.norm(positions, axis=-1), atol=1e-13)
