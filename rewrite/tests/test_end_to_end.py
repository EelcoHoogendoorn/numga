"""Executable examples of the intended zero-boilerplate Numga surface.

Passing tests are the current public contract.  Strict xfails are executable
acceptance criteria: implementing one without removing its marker is a failure.

This is the bounded behavioral target for the trait machinery: construction
and normalization, exp/log/norm, product staging, versor sandwiches, map inverse,
and collections. Tests request useful facts, without prescribing a relation
language or a rule-registration API. Autodiff and signed layouts come later.
Traits let consumers trust declared preconditions. Explicit norm measurements
and normalization still compute from coefficients; the caller controls drift.
"""

import numpy as np
import pytest

from numga import (
    Algebra,
    CoefficientOrthogonal,
    NumpyContext,
    ReverseProductNonzero,
    ReverseProductOne,
    ReverseProductScalar,
    Versor,
)


def test_composed_rotors_keep_their_type_and_select_the_unit_inverse():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    rotor = algebra.gatype.rotor()
    i = context.multivector.rotor([0, 1])

    product = (rotor * rotor * rotor)(i, i, i)
    inverse = product.inverse()

    assert product.gatype is rotor
    assert inverse.gatype is rotor
    np.testing.assert_array_equal(product.kernel, [0, -1])
    np.testing.assert_array_equal(inverse.kernel, [0, 1])


def test_batched_rotor_products_broadcast_without_losing_inverse_dispatch():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    rotor = algebra.gatype.rotor()
    left = context.multivector.rotor(
        [[[1, 0]], [[0, 1]]],
    )
    right = context.multivector.rotor(
        [[[1, 0], [0, 1], [-1, 0]]],
    )

    product = left * right
    inverse = product.inverse()
    identity = product * inverse

    assert product.shape == (2, 3)
    assert product.gatype is rotor
    assert inverse.gatype is rotor
    assert identity.gatype is rotor
    np.testing.assert_array_equal(
        identity.kernel,
        np.broadcast_to([1, 0], (2, 3, 2)),
    )


def test_partial_product_application_keeps_inverse_through_batch_operations():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    rotor = algebra.gatype.rotor()
    quarter_turn = mv.rotor([0, 1])
    inputs = mv.rotor([[1, 0], [0, 1]])

    left_multiply = (rotor * rotor).bind({0: quarter_turn})
    products = left_multiply(inputs)
    selected = products.reshape(2, 1).broadcast_to((2, 3))[:, 1]

    assert left_multiply.arity == 1
    assert selected.gatype <= rotor
    np.testing.assert_allclose(
        selected.inverse().kernel, [[0, -1], [-1, 0]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_negating_a_rotor_keeps_its_unit_inverse():
    algebra = Algebra("x+y+")
    r = NumpyContext(algebra).multivector.rotor([3 / 5, 4 / 5])

    negative = -r

    assert negative.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        negative.inverse().kernel, [-3 / 5, 4 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_identity_map_keeps_a_rotor_ready_for_inverse():
    algebra = Algebra("x+y+")
    r = NumpyContext(algebra).multivector.rotor([3 / 5, 4 / 5])

    identity = algebra.operator.identity(algebra.subspace.even())
    unchanged = identity(r)

    assert unchanged.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        unchanged.inverse().kernel, [3 / 5, -4 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_identity_keeps_an_open_rotation_map():
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    rotor = mv.rotor([3 / 5, 4 / 5, 0, 0])
    rotation = rotor.sandwich(algebra.subspace.vector())

    unchanged = algebra.operator.identity(algebra.subspace.vector())(rotation)

    assert unchanged.gatype is rotation.gatype
    np.testing.assert_allclose(unchanged.kernel, rotation.kernel)
    np.testing.assert_allclose(unchanged.inverse().kernel, rotation.kernel.T)


def test_reversal_before_or_after_binding_keeps_the_same_useful_facts():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    rotor = algebra.gatype.rotor()
    a = mv.rotor([3 / 5, 4 / 5])
    b = mv.rotor([5 / 13, 12 / 13])
    product = rotor * rotor

    reverse_first = product.reverse()(a, b)
    bind_first = product(a, b).reverse()

    assert reverse_first.gatype <= rotor
    assert bind_first.gatype <= rotor
    for result in (reverse_first, bind_first):
        np.testing.assert_allclose(
            result.kernel, [-33 / 65, -56 / 65],
            rtol=1e-14, atol=1e-14, equal_nan=False,
        )
        np.testing.assert_allclose(
            result.inverse().kernel, [-33 / 65, 56 / 65],
            rtol=1e-14, atol=1e-14, equal_nan=False,
        )


def test_averaging_rotors_does_not_promise_a_unit_or_invertible_result():
    algebra = Algebra("x+y+")
    rotors = NumpyContext(algebra).multivector.rotor([[1, 0], [-1, 0]])

    average = rotors.mean(axis=0)

    assert not average.gatype.entails(ReverseProductOne)
    assert not average.gatype.entails(ReverseProductNonzero)
    assert not average.gatype.entails(Versor)
    np.testing.assert_array_equal(average.kernel, [0, 0])


def test_batched_sandwich_maps_broadcast_like_the_direct_expression():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    vector = algebra.subspace.vector()
    rotors = context.multivector.rotor(
        [[[1, 0]], [[0, 1]]],
    )
    points = context.multivector.vector(
        [[[2, 3], [-1, 4], [0.5, -2]]],
    )

    rotations = rotors.sandwich(vector)
    via_maps = rotations(points)
    direct = rotors.sandwich(points)

    assert rotations.shape == (2, 1)
    assert via_maps.shape == (2, 3)
    assert via_maps.gatype is algebra.gatype.vector()
    np.testing.assert_allclose(
        via_maps.kernel,
        [
            [[2, 3], [-1, 4], [0.5, -2]],
            [[-2, -3], [1, -4], [-0.5, 2]],
        ],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        via_maps.kernel,
        direct.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_normalized_even_values_become_rotors_ready_for_inverse():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    values = context.multivector.even([[3.0, 4.0], [5.0, 12.0]])

    unit = values.normalized()
    inverse = unit.inverse()

    assert unit.gatype <= algebra.gatype.rotor()
    assert inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        unit.kernel,
        [[3 / 5, 4 / 5], [5 / 13, 12 / 13]],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        inverse.kernel,
        unit.reverse().kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_explicit_normalization_repairs_drift_even_when_unit_is_declared():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    scale = 1 + 1e-6
    coefficients = scale * np.asarray([3 / 5, 4 / 5])
    drifted = mv.rotor(coefficients)

    repaired = drifted.normalized()

    assert repaired.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        repaired.kernel, [3 / 5, 4 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        repaired.inverse().kernel, [3 / 5, -4 / 5],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_array_equal(drifted.kernel, coefficients)


def test_unit_inverse_trusts_the_input_without_defensive_normalization():
    algebra = Algebra("x+y+")
    scale = 1 + 1e-6
    r = NumpyContext(algebra).multivector.rotor(
        scale * np.asarray([3 / 5, 4 / 5]),
    )

    inverse = r.inverse()

    # This overload requires unit input and trusts that promise. The caller
    # can request r.normalized().inverse() when drift correction is wanted.
    assert inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        inverse.kernel, scale * np.asarray([3 / 5, -4 / 5]),
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_normalized_vectors_get_unit_inverse_without_rotor_construction():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    vectors = mv.vector([[3, 4], [5, 12]])

    # Every vector has a scalar reverse product; its support proves no value.
    assert vectors.gatype.entails(ReverseProductScalar)
    assert not vectors.gatype.entails(ReverseProductNonzero)
    assert not vectors.gatype.entails(Versor)

    unit = vectors.normalized()
    inverse = unit.inverse()

    assert unit.gatype <= algebra.subspace.vector()
    assert unit.gatype.entails(Versor)
    assert unit.gatype.entails(ReverseProductOne)
    np.testing.assert_allclose(
        inverse.kernel, [[3 / 5, 4 / 5], [5 / 13, 12 / 13]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        (unit * inverse).kernel, [[1, 0], [1, 0]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_scalar_inverse_uses_structure_without_constructor_numeric_inference():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    scalar = mv.scalar([-3])

    assert scalar.gatype is mv.scalar([0]).gatype
    assert scalar.gatype.entails(ReverseProductScalar)
    assert not scalar.gatype.entails(ReverseProductNonzero)

    reciprocal = scalar.inverse()

    assert reciprocal.gatype <= algebra.subspace.scalar()
    np.testing.assert_allclose(
        reciprocal.kernel, [-1 / 3],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        (scalar * reciprocal).kernel, [1],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_vector_product_norms_factor_even_for_negative_and_null_norms():
    algebra = Algebra("x+y-z+w+v+")
    mv = NumpyContext(algebra).multivector
    a = mv.vector([
        [2, 1, 0, 0, 0], [1, 2, 0, 0, 0],
        [1, 1, 0, 0, 0], [0, 0, 0, 0, 0],
    ])
    b = mv.vector([3, -1, 0, 0, 0])

    product = a * b

    assert a.gatype.entails(ReverseProductScalar)
    assert product.gatype.entails(ReverseProductScalar)
    # In 5D this carrier also contains values with nonscalar reverse norms.
    # The useful fact must follow from the product, not just its output support.
    assert not algebra.gatype(product.subspace).entails(ReverseProductScalar)
    assert not product.gatype.entails(ReverseProductNonzero)
    squared_norm = product.norm_squared()
    factored = a.norm_squared() * b.norm_squared()

    assert squared_norm.gatype <= algebra.subspace.scalar()
    np.testing.assert_allclose(
        squared_norm.kernel, [[24], [-24], [0], [0]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        squared_norm.kernel, factored.kernel,
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_explicit_norms_measure_coefficients_even_when_unit_is_declared():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    scale = 1 + 1e-6
    left = context.multivector.rotor(scale * np.asarray([3 / 5, 4 / 5]))
    right = context.multivector.rotor([5 / 13, 12 / 13])

    product = left * right
    squared_norm = product.norm_squared()
    norm = product.norm()

    assert product.gatype is algebra.gatype.rotor()
    assert squared_norm.subspace is algebra.subspace.scalar()
    assert norm.subspace is algebra.subspace.scalar()
    np.testing.assert_allclose(
        squared_norm.kernel,
        [scale**2],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        norm.kernel,
        [scale],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


@pytest.mark.parametrize(
    ("description", "scalar_part", "bivector_part"),
    [
        pytest.param("x+y+", np.cos, np.sin, id="rotation"),
        pytest.param("x+y-", np.cosh, np.sinh, id="boost"),
        pytest.param("x+w0", np.ones_like, np.positive, id="translation"),
    ],
)
def test_batched_bivector_exp_returns_rotors_ready_for_inverse(
    description, scalar_part, bivector_part,
):
    algebra = Algebra(description)
    context = NumpyContext(algebra)
    angles = np.asarray([-0.5, 0.0, 0.25])
    bivectors = context.multivector.bivector(angles[:, None])

    rotors = bivectors.exp()
    inverses = rotors.inverse()

    assert rotors.shape == angles.shape
    assert rotors.gatype <= algebra.gatype.rotor()
    assert inverses.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        rotors.kernel,
        np.stack((scalar_part(angles), bivector_part(angles)), axis=-1),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        inverses.kernel,
        np.stack((scalar_part(angles), -bivector_part(angles)), axis=-1),
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    )


@pytest.mark.parametrize(
    ("description", "scalar_part", "bivector_part"),
    [
        pytest.param("x+y+", np.cos, np.sin, id="rotation"),
        pytest.param("x+y-", np.cosh, np.sinh, id="boost"),
        pytest.param("x+w0", np.ones_like, np.positive, id="translation"),
    ],
)
def test_rotor_log_exp_round_trip_preserves_rotor_facts(
    description, scalar_part, bivector_part,
):
    algebra = Algebra(description)
    context = NumpyContext(algebra)
    angles = np.asarray([-0.4, 0.0, 0.3])
    # A local log branch around the identity; no global log(exp(B)) promise.
    rotors = context.multivector.rotor(
        np.stack((scalar_part(angles), bivector_part(angles)), axis=-1),
    )

    bivectors = rotors.log()
    round_trip = bivectors.exp()

    assert bivectors.gatype <= algebra.subspace.bivector()
    assert round_trip.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        bivectors.kernel,
        angles[:, None],
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    )
    for norm in (round_trip.norm(), round_trip.norm_squared()):
        assert norm.gatype <= algebra.subspace.scalar()
        np.testing.assert_allclose(
            norm.kernel, np.ones((3, 1)),
            rtol=1e-10, atol=1e-10, equal_nan=False,
        )
    np.testing.assert_allclose(
        round_trip.kernel,
        rotors.kernel,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=False,
    )


def test_rotor_sandwich_builds_an_orthogonal_map_with_a_fast_inverse():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    vector = algebra.subspace.vector()
    half_angle = 0.35
    rotor = context.multivector.rotor(
        [np.cos(half_angle), np.sin(half_angle)],
    )
    vectors = context.multivector.vector(
        [[1.0, 0.0], [0.0, 1.0], [2.0, -3.0]],
    )

    rotation = rotor.sandwich(vector)
    inverse = rotation.inverse()
    round_trip = inverse(rotation(vectors))

    assert rotation.axes == (vector, vector)
    assert rotation.gatype.entails(CoefficientOrthogonal)
    assert inverse.gatype.entails(CoefficientOrthogonal)
    np.testing.assert_allclose(
        inverse.kernel,
        rotation.kernel.T,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        round_trip.kernel,
        vectors.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_composed_rotation_maps_retain_their_fast_inverse():
    algebra = Algebra("x+y+")
    context = NumpyContext(algebra)
    vector = algebra.subspace.vector()
    first_rotor = context.multivector.rotor([np.cos(0.2), np.sin(0.2)])
    second_rotor = context.multivector.rotor([np.cos(-0.4), np.sin(-0.4)])
    vectors = context.multivector.vector([[1.0, 2.0], [-3.0, 4.0]])

    first = first_rotor.sandwich(vector)
    second = second_rotor.sandwich(vector)
    combined = first(second)
    direct = (first_rotor * second_rotor).sandwich(vector)
    round_trip = combined.inverse()(combined(vectors))

    assert combined.gatype.entails(CoefficientOrthogonal)
    np.testing.assert_allclose(
        combined.kernel,
        direct.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        round_trip.kernel,
        vectors.kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


@pytest.mark.parametrize("grade", range(4))
def test_rotor_sandwich_is_grade_preserving_before_numeric_execution(grade):
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra)
    passenger = algebra.subspace.k_vector(grade)
    rotor = context.multivector.rotor([np.cos(0.2), np.sin(0.2), 0, 0])

    action = rotor.sandwich(passenger)

    assert action.axes == (passenger, passenger)


def test_general_even_sandwich_can_produce_extra_grades_in_five_dimensions():
    algebra = Algebra("x+y+z+w+v+")
    mv = NumpyContext(algebra).multivector
    x, y, z, w, v = mv.vector(np.eye(5))

    # 1 + xyzw is even but is not a versor. Repeated-argument cancellation
    # alone cannot remove its grade-5 output when it sandwiches a vector.
    general = mv.scalar([1]) + x * y * z * w
    actual = general.sandwich(v)
    expected = 2 * (v + x.wedge(y).wedge(z).wedge(w).wedge(v))

    assert general.subspace is algebra.subspace.even()
    assert not general.gatype.entails(Versor)
    assert not actual.gatype <= algebra.subspace.vector()
    np.testing.assert_allclose(
        (actual - expected).kernel, 0,
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_sandwich_grade_guarantee_needs_the_same_sandwicher_not_just_its_type():
    algebra = Algebra("x+y+z+")
    mv = NumpyContext(algebra).multivector
    spaces = algebra.subspace
    x_space = spaces.from_masks([algebra.parse_blade("x").mask])
    x = mv(x_space, [1])
    half = np.sqrt(0.5)
    r = mv.rotor([half, half, 0, 0])
    s = mv.rotor([0, 0, 0, 1])
    sandwich = algebra.gatype.rotor().sandwich(x_space)

    rotation = sandwich.bind({0: r, 2: r})
    independent = sandwich.bind({0: r, 2: s})

    assert r.gatype is s.gatype
    # Preserving grade does not mean preserving the passenger's blade support:
    # the rotation takes x to -y, outside its original one-coordinate carrier.
    assert rotation.axes == (spaces.vector(), x_space)
    np.testing.assert_allclose(
        rotation(x).kernel, [0, -1, 0],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    # Distinct rotors with the same type instead give (z - xyz) / sqrt(2).
    assert independent.output_subspace is spaces.vector() + spaces.trivector()
    assert not independent.gatype.entails(CoefficientOrthogonal)
    np.testing.assert_allclose(
        independent(x).kernel, [0, 0, half, -half],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


@pytest.mark.parametrize(
    ("parity", "blade", "expected_scale"),
    [pytest.param(0, "xy", 4, id="even"), pytest.param(1, "x", -4, id="odd")],
)
def test_known_nonunit_versor_constructs_a_grade_preserving_5d_map(
    parity, blade, expected_scale,
):
    algebra = Algebra("x+y+z+w+v+")
    mv = NumpyContext(algebra).multivector
    spaces = algebra.subspace
    carrier = (
        spaces.even() if parity == 0
        else spaces.vector() + spaces.trivector() + spaces.pseudoscalar()
    )
    blade_mask = algebra.parse_blade(blade).mask
    coefficients = [2 if mask == blade_mask else 0 for mask in carrier.masks]
    # An explicit input certification: 2xy / 2x, stored on a broad carrier.
    # Only Versor is promised; neither normalization nor even parity is needed.
    known = mv(algebra.gatype(carrier, (Versor,)), coefficients)

    action = known.sandwich(spaces.vector())

    # The structural codomain must already be grade 1 before any application.
    assert action.axes == (spaces.vector(), spaces.vector())
    assert action.structural_shape == (5, 5)
    assert not action.gatype.entails(CoefficientOrthogonal)
    result = action(mv.vector([0, 0, 0, 0, 1]))
    np.testing.assert_allclose(
        result.kernel, [0, 0, 0, 0, expected_scale],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_sandwiching_a_rotor_preserves_its_type_and_fast_inverse():
    algebra = Algebra("x+y+z+")
    context = NumpyContext(algebra)
    even = algebra.subspace.even()
    frame = context.multivector.rotor([0, 1, 0, 0])
    attitude = context.multivector.rotor([0, 0, 1, 0])

    action = frame.sandwich(even)
    moved = action(attitude)
    inverse = moved.inverse()

    assert moved.gatype <= algebra.gatype.rotor()
    assert inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        moved.kernel,
        [0, 0, -1, 0],
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )
    np.testing.assert_allclose(
        inverse.kernel,
        moved.reverse().kernel,
        rtol=1e-14,
        atol=1e-14,
        equal_nan=False,
    )


def test_map_collections_preserve_orthogonality_but_reductions_do_not():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    rotations = mv.rotor([[1, 0], [0, 1]]).sandwich(algebra.subspace.vector())

    selected = rotations.reshape(2, 1).broadcast_to((2, 3))[:, 1]

    assert rotations.gatype.entails(CoefficientOrthogonal)
    assert selected.gatype.entails(CoefficientOrthogonal)
    np.testing.assert_allclose(
        selected.inverse().kernel, [[[1, 0], [0, 1]], [[-1, 0], [0, -1]]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    # I and -I cancel: collection membership preserves orthogonality,
    # whereas adding or averaging the represented maps need not do so.
    for reduced in (rotations.sum(axis=0), rotations.mean(axis=0)):
        assert reduced.arity == 1
        assert not reduced.gatype.entails(CoefficientOrthogonal)
        np.testing.assert_allclose(
            reduced(mv.vector([2, 3])).kernel, [0, 0],
            rtol=1e-14, atol=1e-14, equal_nan=False,
        )


def test_lorentz_sandwich_inverse_is_not_a_coefficient_transpose():
    algebra = Algebra("x+y-")
    mv = NumpyContext(algebra).multivector
    boost = mv.rotor([5 / 4, 3 / 4]).sandwich(algebra.subspace.vector())
    vectors = mv.vector([[1, 0], [1, 1]])

    assert not boost.gatype.entails(CoefficientOrthogonal)
    inverse = boost.inverse()

    np.testing.assert_allclose(
        inverse.kernel, [[17 / 8, 15 / 8], [15 / 8, 17 / 8]],
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )
    np.testing.assert_allclose(
        inverse(boost(vectors)).kernel, vectors.kernel,
        rtol=1e-14, atol=1e-14, equal_nan=False,
    )


def test_exp_product_sandwich_inverse_and_norm_work_as_one_expression_chain():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    points = mv.vector([[1, 0], [0, 1]])

    r = mv.bivector([0.2]).exp()
    s = mv.bivector([0.3]).exp()
    combined = r * s
    rotation = combined.sandwich(algebra.subspace.vector())
    moved = rotation(points)
    recovered = rotation.inverse()(moved)

    assert combined.gatype <= algebra.gatype.rotor()
    assert rotation.gatype.entails(CoefficientOrthogonal)
    # Match the approximate exp implementation's tolerance above; the unit
    # promise does not request an extra numerical normalization along this path.
    np.testing.assert_allclose(
        moved.kernel, [[np.cos(1), -np.sin(1)], [np.sin(1), np.cos(1)]],
        rtol=1e-10, atol=1e-10, equal_nan=False,
    )
    np.testing.assert_allclose(
        recovered.kernel, points.kernel,
        rtol=1e-10, atol=1e-10, equal_nan=False,
    )
    norm = combined.norm_squared()
    assert norm.gatype <= algebra.subspace.scalar()
    np.testing.assert_allclose(
        norm.kernel, [1], rtol=1e-10, atol=1e-10, equal_nan=False,
    )
