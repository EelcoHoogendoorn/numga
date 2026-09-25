"""Default log/exp algorithms exercised through the public Extensor surface."""

import numpy as np
import pytest

from numga import Algebra, Extensor, NumpyContext, ReverseProductOne, Versor


def forbid_input_normalization(monkeypatch, *inputs, measured_inputs=()):
    """Root numerators may be normalized; the supplied motor must be trusted."""

    for name in ("norm", "norm_squared", "normalized"):
        original = getattr(Extensor, name)
        guarded_inputs = inputs + measured_inputs if name == "normalized" else inputs

        def checked(value, *args, _original=original, _inputs=guarded_inputs, **kwargs):
            assert all(value is not supplied for supplied in _inputs), (
                "log must not defensively normalize its input"
            )
            return _original(value, *args, **kwargs)

        monkeypatch.setattr(Extensor, name, checked)


def test_empty_exp_log_have_separate_overloads_and_preserve_batch_shape():
    algebra = Algebra("x+y+")
    empty = NumpyContext(algebra).multivector.empty(np.empty((2, 3, 0)))

    unit = empty.exp()
    assert unit.shape == (2, 3)
    assert unit.subspace is algebra.subspace.scalar()
    assert unit.gatype.entails(ReverseProductOne)
    np.testing.assert_array_equal(unit.kernel, np.ones((2, 3, 1)))
    with np.errstate(divide="ignore"):
        logarithm = empty.log()
    np.testing.assert_array_equal(logarithm.kernel, np.full((2, 3, 1), -np.inf))
    exact_unit = algebra.exact.multivector.empty().exp()
    assert exact_unit.context is algebra.exact
    assert exact_unit.kernel.values.tolist() == [1]


def test_scalar_exp_log_are_batched():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    coefficients = np.asarray([[0], [0.3], [-0.2]])
    value = mv.scalar(coefficients)

    exponential = value.exp()
    logarithm = exponential.log()

    assert exponential.shape == (3,)
    np.testing.assert_allclose(exponential.kernel, np.exp(coefficients), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(logarithm.kernel, coefficients, rtol=1e-14, atol=1e-14)


def test_unit_and_nonunit_versor_logs_dispatch_separately_without_widening(monkeypatch):
    algebra = Algebra("x+y+z+")
    xy = algebra.subspace.from_masks((algebra.parse_blade("xy").mask,))
    angles = np.asarray([0, 0.2, -0.4])
    generator = NumpyContext(algebra).multivector(xy, angles[:, None])
    unit = generator.exp()
    scaled = (2 * unit).with_traits(Versor)

    forbid_input_normalization(monkeypatch, unit, measured_inputs=(scaled,))

    unit_log = unit.log()
    scaled_log = scaled.log()
    assert unit_log.subspace is xy
    assert scaled_log.subspace is algebra.subspace.scalar() + xy
    assert unit_log.shape == scaled_log.shape == (3,)
    np.testing.assert_allclose(unit_log.kernel, angles[:, None], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        scaled_log.kernel,
        np.column_stack((np.full(3, np.log(2)), angles)),
        atol=1e-9, rtol=1e-9,
    )


@pytest.mark.parametrize(
    ("description", "scalar_function", "bivector_function"),
    [("x+y+", np.cos, np.sin), ("x+y-", np.cosh, np.sinh),
     ("x+w0", np.ones_like, np.positive)],
)
def test_scalar_square_exp_log_covers_zero_rotation_boost_and_translation(
    description, scalar_function, bivector_function,
):
    algebra = Algebra(description)
    mv = NumpyContext(algebra).multivector
    parameters = np.asarray([-0.7, -1e-10, 0, 1e-10, 0.4])
    generator = mv.bivector(parameters[:, None])

    rotor = generator.exp()

    assert rotor.gatype.entails(Versor)
    assert rotor.gatype.entails(ReverseProductOne)
    np.testing.assert_allclose(
        rotor.kernel,
        np.stack((scalar_function(parameters), bivector_function(parameters)), axis=-1),
        # The quadratic base has angle error abs(a)**3 / (12 * 4**15), in addition
        # to rounding accumulated by 15 squarings.
        rtol=1e-8, atol=1e-8,
    )
    analytic_rotor = mv.rotor(np.stack(
        (scalar_function(parameters), bivector_function(parameters)), axis=-1,
    ))
    np.testing.assert_allclose(
        analytic_rotor.log().kernel, parameters[:, None], rtol=1e-8, atol=1e-8,
    )


def test_pga3_screw_exp_matches_commuting_rotation_and_translation():
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    # No specialization import is needed for generic PGA3 support. xy and zw
    # commute, giving an independent analytic reference for bisection.
    x, y, z, w = mv.vector(np.eye(4))
    plane, ideal = x.wedge(y), z.wedge(w)
    angle = 0.4
    distance = 0.7
    generator = angle * plane + distance * ideal
    rotation = mv.scalar([np.cos(angle)]) + np.sin(angle) * plane
    translation = mv.scalar([1]) + distance * ideal

    screw = generator.exp()
    reference = rotation * translation

    np.testing.assert_allclose((screw - reference).kernel, 0, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        (reference.with_traits(Versor, ReverseProductOne).log() - generator).kernel,
        0, rtol=0, atol=1e-8,
    )
    np.testing.assert_allclose(
        (screw * screw.reverse() - mv.scalar([1])).kernel,
        0, rtol=0, atol=1e-9,
    )


def test_generic_pga3_mixed_motor_round_trip_near_identity():
    algebra = Algebra("x+y+z+w0")
    mv = NumpyContext(algebra).multivector
    generator = mv.bivector([
        [0.2, -0.3, 0.1, 1, -2, 0.5],
        [1e-10, 0, 0, 0.2, -0.3, 0.5],
        [0, 0, 0, 0.7, -1.2, 2],
    ])

    motors = generator.exp()

    assert motors.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(motors.log().kernel, generator.kernel, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        (motors * motors.reverse() - mv.scalar([1])).kernel,
        0, rtol=0, atol=1e-9,
    )


@pytest.mark.parametrize(
    ("description", "angles"),
    [("x+y+z+w+", (0.3, -0.4)), ("x+y+z+w+u+v+", (0.15, -0.2, 0.25))],
)
def test_general_bivector_exp_supports_independent_rotation_planes(description, angles):
    algebra = Algebra(description)
    mv = NumpyContext(algebra).multivector
    vectors = mv.vector(np.eye(algebra.dimension))
    generator = mv.bivector(np.zeros(len(algebra.subspace.bivector())))
    reference = mv.scalar([1])
    for index, angle in enumerate(angles):
        plane = vectors[2 * index].wedge(vectors[2 * index + 1])
        generator = generator + angle * plane
        reference = reference * (mv.scalar([np.cos(angle)]) + np.sin(angle) * plane)

    result = generator.exp()

    assert result.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose((result - reference).kernel, 0, rtol=0, atol=1e-8)
    if algebra.dimension == 6:
        # Motor roots normalize m + 1 through scalar or Study roots.
        # The full 6D even carrier has a more general reverse product.
        with pytest.raises(LookupError, match="no 'normalized'"):
            reference.with_traits(Versor, ReverseProductOne).log()
    else:
        np.testing.assert_allclose(
            reference.with_traits(Versor, ReverseProductOne).log().kernel,
            generator.kernel, rtol=1e-8, atol=1e-8,
        )


def test_log_requires_declared_unit_input_and_never_repairs_it(monkeypatch):
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    coefficients = [1.1, 0.2]
    with pytest.raises(LookupError, match="no 'log'"):
        mv.even(coefficients).log()

    declared = mv.rotor(coefficients)
    forbid_input_normalization(monkeypatch, declared)
    # This assertion deliberately lies about the unit coefficients. The first
    # root trusts it, so its angle is np.arctan2(0.2, 1.1 + 1); subsequent roots
    # are unit. Normalizing the supplied motor would instead give np.arctan2(0.2, 1.1).
    effective_angle = 2 * np.arctan2(0.2, 2.1)
    np.testing.assert_allclose(
        declared.log().kernel, [effective_angle], rtol=1e-10, atol=1e-10,
    )


@pytest.mark.parametrize("signature, blades, coefficients", [
    ("t+x-y-z-", "txyz", [[0.7], [-1.3], [0.0]]),     # the spacetime pseudoscalar squares to -1
    ("x+y+z+w+", "xyzw", [[0.8], [0.0]]),             # the Euclidean 4D pseudoscalar squares to +1
    ("x+y+z+", "xyz", [[1.1]]),
    ("x+y+z+w0", "xyzw", [[0.4]]),                    # the PGA pseudoscalar squares to 0
])
def test_exp_of_an_element_squaring_to_a_scalar_matches_its_power_series(signature, blades, coefficients):
    ga = Algebra(signature)
    mv = NumpyContext(ga).multivector
    x = mv(ga.subspace(blades), np.array(coefficients))
    total, power = x * 0 + 1, x * 0 + 1
    for k in range(1, 40):
        power = power * x / k
        total = total + power
    exponential = x.exp()
    assert exponential.shape == x.shape
    np.testing.assert_allclose(exponential.kernel, total.kernel, atol=1e-12)
