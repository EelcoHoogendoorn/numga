"""Core extension stories shared by NumPy execution and JAX compilation."""

import numpy as np
import pytest

from numga import Algebra, Extensor, NumpyContext


@pytest.fixture(params=("numpy", "jax"))
def backend(request):
    if request.param == "numpy":
        return NumpyContext, lambda function: function
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    return JaxContext, jax.jit


def test_explicit_measurement_and_normalization_correct_declared_unit_drift(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    context = context_type(algebra, dtype=np.float32)
    scale = 1.01
    coefficients = scale * np.asarray([[3 / 5, 4 / 5], [5 / 13, 12 / 13]])
    drifted = context.multivector.rotor(coefficients)

    def evaluate(value):
        return value.norm_squared(), value.normalized(), value.inverse()

    measured, repaired, trusted_inverse = compile(evaluate)(drifted)

    assert measured.shape == repaired.shape == trusted_inverse.shape == (2,)
    assert repaired.gatype <= algebra.gatype.rotor()
    assert trusted_inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        measured.kernel, np.full((2, 1), scale**2),
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    np.testing.assert_allclose(
        repaired.kernel, coefficients / scale,
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    # A consumer of the unit fact trusts it; only the explicit normalization
    # request above corrects coefficients.
    np.testing.assert_allclose(
        trusted_inverse.kernel, coefficients * [1, -1],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    np.testing.assert_allclose(
        drifted.kernel, coefficients,
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )


def test_map_kernel_forwards_array_arguments_and_preserves_traits_only_on_request(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    context = context_type(algebra)
    rotors = context.multivector.rotor([[1, 0], [0, 1]])

    def select(values):
        return values.map_kernel(
            context.xp.take, context.xp.asarray([1, 0, 1]),
            axis=0, preserve_traits=True,
        )

    selected = compile(select)(rotors)
    assert selected.gatype is rotors.gatype
    np.testing.assert_allclose(selected.inverse().kernel, [[0, -1], [1, 0], [0, -1]])

    total = compile(lambda x: x.map_kernel(context.xp.sum, axis=0))(rotors)
    assert total.gatype is rotors.gatype.structural
    np.testing.assert_allclose(total.kernel, [1, 1])


def test_map_kernel_also_operates_on_batches_of_maps(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    context = context_type(algebra)
    rotations = context.multivector.rotor([[1, 0], [0, 1]]).sandwich(algebra.subspace.vector())

    repeated = compile(lambda x: x.map_kernel(
        context.xp.repeat, 2, axis=0, preserve_traits=True,
    ))(rotations)

    assert repeated.gatype is rotations.gatype
    assert repeated.shape == (4,)
    np.testing.assert_allclose(repeated.kernel, np.repeat(rotations.kernel, 2, axis=0))


def test_batched_general_multivector_inverse_without_constructor_traits(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+z+")
    context = context_type(algebra, dtype=np.float32)
    # Full support does not promise a scalar self-product. These particular
    # values are 2 + I and 3 - 2I, where I**2 == -1.
    values = context.multivector.full(
        [[2, 0, 0, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 0, 0, -2]],
    )

    inverse = compile(lambda value: value.inverse())(values)

    assert inverse.shape == (2,)
    assert inverse.subspace is algebra.subspace.full()
    np.testing.assert_allclose(
        inverse.kernel,
        [[2 / 5, 0, 0, 0, 0, 0, 0, -1 / 5],
         [3 / 13, 0, 0, 0, 0, 0, 0, 2 / 13]],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    for identity in (values * inverse, inverse * values):
        np.testing.assert_allclose(
            identity.kernel, np.broadcast_to([1, 0, 0, 0, 0, 0, 0, 0], (2, 8)),
            rtol=2e-6, atol=2e-6, equal_nan=False,
        )


def test_unary_inverse_swaps_distinct_carriers_and_keeps_batch_axes(backend):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    context = context_type(algebra, dtype=np.float32)
    vector, even = algebra.subspace.vector(), algebra.subspace.even()
    maps = context.extensor(
        algebra.gatype((even, vector)),
        [[[2, 1], [0, 3]], [[1, 0], [-2, 4]]],
    )
    value = context.multivector.vector([1, 2])

    inverse = compile(lambda map_: map_.inverse())(maps)
    recovered = compile(lambda map_, undo, x: undo(map_(x)))(maps, inverse, value)

    assert inverse.axes == (vector, even)
    assert inverse.shape == recovered.shape == (2,)
    np.testing.assert_allclose(
        inverse.kernel,
        [[[1 / 2, -1 / 6], [0, 1 / 3]], [[1, 0], [1 / 2, 1 / 4]]],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    np.testing.assert_allclose(
        recovered.kernel, [[1, 2], [1, 2]],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )


@pytest.mark.parametrize(
    ("description", "scalar_part", "bivector_part"),
    [
        pytest.param("x+y+", np.cos, np.sin, id="rotation"),
        pytest.param("x+y-", np.cosh, np.sinh, id="boost"),
        pytest.param("x+w0", np.ones_like, np.positive, id="translation"),
    ],
)
def test_bivector_exp_log_and_inverse_share_backend_and_handle_zero(
    backend, description, scalar_part, bivector_part,
):
    context_type, compile = backend
    algebra = Algebra(description)
    context = context_type(algebra, dtype=np.float32)
    angles = np.asarray([-0.2, 0, 0.3])
    generators = context.multivector.bivector(angles[:, None])

    def evaluate(value):
        # An explicit shallower depth trades truncation against float32
        # repeated-squaring error; both exp and log use the same quadratic map.
        rotor = value.exp(n=8)
        return rotor, rotor.log(n=8), rotor.inverse()

    rotors, recovered, inverses = compile(evaluate)(generators)

    assert rotors.shape == recovered.shape == inverses.shape == (3,)
    assert rotors.gatype <= algebra.gatype.rotor()
    assert inverses.gatype <= algebra.gatype.rotor()
    assert recovered.gatype <= algebra.subspace.bivector()
    for result in (rotors, recovered, inverses):
        assert result.context.key == context.key
        assert np.dtype(result.kernel.dtype) == context.dtype
    expected = np.stack((scalar_part(angles), bivector_part(angles)), axis=-1)
    np.testing.assert_allclose(
        rotors.kernel, expected,
        rtol=0, atol=2e-5, equal_nan=False,
    )
    np.testing.assert_allclose(
        recovered.kernel, angles[:, None],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
    np.testing.assert_allclose(
        inverses.kernel, expected * [1, -1],
        rtol=0, atol=2e-5, equal_nan=False,
    )


def test_user_exp_specialization_returns_useful_traits_without_a_wrapper(backend, monkeypatch):
    context_type, compile = backend
    algebra = Algebra("x+y+")
    context = context_type(algebra, dtype=np.float32)
    bivector = algebra.subspace.bivector()
    # Keep this test's user registration local while retaining the built-ins.
    dispatch = Extensor.exp._dispatch
    monkeypatch.setattr(dispatch, "_predicates", list(dispatch._predicates))
    monkeypatch.setattr(dispatch, "_cache", {})
    calls = []

    @Extensor.exp.register(lambda gatype: gatype.subspaces == (bivector,), position=0)
    def planar_exp(value):
        calls.append(value.gatype)
        xp = value.context.xp
        coefficients = xp.concatenate((xp.cos(value.kernel), xp.sin(value.kernel)), axis=-1)
        return value.context.multivector.rotor(coefficients)

    def evaluate(value):
        return value.exp().inverse()

    inverse = compile(evaluate)(context.multivector.bivector([[0], [0.3]]))

    assert calls == [algebra.gatype.bivector()]
    assert inverse.gatype <= algebra.gatype.rotor()
    np.testing.assert_allclose(
        inverse.kernel, [[1, 0], [np.cos(0.3), -np.sin(0.3)]],
        rtol=2e-6, atol=2e-6, equal_nan=False,
    )
