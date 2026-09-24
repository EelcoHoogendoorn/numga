"""Core extension stories shared by NumPy execution and JAX compilation."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext


@pytest.fixture(params=("numpy", "jax"))
def backend(request):
    if request.param == "numpy":
        return NumpyContext, lambda function: function
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    return JaxContext, jax.jit


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
            rtol=2e-5, atol=2e-5, equal_nan=False,
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


