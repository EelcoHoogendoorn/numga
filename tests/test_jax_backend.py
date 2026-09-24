from contextlib import contextmanager
from itertools import islice

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from numga import Algebra, Extensor, NumpyContext
from numga.algebras import PGA3D, STA
from numga.backend.jax import JaxContext
from tests.backend_surface import operations


@contextmanager
def enable_x64():
    """Double precision for one test, restoring the global setting after."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _agrees_with_numpy(ga: Algebra, execution: str, transform) -> None:
    reference = NumpyContext(ga, execution=execution)
    with enable_x64():
        context = JaxContext(ga, np.float64, execution=execution)
        for name, operation in operations(ga, np.random.default_rng(0)).items():
            result = transform(lambda: operation(context))()
            assert isinstance(result.kernel, jax.Array), name
            np.testing.assert_allclose(
                np.asarray(result.kernel), operation(reference).kernel, atol=1e-10, err_msg=name,
            )


@pytest.mark.parametrize("execution", ["dense", "sparse"])
@pytest.mark.parametrize("ga", [PGA3D, STA, Algebra("x+y+z+w+e-")], ids=str)
def test_jax_agrees_with_numpy_across_the_library_surface(ga, execution):
    _agrees_with_numpy(ga, execution, lambda function: function)


@pytest.mark.parametrize("execution", ["dense", "sparse"])
@pytest.mark.parametrize("ga", [PGA3D, STA], ids=str)
def test_jitted_library_surface_agrees_with_numpy(ga, execution):
    _agrees_with_numpy(ga, execution, jax.jit)


def test_jax_extensor_is_a_stable_pytree_and_traces_product_expressions():
    algebra = Algebra("x+y+")
    even = algebra.subspace.even()
    context = JaxContext(algebra)
    left = context.extensor(even, [[1, 2], [3, 4]])
    right = context.extensor(even, [[3, 4], [5, 6]])

    assert type(left) is Extensor
    assert context.dtype == np.dtype(
        jax.dtypes.canonicalize_dtype(np.dtype(np.float32))
    )
    leaves, tree = jax.tree_util.tree_flatten(left)
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert len(leaves) == 1
    assert rebuilt.gatype is left.gatype
    assert rebuilt.context.algebra is algebra
    assert rebuilt.context.key == context.key
    assert rebuilt.context is not context

    def product(first, second):
        return first * second

    eager = product(left, right)
    traced = jax.jit(product)(left, right)
    vmapped = jax.jit(jax.vmap(product))(left, right)

    expected = np.asarray([[-5, 10], [-9, 38]], dtype=np.float32)
    for result in (eager, traced, vmapped):
        assert type(result) is Extensor
        assert result.gatype is eager.gatype
        assert result.context.algebra is algebra
        assert result.context.key == context.key
        assert result.shape == (2,)
        np.testing.assert_allclose(
            np.asarray(result.kernel),
            expected,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False,
        )


def test_jit_vector_sandwich_agrees_in_staged_and_direct_forms():
    algebra = Algebra("x+y+")
    vector = algebra.subspace.vector()
    context = JaxContext(algebra)
    e1 = context.extensor(vector, [1, 0])
    point = context.extensor(vector, [2, 3])

    reflection = jax.jit(lambda sandwicher: sandwicher.sandwich(vector))(e1)
    staged = jax.jit(lambda map_, value: map_(value))(reflection, point)
    direct = jax.jit(lambda sandwicher, value: sandwicher.sandwich(value))(
        e1, point
    )

    assert reflection.arity == 1
    assert reflection.axes == (vector, vector)
    assert staged.arity == 0
    assert direct.gatype is staged.gatype
    for result in (staged, direct):
        np.testing.assert_allclose(
            np.asarray(result.kernel),
            [2, -3],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=False,
        )


def test_jax_iteration_ends_at_the_batch_length_and_null_inverse_fails_during_trace():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    context = JaxContext("x+y+z+w0")
    rows = context.multivector.vector(np.eye(4))
    assert len(list(islice(rows, 6))) == 4
    x, y, z, w = rows
    np.testing.assert_array_equal(w.kernel, [0, 0, 0, 1])
    with pytest.raises(ZeroDivisionError, match="statically null"):
        jax.jit(lambda value: value.inverse())(context.multivector.w)
