import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from numga import Algebra, Extensor
from numga.backend.jax import JaxContext


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
