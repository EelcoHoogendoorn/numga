"""Scalar numerical operations preserve geometry types and expose batch-shaped masks."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.gatype import Versor


@pytest.fixture(params=["numpy", "jax"])
def context(request):
    algebra = Algebra("x+y+")
    if request.param == "jax":
        pytest.importorskip("jax")
        from numga.backend.jax import JaxContext
        return JaxContext(algebra, dtype=np.float32)
    return NumpyContext(algebra)


def test_scalar_functions_preserve_singleton_batches_and_drop_traits(context):
    angle = context.multivector.scalar(np.array([[[0.2]], [[0.7]]])).with_traits(Versor)
    result = angle.cos().arccos()
    assert result.shape == (2, 1)
    assert result.gatype == context.algebra.gatype.scalar()
    np.testing.assert_allclose(result.to_array(), [[0.2], [0.7]], atol=1e-5)
    np.testing.assert_allclose(angle.sinh().arcsinh().to_array(), angle.to_array(), atol=1e-5)
    np.testing.assert_allclose(angle.tanh().arctanh().to_array(), angle.to_array(), atol=1e-5)
    np.testing.assert_allclose(angle.tan().arctan().to_array(), angle.to_array(), atol=1e-5)
    np.testing.assert_allclose(angle.sin().arcsin().to_array(), angle.to_array(), atol=1e-5)
    np.testing.assert_allclose(angle.cosh().arccosh().to_array(), angle.to_array(), atol=1e-5)


def test_predicates_are_backend_boolean_masks(context):
    value = context.multivector.scalar(np.array([[[np.nan]], [[np.inf]], [[-1.0]]]))
    for name, expected in (("isnan", [[True], [False], [False]]),
                           ("isfinite", [[False], [False], [True]]),
                           ("isinf", [[False], [True], [False]])):
        mask = getattr(value, name)()
        assert mask.shape == value.shape
        assert mask.dtype == np.bool_
        assert type(mask).__module__.startswith(type(context.xp.asarray(0)).__module__.split('.')[0])
        np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(value[value.isfinite()].to_array(), [-1.0])
    assert context.multivector.scalar([1.0]).isfinite().shape == ()


def test_comparisons_and_clipping_broadcast_only_batch_axes(context):
    value = context.multivector.scalar(np.array([[[-1.0]], [[2.0]]]))
    bound = context.multivector.scalar(np.array([0.0, 1.0, 3.0])[:, None])
    np.testing.assert_array_equal(value < bound, [[True, True, True], [False, False, True]])
    np.testing.assert_array_equal(value >= bound, [[False, False, False], [True, True, False]])
    np.testing.assert_array_equal(value <= 0, [[True], [False]])
    np.testing.assert_array_equal(0 < value, [[False], [True]])
    assert (value > np.array([0.0, 1.0, 3.0])).shape == (2, 3)
    np.testing.assert_array_equal(value.clip(0, np.array([0.5, 1.0, 3.0])).to_array(),
                                  [[0, 0, 0], [0.5, 1, 2]])


def test_signed_and_empty_scalar_layouts(context):
    signed = context.multivector.scalar([0.25]).cast(context.algebra.subspace("-1"))
    np.testing.assert_allclose(signed.sin().to_array(), np.sin(0.25), atol=1e-6)
    assert signed > 0
    empty = context.multivector(context.algebra.subspace.empty(), np.empty((2, 1, 0)))
    np.testing.assert_array_equal(empty.cos().to_array(), np.ones((2, 1)))
    np.testing.assert_array_equal(empty.isfinite(), np.ones((2, 1), dtype=bool))


def test_scalar_index_operations_use_backend_arrays_and_batch_axes(context):
    values = np.array([[[3, 1, 2]], [[0, 2, 1]]])
    scalar = context.multivector.scalar(values[..., None]).cast(context.algebra.subspace("-1"))
    for indices, axis in ((scalar.argsort(), -1), (scalar.argsort(axis=0), 0)):
        assert indices.shape == scalar.shape == (2, 1, 3)
        assert indices.dtype.kind in "iu"
        assert isinstance(indices, type(context.xp.asarray(0)))
        np.testing.assert_array_equal(indices, np.argsort(values, axis=axis))
    maximum = scalar.argmax()
    assert maximum.shape == ()
    assert maximum.dtype.kind in "iu"
    np.testing.assert_array_equal(maximum, np.argmax(values))
    maximum = scalar.argmax(axis=0, keepdims=True)
    assert maximum.shape == (1, 1, 3)
    assert maximum.dtype.kind in "iu"
    assert isinstance(maximum, type(context.xp.asarray(0)))
    np.testing.assert_array_equal(maximum, np.argmax(values, axis=0, keepdims=True))
    minimum = scalar.argmin()
    assert minimum.shape == ()
    assert minimum.dtype.kind in "iu"
    np.testing.assert_array_equal(minimum, np.argmin(values))
    minimum = scalar.argmin(axis=0, keepdims=True)
    assert minimum.shape == (1, 1, 3)
    assert minimum.dtype.kind in "iu"
    assert isinstance(minimum, type(context.xp.asarray(0)))
    np.testing.assert_array_equal(minimum, np.argmin(values, axis=0, keepdims=True))

    neg = context.multivector.scalar([-2.5])
    np.testing.assert_allclose(neg.abs().to_array(), 2.5)


def test_nonlinear_scalar_methods_do_not_match_vectors_or_open_forms(context):
    mv = context.multivector
    vector = context.algebra.gatype.vector()
    form = vector | (mv.scalar([1]) >> vector)
    for value in (mv.x, context.lower(form)):
        with pytest.raises(LookupError):
            value.cos()
        with pytest.raises(LookupError):
            value.abs()
        with pytest.raises(LookupError):
            value.isnan()
        with pytest.raises(LookupError):
            value.argsort()
        with pytest.raises(LookupError):
            value.argmax()
        with pytest.raises(LookupError):
            value.argmin()
        with pytest.raises(LookupError):
            value < 0
