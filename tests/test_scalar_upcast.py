"""Arrays in linear arithmetic are batches of scalars; explicit constructors take their axis as given."""

import numpy as np
import pytest

from numga import NumpyContext
from numga.algebras import PGA2D, PGA3D

ctx = NumpyContext(PGA2D)
mv = ctx.multivector
weights = np.linspace(0.0, 1.0, 5)


def test_scalar_constructor_takes_the_structural_axis_as_given():
    assert mv.scalar([1.0]).shape == ()                         # the established spelling
    assert mv.scalar(weights[:, None]).shape == (5,)
    assert mv.scalar(weights[:, None, None]).shape == (5, 1)    # explicit shapes are never reinterpreted
    with pytest.raises(ValueError):
        mv.scalar(weights)                                      # no silent axis for explicit constructors


def test_arrays_scale_offset_and_divide_as_batch_scalars():
    reference = mv.w * mv.scalar(weights[:, None])
    np.testing.assert_allclose((mv.w * weights).kernel, reference.kernel)
    np.testing.assert_allclose((weights * mv.w).kernel, reference.kernel)
    np.testing.assert_allclose((mv.w / (weights + 1)).kernel, (mv.w * mv.scalar(1 / (weights[:, None] + 1))).kernel)
    np.testing.assert_allclose((mv.w + weights).kernel, (mv.w + mv.scalar(weights[:, None])).kernel)
    np.testing.assert_allclose((weights - mv.w).kernel, (mv.scalar(weights[:, None]) - mv.w).kernel)
    np.testing.assert_allclose((weights / mv.scalar([2.0])).kernel[:, 0], weights / 2)


def test_arrays_scale_maps_and_broadcast_against_batches():
    P = PGA2D.subspace.antivector()
    motion = (mv.xy * 0.3).exp()
    scaled_map = (motion >> P) * weights
    assert scaled_map.arity == 1 and scaled_map.shape == (5,)
    np.testing.assert_allclose(scaled_map.kernel, weights[:, None, None] * (motion >> P).kernel)
    points = mv.antivector(np.random.default_rng(0).normal(size=(5, 3)))
    np.testing.assert_allclose((points * weights).kernel, points.kernel * weights[:, None])


def test_geometric_operators_still_reject_arrays():
    with pytest.raises(TypeError):
        mv.w ^ weights
    with pytest.raises(TypeError):
        mv.w & weights


def test_jax_treats_numpy_arrays_as_constants():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    jmv = JaxContext(PGA3D).multivector
    scaled = jax.jit(lambda: (jmv.x * weights).kernel)()
    np.testing.assert_allclose(np.asarray(scaled)[:, 0], weights, rtol=1e-6)


def test_arrays_in_arithmetic_keep_a_trailing_axis_of_length_one_as_batch():
    mask = np.array([[1.0], [0.0]])                             # shape (2, 1): a batch, not a structural axis
    scaled = mv.w.reshape(1) * mask
    assert scaled.shape == (2, 1)
    np.testing.assert_allclose(scaled.kernel[..., 0], mask * mv.w.kernel[0])
    assert (mv.w + mask).shape == (2, 1)
    assert (mv.w / (mask + 1)).shape == (2, 1)
