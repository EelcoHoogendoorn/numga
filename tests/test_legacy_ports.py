"""Geometric reconstruction and numerical-boundary checks for the legacy ports."""

from fractions import Fraction

import numpy as np
import pytest

from numga import Algebra, NumpyContext
from numga.algebras import PGA3D
from numga.extensions.optimized import exp_pga3, normalize_pga3


@pytest.mark.parametrize("signature", ("x+y+z+w0", "x+y+z+w+", "x+y+z+p+n-"))
def test_decompositions_and_motor_approximations_reconstruct(signature):
    context = NumpyContext(signature)
    mv, ga = context.multivector, context.algebra
    b = mv.bivector(np.arange(len(ga.subspace.bivector())) * .03 + .02)
    left, right = b.decompose_invariant()
    np.testing.assert_allclose((left + right - b).kernel, 0, atol=1e-14)
    np.testing.assert_allclose((left ^ left).kernel, 0, atol=1e-14)
    np.testing.assert_allclose((right ^ right).kernel, 0, atol=1e-14)
    np.testing.assert_allclose(left.commutator(right).kernel, 0, atol=1e-14)
    line, scale = b.decompose_polar()
    np.testing.assert_allclose((line * scale - b).kernel, 0, atol=1e-14)
    motor = b.exp()
    origin = mv.antivector([.1] * (ga.dimension - 1) + [1])
    translation, rotation = motor.motor_split(origin)
    np.testing.assert_allclose((translation * rotation - motor).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((rotation >> origin).kernel, origin.kernel, atol=1e-10)
    for exponential, logarithm in (("exp_linear_normalized", "log_linear_normalized"),
                                  ("exp_quadratic", "log_quadratic")):
        reconstructed = getattr(getattr(b, exponential)(), logarithm)()
        np.testing.assert_allclose((reconstructed - b).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((motor.log_pade() - b).kernel, 0, atol=1e-9)
    np.testing.assert_allclose((motor.square_root_denman_beavers().squared() - motor).kernel, 0, atol=1e-10)


def test_canonical_motor_split_and_simple_bivector_batches():
    mv = NumpyContext(PGA3D).multivector
    b = mv.xy * np.array([.1, .2, .3])
    left, right = b.decompose_invariant()
    assert right.shape == (3,)
    np.testing.assert_allclose((left + right - b).kernel, 0)
    motor = (b + mv.xw * .2).exp()
    translation, rotation = motor.motor_split(mv.zyx)
    np.testing.assert_allclose((translation * rotation - motor).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((rotation >> mv.zyx).kernel, [[1], [1], [1]], atol=1e-10)


@pytest.mark.parametrize("signature", ("x+y+z+", "x+y+z+w0", "x+y+z+p+n-"))
def test_shirokov_matches_inverse_and_native_extensor_solve(signature):
    context = NumpyContext(signature)
    ga, mv = context.algebra, context.multivector
    Full = context.gatype.full()
    value = mv.full(np.r_[3., np.arange(ga.blade_count - 1) * .001])
    multiplication = value * Full
    inverse_la = multiplication.inverse()(mv.scalar())
    inverse = value.inverse_shirokov()
    np.testing.assert_allclose((inverse - inverse_la).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((inverse * value - 1).kernel, 0, atol=1e-12)
    rhs = mv.full(np.arange(ga.blade_count))
    solution = multiplication.solve(rhs)
    np.testing.assert_allclose((value * solution - rhs).kernel, 0, atol=1e-12)


def test_hitzer_factor_is_an_adjugate_where_its_product_reduces_to_scalar():
    mv = NumpyContext("x+y+z+").multivector
    value = mv.full([3, .1, -.2, .3, .2, -.3, .4, .1])
    inverse = value.inverse_hitzer()
    np.testing.assert_allclose((value * inverse - 1).kernel, 0, atol=1e-14)
    np.testing.assert_allclose((inverse * value - 1).kernel, 0, atol=1e-14)


@pytest.mark.parametrize("backend", ("numpy", "jax"))
def test_optimized_pga3_formulas_handle_signed_layouts_and_translation_limit(backend):
    if backend == "jax":
        jax = pytest.importorskip("jax")
        from numga.backend.jax import JaxContext
        context, compile = JaxContext(PGA3D), jax.jit
    else:
        context, compile = NumpyContext(PGA3D), lambda f: f
    mv = context.multivector
    rotation = np.array([[.1, .2, -.3], [0, 0, 0], [1e-8, -1e-8, 1e-8]])
    coefficients = np.concatenate((rotation, np.full((3, 3), .2)), axis=-1)
    b = mv.bivector(coefficients)
    result = compile(exp_pga3)(b)
    expected = NumpyContext(PGA3D).multivector.bivector(coefficients).exp()
    np.testing.assert_allclose((result - context.lower(context.extensor(expected.gatype, expected.kernel))).kernel,
                               0, atol=2e-7)
    raw = mv.even(np.random.default_rng(2).normal(size=(3, 8)))
    reference = NumpyContext(PGA3D).multivector.even(np.asarray(raw.kernel)).normalized()
    normalized = compile(normalize_pga3)(raw)
    np.testing.assert_allclose(normalized.cast(reference.subspace).kernel, reference.kernel, atol=1e-6)


def test_formula_and_generated_python_preserve_rational_coefficients():
    ga = Algebra("x+y+")
    V = ga.gatype.vector()
    expression = (V * V) * Fraction(1, 3)
    namespace = {}
    exec(expression.to_python("product"), namespace)
    actual = namespace["product"]([1, 2], [3, 4])
    expected = expression(ga.exact.multivector.vector([1, 2]), ga.exact.multivector.vector([3, 4]))
    assert actual == list(expected.kernel.to_object_array())
    assert "Fraction(1, 3)" in expression.formula()
    assert "a0[x]" in expression.formula()
