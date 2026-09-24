"""Small expression primitives used by mathematical extension bodies."""

from fractions import Fraction

import numpy as np
import pytest

from numga import Algebra, NumpyContext, ReverseProductOne, ReverseProductZero


@pytest.mark.parametrize("scalar", [2, np.float64(2)])
def test_scalar_offsets_broadcast_and_live_in_the_scalar_grade(scalar):
    algebra = Algebra("x+y+")
    vectors = NumpyContext(algebra).multivector.vector([[3, 4], [5, 12]])

    assert (vectors + scalar).subspace is algebra.subspace.scalar() + algebra.subspace.vector()
    np.testing.assert_array_equal((vectors + scalar).kernel, [[2, 3, 4], [2, 5, 12]])
    np.testing.assert_array_equal((scalar + vectors).kernel, [[2, 3, 4], [2, 5, 12]])
    np.testing.assert_array_equal((vectors - scalar).kernel, [[-2, 3, 4], [-2, 5, 12]])
    np.testing.assert_array_equal((scalar - vectors).kernel, [[2, -3, -4], [2, -5, -12]])


def test_scalar_offsets_work_inside_jax_tracing():
    jax = pytest.importorskip("jax")
    from numga.backend.jax import JaxContext

    algebra = Algebra("x+y+")
    value = JaxContext(algebra).multivector.vector([[3, 4], [5, 12]])
    result = jax.jit(lambda x: (x + 1) / 2 - 1)(value)
    np.testing.assert_allclose(result.kernel, [[-0.5, 1.5, 2], [-0.5, 2.5, 6]])


def test_division_uses_the_right_inverse_and_broadcasts_scalar_values():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    x = mv.vector([1, 0])
    y = mv.vector([0, 1])

    quotient = x / y

    np.testing.assert_allclose(quotient.kernel, [0, 1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((y.inverse() * x).kernel, [0, -1], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((quotient * y).kernel, x.kernel, atol=1e-14, rtol=1e-14)
    values = mv.vector([[2, 4], [8, 12]])
    divisors = mv.scalar([[2], [4]])
    np.testing.assert_allclose((values / divisors).kernel, [[1, 2], [2, 3]], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose((2 / divisors).kernel, [[1], [0.5]], atol=1e-14, rtol=1e-14)


def test_numeric_kernel_division_and_output_cast_work_before_binding():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    vector = algebra.subspace.vector()
    product = vector * vector
    a = mv.vector([1, 0])
    b = mv.vector([0, 1])

    half_wedge = (product / 2).cast(algebra.subspace.bivector())

    assert half_wedge.arity == 2
    assert half_wedge.context is algebra.exact
    np.testing.assert_allclose(half_wedge(a, b).kernel, [0.5], atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(a.cast(algebra.subspace.full()).kernel, [0, 1, 0, 0])
    np.testing.assert_array_equal(a.cast(algebra.subspace.scalar()).kernel, [0])
    assert a.cast(a.subspace) is a


def test_select_zero_fills_but_restrict_keeps_only_existing_blade_support():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    xy = spaces.from_masks((algebra.parse_blade("xy").mask,))
    value = NumpyContext(algebra).multivector(xy, [[0], [2]]) + 1

    restricted = value.restrict[2]
    selected = value.select[2]

    assert restricted.subspace is xy
    assert restricted.shape == selected.shape == (2,)
    np.testing.assert_array_equal(restricted.kernel, [[0], [2]])
    assert selected.subspace is spaces.bivector()
    np.testing.assert_array_equal(selected.kernel, [[0, 0, 0], [2, 0, 0]])
    assert value.restrict[1].subspace is spaces.empty()
    assert value.restrict[1].kernel.shape == (2, 0)
    np.testing.assert_array_equal(value.select[1].kernel, np.zeros((2, 3)))
    assert value.restrict[0, 2].subspace is value.subspace
    assert value.restrict.bivector().subspace is xy
    assert value.restrict_grade(2).subspace is xy
    assert value.select.bivector().subspace is spaces.bivector()
    assert value.select_grade(2).subspace is spaces.bivector()


def test_restriction_preserves_unbound_inputs_and_batch_shape():
    algebra = Algebra("x+y+z+")
    spaces = algebra.subspace
    xy = spaces.from_masks((algebra.parse_blade("xy").mask,))
    even_plane = spaces.scalar() + xy
    product = even_plane * even_plane
    selected = product.select[2]
    restricted = product.restrict[2]

    assert selected.input_subspaces == restricted.input_subspaces == product.input_subspaces
    assert restricted.subspace is xy
    assert selected.subspace is spaces.bivector()
    assert restricted.context is algebra.exact
    mv = NumpyContext(algebra).multivector
    left = mv(even_plane, [[1, 2], [3, 4]])
    right = mv(even_plane, [5, 6])
    result = restricted(left, right)
    assert result.shape == (2,)
    np.testing.assert_allclose(result.kernel, [[16], [38]], atol=1e-14)
    partial = product.bind({0: left}).restrict[2]
    assert partial.arity == 1
    assert partial.shape == (2,)
    np.testing.assert_allclose(partial(right).kernel, result.kernel, atol=1e-14)


def test_self_products_combine_structural_cancellation_but_compute_values():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    vectors = mv.vector([[3, 4], [5, 12]])
    drift = 1.01
    rotor = mv.rotor(drift * np.asarray([3 / 5, 4 / 5]))

    assert vectors.squared().subspace is algebra.subspace.scalar()
    np.testing.assert_allclose(vectors.squared().kernel, [[25], [169]], atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(rotor.symmetric_reverse_product().kernel, [drift**2], atol=1e-14, rtol=1e-14)


def test_trait_annotation_trusts_coefficients_and_reuses_immutable_storage():
    algebra = Algebra("x+y+")
    value = NumpyContext(algebra).multivector.even([2, 0])

    declared = value.with_traits(ReverseProductOne)

    assert declared.gatype.entails(ReverseProductOne)
    assert not value.gatype.entails(ReverseProductOne)
    assert np.shares_memory(declared.kernel, value.kernel)
    np.testing.assert_array_equal(declared.kernel, [2, 0])
    assert declared.with_traits(ReverseProductOne) is declared
    with pytest.raises(ValueError, match="contradicts"):
        declared.with_traits(ReverseProductZero)


def test_scalar_affine_and_geometric_inverse_operations_reject_open_extensors():
    algebra = Algebra("x+y+")
    vector = algebra.subspace.vector()
    product = vector * vector
    scalar = algebra.exact.multivector.scalar([2])

    for operation in (
        lambda: product + 1,
        lambda: 1 + product,
        lambda: product - 1,
        lambda: 1 - product,
        lambda: scalar / product,
        lambda: 1 / product,
    ):
        with pytest.raises(TypeError, match="nullary"):
            operation()

    # Scalar extensor division on open extensors behaves consistently with numeric division:
    assert (product / scalar).gatype.subspaces == (product / 2).gatype.subspaces


def test_raw_arrays_are_batch_scalars_in_linear_arithmetic():
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    value = mv.vector([1, 2])
    raw = np.asarray([2.0, 3.0])
    np.testing.assert_allclose((value * raw).kernel, (value * mv.scalar(raw[:, None])).kernel)
    np.testing.assert_allclose((raw * value).kernel, (mv.scalar(raw[:, None]) * value).kernel)
    np.testing.assert_allclose((value + raw).kernel, (value + mv.scalar(raw[:, None])).kernel)
    np.testing.assert_allclose((raw - value).kernel, (mv.scalar(raw[:, None]) - value).kernel)
    np.testing.assert_allclose((value / raw).kernel, (value * mv.scalar(1 / raw[:, None])).kernel)
    np.testing.assert_allclose((raw / mv.scalar([4.0])).kernel, mv.scalar(raw[:, None] / 4).kernel)
