"""Regressions for the actionable rewrite review findings."""

import gc
from fractions import Fraction
from itertools import islice
import weakref

import numpy as np
import pytest

from numga import Algebra, NumpyContext


def test_contexts_and_bound_constructors_are_collectible_after_application():
    algebra = Algebra("x+y+z+")

    def evaluate():
        context = NumpyContext(algebra)
        other = NumpyContext(algebra)
        mv = context.multivector
        vector = mv.vector([1, 2, 3])
        motor = mv.bivector([.1, .2, .3]).exp()
        (motor >> algebra.gatype.vector())(other.multivector.vector([3, 2, 1]))
        motor >> vector
        return weakref.ref(context), weakref.ref(other)

    references = [reference for _ in range(4) for reference in evaluate()]
    gc.collect()
    assert all(reference() is None for reference in references)


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


@pytest.mark.parametrize("signature", ("x+y+", "x-y+z+", "x+y+z+w0", "x+y+z+p+n-"))
def test_inner_product_of_lower_left_grade_keeps_compact_grade_arrays(signature):
    context = NumpyContext(signature)
    mv = context.multivector
    np.testing.assert_array_equal((mv.x | mv.xy).kernel, (mv.x * mv.xy).kernel)
    grades = context.algebra.grade(np.asarray(context.gatype.vector().output_subspace.masks))
    assert grades.dtype == np.uint8
    assert context.gatype is context.algebra.gatype
    assert context.subspace is context.algebra.subspace
    np.testing.assert_array_equal((~mv.xy).kernel, mv.xy.reverse().kernel)


def test_signed_subspace_comparison_uses_the_canonical_gatype_factory():
    from numga.gatype.gatype import _comparison_gatype

    algebra = Algebra("x+y+z+")
    space = algebra.subspace("yz zx xy")
    assert _comparison_gatype(space) is algebra.gatype(space)


def test_sandwich_symmetry_removes_exact_zeros_before_binding():
    algebra = Algebra("x+y+z+w0")
    even, point = algebra.gatype.even(), algebra.gatype.antivector()
    expression = even.sandwich(point)
    assert expression.output_subspace is point.output_subspace
    kernel = expression.kernel.values
    np.testing.assert_array_equal(kernel, kernel.swapaxes(1, 3))
    mv = NumpyContext(algebra).multivector
    motor = mv.even(np.arange(8) / 10)
    vertex = mv.antivector([1, 2, 3, 1])
    result = expression(motor, vertex, motor)
    raw = motor * vertex * ~motor
    np.testing.assert_allclose(result.cast(raw.subspace).kernel, raw.kernel, atol=1e-14)
