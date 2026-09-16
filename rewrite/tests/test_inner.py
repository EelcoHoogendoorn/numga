"""The inner product: grade |r - s| selection, wired to the | operator on every operand kind."""

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from numga.algebra import Algebra


def test_inner_of_equal_grades_is_the_scalar_product():
    ga = Algebra("x+y+z+")
    mv = NumpyContext(ga).multivector
    a = mv.vector(np.array([1.0, 2.0, 3.0]))
    b = mv.vector(np.array([-2.0, 0.5, 4.0]))
    np.testing.assert_allclose((a | b).kernel, a.scalar_product(b).kernel)
    B = mv.bivector(np.array([0.3, -0.2, 0.5]))
    np.testing.assert_allclose((B | B).kernel, B.scalar_product(B).kernel)


def test_inner_of_bivector_and_vector_is_the_commutator_in_three_dimensions():
    ga = Algebra("x+y+z+")
    mv = NumpyContext(ga).multivector
    B = mv.bivector(np.array([0.3, -0.2, 0.5]))
    v = mv.vector(np.array([1.0, 2.0, 3.0]))
    inner = B | v
    assert inner.gatype.output_subspace == ga.subspace.vector()
    np.testing.assert_allclose(inner.kernel, B.commutator(v).kernel)


def test_inner_operator_works_on_subspaces_gatypes_and_open_extensors():
    ga = PGA3D
    ctx = NumpyContext(ga)
    mv = ctx.multivector
    V = ga.subspace.vector()
    Vector = ga.gatype.vector()
    x = mv.vector(np.array([1.0, 2.0, 3.0, 4.0]))

    open_map = x | V                              # vector with an open vector slot
    assert open_map.arity == 1
    np.testing.assert_allclose(open_map.kernel, x.scalar_product(V).kernel)
    np.testing.assert_allclose((V | x).kernel, (x | V).kernel)

    form = ctx.lower(Vector | Vector)             # the metric as an arity-2 form
    assert form.arity == 2
    np.testing.assert_allclose(np.asarray(form.kernel).squeeze(), np.diag([1.0, 1.0, 1.0, 0.0]))


def test_inner_keeps_the_grade_difference_only():
    ga = PGA3D
    mv = NumpyContext(ga).multivector
    point = mv.antivector(np.array([1.0, 2.0, 3.0, 1.0]))
    plane = mv.vector(np.array([0.0, 0.0, 1.0, -3.0]))
    line = point | plane                          # grade |3 - 1| = 2: the line through the point normal to the plane
    assert line.gatype.output_subspace == ga.subspace.bivector()
    np.testing.assert_allclose(line.kernel, (point * plane).restrict[2].cast(ga.subspace.bivector()).kernel)
