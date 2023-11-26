import numpy as np
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *
from numga.multivector.numerical_normalize import normalize_motor


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), #(3, 2, 0),
	(6, 0, 0), (5, 0, 1), (5, 1, 0), (4, 1, 1), #(4, 2, 0),
	(7, 0, 0), (6, 0, 1), (6, 1, 0), (5, 1, 1), #(5, 2, 0),
])
def test_motor_properties(descr):
	"""Test ability to construct proper normalized motors in a variety of spaces"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	m = random_motor(context, shape=(100,))

	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), #(3, 2, 0),
	# (6, 0, 0),
])
def test_motor_normalize(descr):
	"""Test motor normalization, employing the analytical manipulation of study numbers in dimensions < 6"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	v = random_non_motor(context, shape=(10,), n=5)
	m = v.normalized()

	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)


@pytest.mark.parametrize('descr', [
	(3, 0, 1),
])
def test_object_normalize(descr):
	"""Test normalization of projective objects in degenerate metric"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	# we can normalize euclidian points; make their spatial pseudoscalar 1
	v = random_subspace(context, algebra.subspace.antivector(), (10,))
	print(v.normalized())

	# cant normalize ideal points
	with pytest.raises(Exception):
		v = random_subspace(context, algebra.subspace.antivector().degenerate(), (10,))
		print(v.normalized())

	# have to normalize their duals if interested in that
	v = random_subspace(context, algebra.subspace.antivector().degenerate(), (10,))
	print(v.dual().normalized().dual_inverse())


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), #(1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), #(2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), #(3, 1, 0), (2, 1, 1), #(2, 2, 0),
	# (5, 0, 0), (4, 0, 1), #(4, 1, 0), (3, 1, 1), #(3, 2, 0),
	# (6, 0, 0),
])
def test_multivector_normalize(descr):
	"""Test normalization of arbitrary multivectors
	Works for all non-null multivectors in dimensions < 5
	"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)
	print()
	print(descr)
	print()

	for grades in all_grade_combinations(algebra):
		s = context.subspace.from_grades(grades)
		if not s.symmetric_reverse().equals.empty():
			x = random_subspace(context, s, (1,))
			y = x.normalized()
			assert_close(y.symmetric_reverse_product(), 1, atol=1e-6)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
])
def test_motor_normalize_numerical_equivalence(descr):
	"""Test motor normalization equivalence, analytical vs numerical
	good way to catch bugs in either implementation
	"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	v = random_non_motor(context, shape=(10,), n=4)

	m = v.normalized()
	n = normalize_motor(v, inner=6, outer=1)

	assert_close(m, n, atol=1e-9)

	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)

	v0, v1 = motor_properties(n)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)
