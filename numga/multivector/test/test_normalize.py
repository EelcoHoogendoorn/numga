
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import random_motor, random_subspace, motor_properties, random_non_motor
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
	"""Test ability to construct proper motors in a variety of spaces"""
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

	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)

	v0, v1 = motor_properties(n)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)

	npt.assert_allclose(m.values, n.values, atol=1e-9)
