import numpy as np
import pytest
from numpy import testing as npt

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext
from numga.multivector.test.util import *


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), #(2, 2, 1),
])
def test_invariant_decomposition(descr):
	np.random.seed(1)
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)
	for i in range(10):
		b = random_motor(ga).select[2]

		l, r = b.decompose_invariant()
		assert_close(l.inner(r), 0)
		assert_close(l.squared().select.nonscalar(), 0)
		assert_close(r.squared().select.nonscalar(), 0)

		assert_close(b, l + r)

		L = l.exp()
		R = r.exp()
		# test that component bivectors commute under exponentiation
		assert_close(L * R, R * L)


@pytest.mark.parametrize('descr', [
	(1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1),
])
def test_motor_decompose_euclidean(descr):
	ga = NumpyContext(descr)
	m = random_motor(ga, (10,))
	t = m.motor_translator()
	# print(t)
	r = m.motor_rotor()

	assert_close(t * r, m)
	# alternate method of constructing translator
	tt = (m * ~r).select.translator()
	assert_close(tt, t)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0),
	(1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1),
	(1, 1, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0),
	# (3, 0, 0)
])
def test_motor_split(descr):
	"""test splitting of motors relative to a specific point"""
	np.random.seed(10)
	ga = NumpyContext(descr)
	N = 10
	m = random_motor(ga, (N,))

	# test for both canonical origin, or some arbitrary point
	o = ga.multivector.basis()[-1].dual()
	os = [o, random_motor(ga, (N,)) >> o]

	for o in os:
		t, r = m.motor_split(o)
		# print(m)
		# print(t.subspace)
		# print(t.subspace.is_subalgebra)
		# print(t.subspace)
		# print(r.subspace.is_subalgebra)
		# check that it is a valid decomposition of the original motor
		assert_close(t * r, m)
		# check that r leaves o invariant
		assert_close(r >> o, o, atol=1e-6)
		# in d<6, both of these should be simple motors
		assert_close(t * ~t, 1, atol=1e-6)
		assert_close(r * ~r, 1, atol=1e-6)
		# and their respective bivectors are orthogonal
		assert_close(t.motor_log().inner(r.motor_log()), 0)

