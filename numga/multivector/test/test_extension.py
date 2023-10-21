
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import random_motor, random_non_motor


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
])
def test_study(descr):
	"""test basic properties of study numbers"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	# study numbers constructed from products of the even subalgebra
	# for general randomly sampled study numbers, square roots might not exist in GAs over the real numbers
	# as an example, in d<4, we have degenerate scalar study numbers; and negative scalars wont fly
	# this is fine though, since we are only using study number functionality for motor normalization
	m = random_non_motor(context, shape=(10, ))
	s = m.symmetric_reverse_product()
	assert s.subspace.inside.study()

	print('inverse')
	Is = s.inverse()
	r = Is * s - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)

	print('square root')
	rs = s.square_root()
	r = rs * rs - s
	npt.assert_allclose(r.values, 0, atol=1e-9)
	zero = s * 0
	rs = zero.square_root()
	npt.assert_allclose(rs.values, 0, atol=1e-9)

	print('inverse square root')
	Irs = s.inverse().square_root()

	Irs2 = s.square_root().inverse()
	npt.assert_allclose(Irs2.values, Irs.values, atol=1e-6)

	r = s * Irs * Irs - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1), (1, 0, 2), (1, 2, 0),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1),
])
def test_bisect(descr):
	"""Test roundtrip qualities of bisection based log and exp"""
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)

	for i in range(10):
		m = random_motor(ga)

		# check exact inverse props
		r = m.motor_log().exp()
		npt.assert_allclose(m.values, r.values, atol=1e-9)


def test_interpolate():
	"""test various methods of motor interpolation"""
	descr = (3, 0, 0)
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)

	a = random_motor(ga)
	b = random_motor(ga)

	t = 0.33
	def relative_lerp(a, b, t):
		d = (~a * b).motor_log()
		return a * (d * t).exp()
	def absolute_lerp(a, b, t):
		"""Note: this one is broken!"""
		a, b = a.motor_log(), b.motor_log()
		i = (a * (1 - t) + b * t)
		return i.exp()
	def linear_lerp(a, b, t):
		return (a * (1-t) + b * t).normalized()
	def bisection_lerp(a, b, t, n=3):
		if 0 == n:
			return linear_lerp(a, b, t)
		else:
			m = (a + b).normalized()
			if t < 0.5:
				return bisection_lerp(a, m, t*2, n - 1)
			else:
				return bisection_lerp(m, b, t*2-1, n - 1)

	print(a)
	print(b)
	print()
	# print(absolute_lerp(a, b, t))
	print(relative_lerp(a, b, t))
	print(bisection_lerp(a, b, t))
	# print(linear_lerp(a, b, t))
