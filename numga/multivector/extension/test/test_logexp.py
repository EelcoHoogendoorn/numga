
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *


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

	m = random_motor(ga, (10, ))

	# check exact inverse props
	r = m.motor_log().exp()
	npt.assert_allclose(m.values, r.values, atol=1e-9)

	# check for low iteration count
	from numga.multivector.extension.logexp import exp_bisect, motor_log_bisect
	r = exp_bisect(motor_log_bisect(m, 2), 2)
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


def test_exp_high_dim():
	"""Exps in dim > 5"""
	ga = NumpyContext((6, 0, 0))

	b = random_subspace(ga, ga.subspace.bivector())
	h = 1 + b / 2
	q = h.symmetric_reverse_product()
	if False:
		# interestingly shirokov has dodgy accuracy for these kind of inputs
		i = q.inverse_shirokov()
		r = h.squared() * i
	elif False:
		i = q.inverse_la()
		r = h.squared() * i
	else:
		r = h.squared().solve(q)
	res = r.symmetric_reverse_product() - 1
	print(res)
	npt.assert_allclose(res.values, 0, atol=1e-9)