from jax import config
config.update("jax_enable_x64", True)
import jax

import numpy as np
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1), (1, 0, 2), #(1, 2, 0),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1),
])
def test_bisect(descr):
	"""Test roundtrip qualities of bisection based log and exp"""
	np.random.seed(0)
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)

	m = random_motor(ga, (10, ))

	r = m.motor_log().exp()
	assert_close(m, r)

	# check for low iteration count
	from numga.multivector.extension.logexp import exp_bisect, motor_log_bisect
	r = exp_bisect(motor_log_bisect(m, 2), 2)
	assert_close(m, r)

	# check zero bivectors
	assert_close(ga.multivector.scalar(), exp_bisect(ga.multivector.bivector()))


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1), (1, 0, 2), #(1, 2, 0),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1),
])
def test_bisect_perf(descr):
	"""Test numerical and performance difference between different quadratic exp approximations"""
	np.random.seed(0)
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	from numga.backend.jax.context import JaxContext
	from numga.backend.jax.operator import JaxSparseOperator

	ga = JaxContext(algebra, otype=JaxSparseOperator, dtype=jax.numpy.float64)
	import time

	m = random_motor(ga, (1000, ))

	b = m.motor_log()

	def Q1(b):
		c = 1 + b / 2
		return c.squared() / c.symmetric_reverse_product()
	def Q2(b):
		c = b / 2
		return (1+c)/(1-c)

	Q1 = jax.jit(Q1)
	Q2 = jax.jit(Q2)
	q1 = Q1(b)
	q2 = Q2(b)

	for i in range(2):
		t = time.time()
		q1 = Q1(b)
		total = time.time() - t
		if i:
			print(total)

		t = time.time()
		q2 = Q2(b)
		total = time.time() - t
		if i:
			print(total)

	assert_close(q1, q2, atol=1e-9)


def test_bisect_exp_accuracy():
	"""Test the accuracy of bisection based exponentiation versus trig based implementation

	for float32, we max out at 7 iterations, with 6e-6 MSE
	for float64, we max at 17 iterations, with 9e-12 MSE

	Note that scaling down the space of bivectors,
	gives more digits of accuracy for a lower bisection iteration count
	"""
	def euler_simple_exp(b: "BiVector") -> "Motor":
		a = np.sqrt(b.symmetric_reverse_product().values)
		c = np.cos(a)
		s = np.where(a > 1e-20, np.sin(a) / a, 1)	# FIXME: sinc?
		return b.context.multivector(
			values=np.concatenate([c, s * b.values], axis=-1),
			subspace=b.context.subspace.rotor()
		)

	np.random.seed(0)
	ga = NumpyContext((6,0,1), dtype=np.float64)
	m = random_motor(ga, (10, ))
	from numga.multivector.extension.logexp import motor_sqrt_denman_beaver
	ms = motor_sqrt_denman_beaver(m)
	# print(ms)
	print(np.max(np.abs((ms*ms - m).values)))
	return

	# print(m.values.shape)
	# print(np.sign(m.select[0].values))
	# m = m * (np.sign(m.select[0].values[:, 0]))
	# print(m.values.shape)
	# return
	# print(m.values[:, 0])
	# return
	# m = m.motor_square_root()
	b = m.motor_log() #/ 10

	bp = m.motor_log_pade()
	print(np.abs((b-bp).values).max())
	return

	m_trig = euler_simple_exp(b)

	from numga.multivector.extension.logexp import exp_bisect
	for i in range(2, 20):
		m_bisect = exp_bisect(b, i)
		res = m_bisect - m_trig
		print(i, np.sqrt((res.values**2).mean()))


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
	elif True:
		i = q.inverse_la()
		r = h.squared() * i
	# else:
	# 	r = h.squared().solve(q)
	assert_close(r.symmetric_reverse_product(), 1)
