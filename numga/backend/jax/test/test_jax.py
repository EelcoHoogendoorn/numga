import time

import contexttimer
import jax
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

from numga.algebra.algebra import Algebra
from numga.backend.jax.context import JaxContext
from numga.backend.jax.operator import *
from numga.multivector.test.util import random_motor, random_subspace


def test_sparse_operator():
	print()
	ga = JaxContext('x+y+z+', otype=JaxSparseOperator)
	Q, V = ga.subspace.even_grade(), ga.subspace.vector()
	q, v = ga.multivector(Q), ga.multivector(subspace=V, values=jnp.ones(len(V)))
	print(q)
	v = v.at[0].set(10)
	print(v)
	# return

	output = q.sandwich(v)
	print('output')
	print(output)

	print('op')
	op = q.sandwich_map()
	print(op.kernel)
	op = op.at[0, :].set(0)
	print(op.kernel)
	print(op.operator.axes)
	print(op(v))


def test_performance():
	"""test different memory layout and operator implementations"""

	np.random.seed(0)
	ga = JaxContext('x+y+z+w0', otype=JaxEinsumOperator)

	R = ga.subspace.even_grade()
	V = ga.subspace.vector()

	key = jax.random.PRNGKey(0)
	make_v = lambda v: ga.multivector(V, v)
	make_r = lambda r: ga.multivector(R, r)
	bs = 1024*16
	dv = jax.random.normal(key, (len(V), bs), np.float32)
	dr = jax.random.normal(key, (len(R), bs), np.float32)

	# bv = jax.vmap(make_v, in_axes=(0,), out_axes=0)(d.T)
	# print(bv)
	sandwich = ga.operator.sandwich(R, V)
	print(sandwich.kernel.shape)
	# FIXME: would be interesting to compare to precomputed sandwich as well,
	#  in scenario with multiple v's per R
	do_stuff = lambda r, v: r.sandwich(v)
	# out = jax.vmap(do_stuff, in_axes=(-1,-1), out_axes=-1)(r, v)

	print(sandwich)
	from time import time

	v = jax.vmap(make_v, in_axes=(-1,), out_axes=-1)(dv)
	r = jax.vmap(make_r, in_axes=(-1,), out_axes=-1)(dr)
	vmapped = jax.jit(jax.vmap(do_stuff, in_axes=(-1, -1), out_axes=-1))

	t = time()
	out = vmapped(r, v)
	print('compilation time')
	print(time()-t)
	# return

	t = time()
	for i in range(100):
		out = vmapped(r, v)
	print('run time')
	print(time()-t)

	# this should be the worse config; at least on gpu. cpu so far unimpressed
	v = jax.vmap(make_v, in_axes=(0,), out_axes=0)(dv.T)
	r = jax.vmap(make_r, in_axes=(0,), out_axes=0)(dr.T)
	vmapped = jax.jit(jax.vmap(do_stuff, in_axes=(0, 0), out_axes=0))
	t = time()
	out = vmapped(r, v)
	print('compilation time')
	print(time()-t)

	t = time()
	for i in range(100):
		out = vmapped(r, v)
	print('run time')
	print(time()-t)


def check_inverse(x, i, atol=1e-9):
	# print(x.subspace, i.subspace)
	print(i.values.shape)
	assert np.allclose((x * i - 1).values, 0, atol=atol)
	assert np.allclose((i * x - 1).values, 0, atol=atol)

def mytime(callable, msg='', iterations=30):
	callable = jax.jit(callable)
	q = callable()	# warmup
	jax.block_until_ready(q.values)
	# print(q)		# make sure warmup cant be optimized

	with contexttimer.Timer() as t:
		for i in range(iterations):
			rs = callable()
			jax.block_until_ready(rs.values)
	print(msg, t.elapsed)
	return rs


def test_isolated():

	n = 128
	K = np.random.normal(size=(1, n,n))
	print(K.shape)
	u = np.arange(n)==0

	def callable():
		def solve_one_system(a):
			return jnp.linalg.lstsq(a, u)[0]

		r = jax.vmap(solve_one_system)(K)

		# r = jnp.linalg.solve(K, u)
		return r

	q = mytime(lambda: callable(), 'isolated', iterations=30)
	for i in range(1, 100):
		q = mytime(lambda: callable(), f'isolated {i}', iterations=i)




def test_inverse():
	"""test some inversion in 6 dimensions"""
	print()
	np.random.seed(0)
	# from numga.backend.jax.operator import JaxSparseOperator	# very slow compile in high dims; no actual performance gain

	ga = JaxContext(Algebra.from_pqr(7, 0, 0), dtype=jnp.float64)#, otype=JaxSparseOperator)
	N = 10

	V = ga.subspace.vector()
	x = random_subspace(ga, V, (N,))

	# check_inverse(x, mytime(lambda: x.inverse_la(), 'la'), atol=1e-9)
	# check_inverse(x, mytime(lambda: x.inverse_shirokov(), 'sh'), atol=1e-9)

	# V = ga.subspace.even_grade()
	# x = random_subspace(ga, V, (N,))
	#
	# check_inverse(x, mytime(lambda: x.inverse_la(), 'la'), atol=1e-9)
	# # check_inverse(x, mytime(lambda: x.inverse_shirokov(), 'sh'), atol=1e-6)

	# V = ga.subspace.multivector()
	# V = ga.subspace.from_grades([0, 1])
	V = ga.subspace.even_grade()
	x = random_subspace(ga, V, (N,))

	def foo(x):
		q = x.symmetric_reverse_product()
		i = q.inverse_la()
		return ~x * i

	i = mytime(lambda: x.inverse_shirokov(), 'sh')
	check_inverse(x, i, atol=1e-2)
	i = mytime(lambda: x.inverse_la(), 'la')
	check_inverse(x, i, atol=1e-9)
	# i = mytime(lambda: x.inverse(), 'hitz')
	# check_inverse(x, i, atol=1e-9)
	i = mytime(lambda: foo(x), 'lah')
	check_inverse(x, i, atol=1e-8)


def test_solve():
	"""test some solutions of linear multivector equations"""
	np.random.seed(0)
	import time
	ga = JaxContext((3, 0, 0), dtype=jnp.float64)
	V = ga.subspace.bivector()
	b = random_subspace(ga, V) * 0.1
	h = 1+ b
	lhs = h.squared()
	rhs = h.symmetric_reverse_product()
	r = lhs.solve(rhs)
	print(r)
	print(lhs * r - rhs)