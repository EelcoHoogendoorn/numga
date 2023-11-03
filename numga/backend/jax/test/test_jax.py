import numpy as np
from jax.config import config
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


def check_inverse(x, i):
	print(x.subspace, i.subspace)
	assert np.allclose((x * i - 1).values, 0, atol=1e-9)
	assert np.allclose((i * x - 1).values, 0, atol=1e-9)


def test_inverse():
	"""test some inversion in 6 dimensions"""
	import time
	ga = JaxContext(Algebra.from_pqr(6, 0, 0), dtype=jnp.float64)
	N = 100
	V = ga.subspace.vector()
	x = random_subspace(ga, V, (N,))
	t = time.time()
	check_inverse(x, x.inverse_la())
	print('la', time.time() - t)
	t = time.time()
	check_inverse(x, x.inverse_shirokov())
	print('shir', time.time() - t)

	V = ga.subspace.even_grade()
	x = random_subspace(ga, V, (N,))
	t = time.time()
	check_inverse(x, x.inverse_la())
	print('la', time.time() - t)
	t = time.time()
	check_inverse(x, x.inverse_shirokov())
	print('shir', time.time() - t)

	V = ga.subspace.multivector()
	x = random_subspace(ga, V, (N,))
	foo = (lambda x: x.inverse_la())
	foo(x)
	t = time.time()
	check_inverse(x, foo(x))
	print('la', time.time() - t)
	foo = jax.jit(lambda x: x.inverse_shirokov())
	foo(x)
	t = time.time()
	check_inverse(x, foo(x))
	print('shir', time.time() - t)
