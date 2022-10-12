import numpy as np

from numga.backend.jax.context import JaxContext
from numga.backend.jax.operator import *


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


def test_expr():
	"""check and see what a compiled expression looks like"""
