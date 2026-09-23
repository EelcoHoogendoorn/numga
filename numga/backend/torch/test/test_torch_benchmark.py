import numpy as np
import torch

from numga.backend.torch.context import TorchContext
from numga.backend.torch.operator import *

from numga.algebra.algebra import Algebra
from numga.multivector.test.util import random_motor, random_subspace


def test_sparse_operator():
	print()
	ga = TorchContext('x+y+z+', otype=TorchSparseOperator)
	Q, V = ga.subspace.even_grade(), ga.subspace.vector()
	q, v = ga.multivector(Q), ga.multivector(subspace=V, values=torch.ones(len(V)))
	print(q)
	v = v.at[0].set(10)
	print(v)

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
	ga = TorchContext('x+y+z+w0', otype=TorchSparseOperator)

	R = ga.subspace.even_grade()
	V = ga.subspace.vector()


	bs = 1024*16
	dv = torch.randn(size=(bs, len(V)))
	dr = torch.randn(size=(bs, len(R)))

	r = ga.multivector(R, dr)
	v = ga.multivector(V, dv)
	print(v)
	sandwich = ga.operator.sandwich(R, V)
	print(sandwich.kernel.shape)
	print(sandwich.sparsity)


	from time import time

	t = time()
	for i in range(100):
		r.sandwich(v)
	print('run time')
	print(time()-t)


def check_inverse(x, i, atol=1e-9):
	# print(x.subspace, i.subspace)
	print(i.values.shape)
	assert np.allclose((x * i - 1).values, 0, atol=atol)
	assert np.allclose((i * x - 1).values, 0, atol=atol)

def mytime(callable, msg='', iterations=30):
	# callable = torch.jit.trace(callable)
	q = callable()	# warmup
	print(q.subspace)		# make sure warmup cant be optimized
	import contexttimer
	with contexttimer.Timer() as t:
		rs = [callable() for i in range(iterations)]
	print(msg, t.elapsed)
	return rs[-1]


def test_inverse():
	"""test some inversion in 6 dimensions"""
	print()
	np.random.seed(0)
	from numga.backend.jax.operator import JaxSparseOperator	# very slow compile in high dims; no actual performance gain
	ga = TorchContext(Algebra.from_pqr(8, 0, 0), dtype=torch.float32)#, otype=JaxSparseOperator)
	N = 100

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
	V = ga.subspace.from_grades([0, 1])
	x = random_subspace(ga, V, (N,))

	def foo(x):
		q = x.symmetric_reverse_product()
		i = q.inverse_la()
		return ~x * i

	i = mytime(lambda: x.inverse_shirokov(), 'sh')
	# check_inverse(x, i, atol=1e-6)
	i = mytime(lambda: x.inverse_la(), 'la')
	# check_inverse(x, i, atol=1e-6)
	# check_inverse(x, mytime(lambda: foo(x), 'lah'), atol=1e-9)

