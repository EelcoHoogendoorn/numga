import numpy as np

from numga.backend.torch.context import TorchContext
from numga.backend.torch.operator import *


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


