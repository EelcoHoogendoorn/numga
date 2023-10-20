import numpy as np

from numga.backend.numpy.context import *
from numga.backend.numpy.operator import NumpySparseOperator

import pytest

def test_basic():
	print()
	ga = NumpyContext('x+y+z+w0')
	Q, V, B = ga.subspace.even_grade(), ga.subspace.vector(), ga.subspace.bivector()
	q, v = ga.multivector(Q), ga.multivector(V)
	print(q)
	print(v)

	output = q.sandwich(v)

	op = q.sandwich_map()
	print('our projective matrix')
	print(op.operator.kernel)
	print(op.operator.axes)
	print(op(v))

	def print_op(op):
		print(op.operator.axes)
		print(op.intensity)
		print(op.sparsity)
	# FIXME: lets try and make partial version of sparse too
	S = ga.operator.sandwich(Q, B)
	print_op(S)
	S = ga.operator.sandwich(Q, V)
	print_op(S)
	S = ga.operator.product(Q, Q)
	print_op(S)



def test_operator():
	context = NumpyContext('x+y+z+')
	Q, V = context.subspace.even_grade(), context.subspace.vector()
	op = context.operator.sandwich(Q, V)
	print()
	print(op.precompute)
	print(op.precompute_einsum_partial((0, 2)))

	inputs = [context.multivector(s) for s in op.inputs]
	output = op(*inputs)
	print(inputs)
	print(output)
	print(inputs[0].sandwich(inputs[1]))


def test_sparse_operator():
	print()
	context = NumpyContext('x+y+z+w0', otype=NumpySparseOperator)

	Q, V = context.subspace.even_grade(), context.subspace.vector()
	op = context.operator.sandwich(Q, V)
	assert op.output == V

	print(op.sparsity)
	for row in op.precompute_sparse:
		print(row)
	# FIXME: test broadcasting behavior
	inputs = [context.multivector(subspace=s, values=np.ones(len(s))) for s in op.inputs]
	output = op(*inputs)
	print(output)


def test_translate():
	context = NumpyContext('x-y+z+w0')
	x, y, z, w = context.multivector.basis()
	# construct basis translators
	xw = (x ^ w) + 1
	yw = (y ^ w) + 1
	zw = (z ^ w) + 1
	# basis rotors
	xy = x ^ y
	yz = y ^ z
	zx = z ^ x


	T = xw * yw
	print(T)
	print(T.symmetric_reverse_product())
	print(xy.symmetric_reverse_product())
	print(xy * zw)
	print((xy * zw).symmetric_reverse_product())
	# embed a point
	v = (x + w).dual()
	print('point')
	print(v)
	print(T << v)
	print(T >> v)
	print(T >> x.dual())   # leave directions unchanged


def test_operator_composition():
	ga = NumpyContext('x+y+z+')
	V = ga.subspace.vector()
	R = ga.multivector.vector(values=np.random.normal(size=len(V)))
	op = ga.algebra.operator.wedge(V, ga.subspace.x)
	op = ga.operator(op).partial({1:ga.multivector.x})
	print()
	print(op.kernel)
	print(op.partial({0:R}).kernel)
	print(R ^ ga.multivector.x)
	print()


def test_inverse():
	def check_inverse(x, i):
		assert np.allclose((x * i - 1).values, 0, atol=1e-9)
		assert np.allclose((i * x - 1).values, 0, atol=1e-9)

	ga = NumpyContext('x+y+z+w+')
	V = ga.subspace.vector()
	x = ga.multivector.vector(values=np.random.normal(size=(2, len(V))))
	check_inverse(x, x.la_inverse())
	check_inverse(x, x.inverse())

	V = ga.subspace.even_grade()
	x = ga.multivector.even_grade(values=np.random.normal(size=(len(V))))
	check_inverse(x, x.la_inverse())
	check_inverse(x, x.inverse())

	V = ga.subspace.multivector()
	x = ga.multivector.multivector(values=np.random.normal(size=(len(V))))
	check_inverse(x, x.la_inverse())
	q = x.conjugate() * x.involute() * x.reverse()
	qq = x.inverse_factor()
	op = ga.operator.inverse_factor(x.subspace)
	print(np.count_nonzero(op.kernel))
	assert np.allclose(q.values, qq.values)


	x = ga.multivector.x + 2
	check_inverse(x, x.la_inverse())
	q = x.inverse_factor()
	check_inverse(x, q / x.scalar_product(q))

	with pytest.raises(Exception):
		ga = NumpyContext('x+w0')
		x = ga.multivector.w
		i = x.la_inverse()


