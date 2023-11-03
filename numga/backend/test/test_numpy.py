import numpy as np

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import *
from numga.backend.numpy.operator import NumpySparseOperator

import pytest
from numga.multivector.test.util import random_motor, random_subspace


def test_basic():
	print()
	ga = NumpyContext('x+y+z+w0')
	Q, V, B = ga.subspace.even_grade(), ga.subspace.vector(), ga.subspace.bivector()
	q, v = random_motor(ga), random_subspace(ga, V)
	print(q)
	print(v)

	op = ga.operator.sandwich(Q, V)
	op_partial = op.partial({0: q, 2: q})
	op_partial2 = q.sandwich_map(V)   # same as above
	assert np.allclose(op_partial.kernel, op_partial2.kernel)

	r0 = (q * v * ~q).select_subspace(v.subspace)   # direct sandwich product
	r1 = q.sandwich(v)      # using optimized sandwich operator
	r2 = op_partial(v)      # using pre-bound q argument
	r3 = op_partial2(v)

	assert np.allclose(r0.values, r1.values)
	assert np.allclose(r0.values, r2.values)
	assert np.allclose(r0.values, r3.values)

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


def check_inverse(x, i, atol=1e-9):
	assert np.allclose((x * i - 1).values, 0, atol=atol)
	assert np.allclose((i * x - 1).values, 0, atol=atol)


@pytest.mark.parametrize(
	'descr', [
		# (1, 0, 0), (0, 1, 0),
		# (2, 0, 0), (1, 1, 0), (0, 2, 0),
		# (3, 0, 0), (2, 1, 0), (1, 2, 0),
		# (4, 0, 0), #(3, 1, 0), (2, 2, 0),
		(5, 0, 0),# (4, 1, 0), (3, 2, 0)
	],
)
def test_inverse_exhaustive(descr):
	"""Test general inversion cases for all multivector grade combos in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 100
	print()
	print(descr)

	all_grades = np.arange(ga.algebra.n_grades)
	import itertools
	for r in all_grades:
		for grades in itertools.combinations(all_grades, r+1):
			try:
				V = ga.subspace.from_grades(list(grades))
				print()
				print(V.simplicity, list(grades), end='')
				x = random_subspace(ga, V, (N,))
				check_inverse(x, x.inverse(), atol=1e-10)
			except:
				pass


@pytest.mark.parametrize(
	'descr', [
		(1, 0, 0), (0, 1, 0),
		(2, 0, 0), (1, 1, 0), (0, 2, 0),
		(3, 0, 0), (2, 1, 0), (1, 2, 0),
		(4, 0, 0), (3, 1, 0), (2, 2, 0),
		(5, 0, 0), (4, 1, 0), (3, 2, 0)
	],
)
def test_inverse_compare(descr):
	"""Test some general inversion cases in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 100
	V = ga.subspace.vector()
	x = random_subspace(ga, V, (N,))
	check_inverse(x, x.inverse(), atol=1e-12)
	check_inverse(x, x.inverse_la(), atol=1e-5)
	check_inverse(x, x.inverse_shirokov(), atol=1e-9)

	V = ga.subspace.even_grade()
	x = random_subspace(ga, V, (N,))
	check_inverse(x, x.inverse(), atol=1e-12)
	check_inverse(x, x.inverse_la(), atol=1e-5)
	check_inverse(x, x.inverse_shirokov(), atol=1e-8)

	V = ga.subspace.multivector()
	x = random_subspace(ga, V, (N,))
	check_inverse(x, x.inverse(), atol=1e-12)
	check_inverse(x, x.inverse_la(), atol=1e-5)
	check_inverse(x, x.inverse_shirokov(), atol=1e-9)


def test_inverse_la():
	"""test some inversion in 6 dimensions"""
	import time
	ga = NumpyContext(Algebra.from_pqr(6, 0, 0))
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
	t = time.time()
	check_inverse(x, x.inverse_la())
	print('la', time.time() - t)
	t = time.time()
	check_inverse(x, x.inverse_shirokov())
	print('shir', time.time() - t)


def test_inverse_degenerate():
	"""Test that degenerate vectors may not be invertable"""
	with pytest.raises(Exception):
		ga = NumpyContext('x+w0')
		x = ga.multivector.w
		i = x.inverse_la()

	with pytest.raises(np.linalg.LinAlgError):
		ga = NumpyContext('x+t-')
		x = ga.multivector.x + ga.multivector.t
		i = x.inverse_la()


def test_inverse_simplicifation_failure():
	"""succssive involute products sometimes fail to simplify fully.
	this results in extra recursion and poorer high dim genealization
	"""
	ga = NumpyContext(Algebra.from_pqr(5,0,0))
	V = ga.subspace.from_grades([1,2,5])
	x = random_subspace(ga, V, (1,))
	x.inverse()
	print()
	y = x.symmetric_reverse_product()
	z = y.symmetric_pseudoscalar_negation_product()
	print(x)
	print(y)
	print(z)
	return


def test_inverse_6d():
	"""test recursive inverse handles some 6d example
	"""
	ga = NumpyContext('x+y+z+a+b+c+')
	mv = ga.multivector
	x = 1 + mv.xy + mv.ab + mv.xyzabc
	i = x.inverse_la()
	print(i)    # note this particular 4-component multivector has an 8-component inverse
	check_inverse(x, i)

	i = x.inverse()
	print(i)
	check_inverse(x, i)

	# m = random_motor(ga, (1,))
	# check_inverse(m, m.inverse())

