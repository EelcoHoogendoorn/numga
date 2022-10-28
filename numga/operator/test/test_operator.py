"""
Test new operator system
"""

import numpy as np

from numga.algebra.algebra import Algebra
from numga.operator.operator import Operator


def test_fuse():
	"""Simple entry point to debug operator fusion"""
	a = Operator(np.ones((1, 2, 3)), [None] * 3)
	b = Operator(np.ones((4, 7, 3, 6)), [None] * 4)
	c = a.fuse(b, axis=2)
	assert c.kernel.shape == (4, 7, 1, 2, 6)


def test_print_quat():
	"""Visualize quaternion multiplication table"""
	algebra = Algebra('x+y+z+')
	even = algebra.subspace.even_grade()
	op = algebra.operator.product(even, even)
	print()
	print(op)


def test_squared_norm():
	"""Visualize squared norm of motor in 4d"""
	algebra = Algebra('x+y+z+w+')
	even = algebra.subspace.even_grade()
	op = algebra.operator.symmetric_reverse_product(even)
	print()
	print(op)


def test_basic():
	"""Test some operator output spaces"""
	ga = Algebra('x+y+z+w0')
	V = ga.subspace.vector()
	Q = ga.subspace.even_grade()

	# we can slice operators over all subalgebras; including such nonclassical ones
	# we can get operators and subalgebras thereof without instantiating any vectors
	sandwich = ga.operator.sandwich(Q, V)
	assert sandwich.output == V, "sandwich should be grade preserving"

	dot = ga.operator.dot(Q, V)
	assert dot.output.equals.empty()
	dot = ga.operator.dot(V, V)
	assert dot.output.equals.scalar()


def test_commutator():
	"""Visualize the output grade of the commutator of pairs of i-j vectors"""
	print()
	algebra = Algebra('x+y+z+w+')
	n = algebra.n_dimensions + 1
	dims = range(n)
	v = [algebra.subspace.k_vector(i) for i in dims]
	r = -np.ones((n, n))
	for i in dims:
		for j in dims:
			try:
				r[i, j] = algebra.operator.anti_commutator(v[i], v[j]).output.grade()
			except:
				pass
	print(r)

from numga.multivector.test.util import random_subspace
import numpy.testing as npt

def test_inertia():
	"""Test equivalence of composed ternary operators to their direct expression form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z0')
	context = NumpyContext(algebra)

	P = context.subspace.antivector()
	B = context.subspace.bivector()
	# I = context.operator.inertia(P, B)

	p = random_subspace(context, P, (10,))
	b = random_subspace(context, B, (10,))

	p = p.normalized()
	direct = p.regressive(p.commutator(b))
	I = p.inertia_map()
	operator = I(b)
	npt.assert_allclose(direct.values, operator.values)
