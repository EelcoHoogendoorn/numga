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


def test_print_Levi_Civita():
	"""Visualize Levi-Civita symbol"""
	algebra = Algebra('x+y+z+')
	V = algebra.subspace.vector()
	op = algebra.operator.dual(algebra.operator.outer(V, V))
	op = algebra.operator.cross_product(V, V)
	print()
	print(op)
	print(op.kernel)


def test_print_sta():
	"""Visualize sta multiplication table"""
	algebra = Algebra('w+x+y+t-')
	even = algebra.subspace.even_grade()
	op = algebra.operator.symmetric_reverse_product(even)
	print()
	print(op)


def test_print_maxwell():
	"""Visualize sta multiplication table"""
	algebra = Algebra('x+y+z+t-')
	one, bi = algebra.subspace.vector(), algebra.subspace.bivector()
	op = algebra.operator.product(one, bi)
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
	V = ga.subspace.bivector()
	Q = ga.subspace.even_grade()

	# we can slice operators over all subalgebras; including such nonclassical ones
	# we can get operators and subalgebras thereof without instantiating any vectors
	sandwich = ga.operator.sandwich(Q, V)
	assert sandwich.output == V, "sandwich should be grade preserving"
	kernel = sandwich.kernel
	sparsity = np.count_nonzero(kernel) / kernel.size
	print(1/sparsity)
	print(np.count_nonzero(kernel))

	# dot = ga.operator.dot(Q, V)
	# assert dot.output.equals.empty()
	# dot = ga.operator.dot(V, V)
	# assert dot.output.equals.scalar()


def test_commutator():
	"""Visualize the output grade of the commutator of pairs of i-j vectors"""
	print()
	algebra = Algebra.from_pqr(8, 0, 0)
	n = algebra.n_dimensions + 1
	dims = range(n)
	v = [algebra.subspace.k_vector(i) for i in dims]
	r = np.empty((n, n), dtype=object)

	for i in dims:
		for j in dims:
			try:
				s = str(np.unique(algebra.operator.commutator(v[i], v[j]).output.grades()))
				r[i, j] = s.rjust(5, ' ')
			except:
				pass
	print('\n'.join(' '.join(q) for q in r))


def test_square_signs():
	print()
	algebra = Algebra('w+x+y+z+t-')
	op = algebra.operator.squared(algebra.subspace.full())
	# op = algebra.operator.dual(algebra.subspace.full())
	print(op)


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


def test_quat_matrix():
	"""Test reduction of quat multiplication to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+')
	context = NumpyContext(algebra)

	Q = context.subspace.even_grade()
	q = context.multivector(Q, [1, 2, 3, 4])
	op = context.operator.product(Q, Q)
	o = op.partial({0: q})
	print()
	print(o.kernel)


def test_quat_sandwich_matrix():
	"""Test reduction of quat sandwich multiplication to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+w0')
	context = NumpyContext(algebra)

	Q = context.subspace.even_grade()
	V = context.subspace.vector()
	q = random_subspace(context, Q, (1,)).normalized()
	op = context.operator.sandwich(Q, V)
	o = op.partial({0:q, 2:q})
	print()
	print(o.kernel)


def test_projection_matrix():
	"""Test reduction of camera projection to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+w0')
	context = NumpyContext(algebra)

	V = context.subspace.vector()
	B = context.subspace.bivector()
	T = context.subspace.antivector()

	# tet = random_subspace(context, T, (4,)).normalized()
	# camera_origin = tet[0]
	# camera_plane = tet[1] & tet[2] & tet[3]

	camera_origin = random_subspace(context, T).normalized()
	camera_plane = random_subspace(context, V).normalized()

	M = context.multivector
	camera_origin = (M.vector() + M.z*0 + M.w).dual_inverse()
	camera_plane = M.vector() + M.z + M.w


	def projection(S):
		op = algebra.operator.wedge(V, algebra.operator.regressive(T, S))
		assert op.output == S
		return op

	def static_projection(origin, plane, S):
		P = projection(S)
		return context.operator(P).partial({0: plane, 1: origin})

	# P = projection(B)
	P = static_projection(camera_origin, camera_plane, T)
	# P = projection(B)
	print()
	print(P.kernel.T)
	print(P.operator.axes)
	# print(P.inverse().kernel)


def test_hitzer_fusion():
	"""Test fused vs nonfused hitzer operator difference"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra.from_pqr(5,0,0)

	# V = context.subspace.vector()
	B = algebra.subspace.bivector()
	# T = context.subspace.antivector()
	op = algebra.operator
	rp = op.product(B, op.reverse(B)).symmetry((0,1))
	foo = op.product(rp, op.scalar_negation(rp)).symmetry((0,1,2,3))
	# bar = op.product(rp.subspace, op.scalar_negation(rp.subspace)).symmetry((0,1))
	bar = B.squared().symmetric_scalar_negation()
	print(foo.subspace)
	print(bar.subspace)



# def test_projection_matrix_aspirational():
# 	"""Test reduction of camera projection to matrix form"""
# 	from numga.backend.numpy.context import NumpyContext
# 	algebra = Algebra('x+y+z+w0')
# 	context = NumpyContext(algebra)
#
# 	V = context.subspace.vector()
# 	B = context.subspace.bivector()
# 	T = context.subspace.antivector()
# 	Q = context.subspace.even_grade()
#
# 	camera_origin = random_subspace(context, T).normalized()
# 	camera_plane = random_subspace(context, V).normalized()
# 	camera_frame = random_subspace(context, Q).normalized()
#
# 	rotation = camera_frame << T
# 	# camera_frame.sandwich
# 	projection = camera_plane ^ (camera_origin & T)
# 	camera_transform = projection(rotation)
