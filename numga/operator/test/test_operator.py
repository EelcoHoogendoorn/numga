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


def test_print_lorentz():
	"""Visualize lorentz force law"""
	algebra = Algebra('x+y+z+t-')
	b = algebra.subspace.bivector()
	v = algebra.subspace.vector()
	opi = algebra.operator.inner(v, b)
	opc = algebra.operator.commutator(v, b)
	assert opi.equals(opc)
	print()
	print(opc)


def test_print_dual():
	"""Visualize dual"""
	algebra = Algebra((3,0,0))
	V = algebra.subspace.vector()
	op = algebra.operator.dual(V)
	print()
	print(op)
	print(op.kernel)


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


def point_embed(
		context,
		distances
):
	"""Code to embed PGA points in spaces of arbitrary signature

	Distances denote the distances of points away from the origin
	The origin is the last axis of the algebra
	"""
	*axes, origin = context.multivector.basis()     # arbitrarily pick last axis as the origin
	model_space = context.subspace.vector().difference(origin.subspace)
	distances = context.multivector(model_space, distances)
	translator = ((origin / -2) ^ distances).exp()
	# need the dual-inverse to avoid picking up that annoying minus sign in alternating dimensions
	return translator >> origin.dual_inverse()


def test_inertia_more():
	"""Test seperability of inertia"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+w0')
	context = NumpyContext(algebra)

	P = context.subspace.antivector()
	B = context.subspace.bivector()
	# I = context.operator.inertia(P, B)

	p = random_subspace(context, P, (10,))
	b = random_subspace(context, B, (10,))

	def make_n_cube(N):
		b = ((np.arange(2 ** N)[:, None] & (1 << np.arange(N))) > 0)
		return (2 * b - 1)

	def make_n_rect(N):
		return make_n_cube(N) * (np.arange(N) + 1)
	p = point_embed(context, make_n_rect(algebra.n_dimensions-1) * 0.1)

	p = p.normalized()
	I = p.inertia_map().sum(axis=0)

	print(I.kernel)
	momentum = I(b)
	print(momentum)

	# FIXME: these partial inertia maps return a full antibivector
	I = p.inertia_map(B.degenerate()).sum(axis=0)
	print(I.kernel)
	I = p.inertia_map(B.nondegenerate()).sum(axis=0)
	print(I.kernel)



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
	algebra = Algebra((2,0,0))
	context = NumpyContext(algebra)

	Q = context.subspace.even_grade()
	q = context.multivector(Q, np.arange(len(Q))+1)
	op = context.operator.product(Q, Q)
	o = op.partial({0: q})
	print()
	print(o.kernel)


def test_quat_matrix_cl2():
	"""Test reduction of quat multiplication to matrix form"""
	import jax
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext
	from numga.backend.jax.operator import JaxSparseOperator
	algebra = Algebra((2,0,0))
	context = JaxContext(algebra, otype=JaxSparseOperator)

	Q = context.subspace.even_grade()
	V = context.subspace.vector()
	q = context.multivector(Q, np.arange(len(Q))+1)
	op = context.operator.sandwich(Q, V)
	op = context.operator.reverse(Q)

	op = context.operator.product(Q, V)
	print(op.kernel)
	print(op.operator.axes)
	return



	for term in op.precompute_sparse_tensor(output_axes=(1, 3)):
		print(term)
	# return

	# print(context.operator.sandwich(Q, V))
	@jax.jit
	def to_matrix(r):
		return r.sandwich_map(V)
	
	def compute_kernel(values):
		return to_matrix(context.multivector(Q, values)).kernel

	@jax.jit
	def explicit(r):
		# map rotor sandwich to matrix, in vectorized manner
		s, b = r.values[..., 0], r.values[..., 1] 	# unpack to scalar and bivector
		d = s**2 - b**2 	# diagonal elements
		o = 2 * s * b	 	# off-diagonal elements
		out = jnp.stack([d, o, -o, d], axis=-1)
		return out.reshape(r.shape[:-1] + (2,2))

	@jax.jit
	def explicit_for(r):
		# map rotor sandwich to matrix, in vectorized manner
		out = jnp.empty(r.shape[:-1] + (2,2))
		r = r.values
		def product_term(idx, sign):
			prod = sign
			for i in idx:
				prod *= r[..., i]
			return prod
		for idx, terms in op.precompute_sparse_tensor(output_axes=(1, 3)):
			t = sum(product_term(*foo) for foo in terms)
			out = out.at[idx].set(t)
		return out

	def explicit_for_roll(r):
		# map rotor sandwich to matrix, in vectorized manner
		o = np.empty((2,2))
		o[0,0] = +r[0] * r[0] - r[1] * r[1]
		o[0,1] = +r[0] * r[1] + r[0] * r[1]
		o[1,0] = -r[0] * r[1] - r[0] * r[1]
		o[1,1] = +r[0] * r[0] - r[1] * r[1]
		return o

	def explicit_for_roll_opt(r):
		# map rotor sandwich to matrix, in vectorized manner
		o = np.empty((2,2))
		d = r[0] * r[0] - r[1] * r[1]
		o = 2 * r[0] * r[1]
		o[0,0] = d
		o[0,1] = o
		o[1,0] = -o
		o[1,1] = d
		return o


	# lets print the compiled jax code, as traced for a simple rotor/complex number input,
	# so we can check how the code compiles down.
	# if all is well, we would expect to see soemthing like
	# c = r[0]**2-r[1]**2
	# s = 2*r[0]*r[1]
	# and this then being packed into a 2x2 matrix output
	# as the totality of all traced and optimized jax operations
	print()
	print(jax.make_jaxpr(explicit)(q))
	print("--- HLO ---")
	print()
	print(jax.make_jaxpr(explicit_for)(q))
	return
	# use abstract shape to avoid constant folding
	abstract_q = jax.ShapeDtypeStruct(q.values.shape, q.values.dtype)
	print(jax.jit(explicit).lower(abstract_q).compile().as_text())
	print(op.kernel)


def test_sparse_partial_equivalence():
	"""Test that sparse and dense partial application yield same results"""
	import jax
	import numpy as np
	import numpy.testing as npt
	from numga.backend.jax.context import JaxContext
	from numga.backend.jax.operator import JaxSparseOperator, JaxEinsumOperator
	
	algebra = Algebra((2,0,0))
	
	# Dense context
	ctx_dense = JaxContext(algebra, otype=JaxEinsumOperator)
	Q_d = ctx_dense.subspace.even_grade()
	V_d = ctx_dense.subspace.vector()
	op_d = ctx_dense.operator.sandwich(Q_d, V_d)
	
	# Sparse context
	ctx_sparse = JaxContext(algebra, otype=JaxSparseOperator)
	Q_s = ctx_sparse.subspace.even_grade()
	V_s = ctx_sparse.subspace.vector()
	op_s = ctx_sparse.operator.sandwich(Q_s, V_s)
	
	# Random input
	rng = np.random.default_rng(42)
	q_val = rng.normal(size=len(Q_d))
	
	q_d = ctx_dense.multivector(Q_d, q_val)
	q_s = ctx_sparse.multivector(Q_s, q_val)
	
	# Partial apply
	res_d = op_d.partial({0: q_d, 2: q_d})
	res_s = op_s.partial({0: q_s, 2: q_s})
	
	# Check kernels
	print("Dense kernel shape:", res_d.kernel.shape)
	print("Sparse kernel shape:", res_s.kernel.shape)
	
	npt.assert_allclose(res_d.kernel, res_s.kernel, atol=1e-5)
	print("Kernels match!")


def test_levi_matrix():
	"""Test reduction of Levi-Civita symbol to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+')
	context = NumpyContext(algebra)

	V = context.subspace.vector()
	op = context.operator.cross_product(V, V)
	v = context.multivector.vector(np.arange(3)+1)
	o = op.partial({0: v})
	print()
	print(o.kernel)
	print(np.einsum('i,ijk->jk', v.values, op.kernel))



def test_quat_sandwich_matrix():
	"""Test reduction of quat sandwich multiplication to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+w0')
	algebra = Algebra((4,1,0))	# cga
	context = NumpyContext(algebra)

	Q = context.subspace.even_grade()
	V = context.subspace.k_vector(3)
	q = random_subspace(context, Q, (1,)).normalized()
	# q = context.multivector.even_grade(np.array([1,2,3,4]))
	op = context.operator.sandwich(Q, V)
	print(np.count_nonzero(op.kernel))
	return
	o = op.partial({0:q, 2:q})
	print()
	# print(o.operator.axes)
	print(o.kernel[0])
	A = context.subspace.k_vector(-3)
	# q = random_subspace(context, Q, (1,)).normalized()
	op = context.operator.sandwich(Q, A)
	o = op.partial({0:q, 2:q})
	print()
	# print(o.operator.axes)
	print(o.kernel[0, ::-1,::-1])


def test_projection_grade():
	"""Test reduction of camera projection to matrix form"""
	from numga.backend.numpy.context import NumpyContext
	algebra = Algebra('x+y+z+w0')
	context = NumpyContext(algebra)

	V = context.subspace.vector()
	B = context.subspace.bivector()
	T = context.subspace.antivector()

	a = random_subspace(context, V).normalized()
	b = random_subspace(context, B).normalized()

	print(a.project(b))



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
