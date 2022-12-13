"""
Design notes
------------
This operator factory is designed to perform 'late binding' of subspace arguments.
That is, the 'slicing' of full multiplication tables down to the relevant subspace,
is propagated through the constructed expressions, so at the lowest level,
we only construct those parts of the multiplication table that are actually required.

This should pay off massively in performance for high dimensional algebras;
but given that this is not really the focus of this library, I somewhat question if its worth the complexity.
Just eagerly constructing all multiplication tables, and slicing out the relevant parts at the end,
would result in much more transparent and maintainable code.


Currently, sparse tensors are only used as an intermediate format, to convert cayley tables to dense tensors.
Maintaining the sparse format internally in the operator, as different expressions are combined and sliced,
would be great for scalability to high dimensional algebras. But the code is more likely to move in the other direction.

Probably, it would be best to maintain two implementations; one based on eager construction of cayley tables,
and dense tensors, for minimum code complexity; and likely optimal performance for low dimensional algebras.
And then maintain a separate set of classes, with sparse tensors, and late binding of subspace arguments.
It should then also be easy to test for equality of outcome between the two.

"""

from typing import Tuple

import numpy as np

from numga.algebra.algebra import Algebra
from numga.algebra.bitops import parity_to_sign
from numga.operator.operator import Operator
from numga.operator.sparse_tensor import SparseTensor
from numga.subspace.subspace import SubSpace
from numga.util import cache, match


class OperatorFactory:
	"""Contains constructors for a rich set of geometric linear operators,
	and acts as a cache of those operators.

	Note that the operators merely manage the 'symbolic' side of things;
	we can use these operators to reason about how operators compose, and what their input and output subspace are.
	But how those operators are stored in memory, or how we execute them,
	is deferred to backend specific operator classes.

	Perhaps its an OperatorWarehouse, as much as a Factory?

	Caching of al created operators happens on the basis of the subspace arguments,
	and the subspace class implement a flyweight pattern, to make this efficient.
	"""

	def __init__(self, algebra: Algebra):
		self.algebra = algebra

	def build(self, kernel, axes):
		return Operator.build(kernel, axes)

	@cache
	def negatives(self, v: SubSpace) -> int:
		"""Negatives signs picked up by dotting these blades with themselves"""
		return parity_to_sign(self.algebra.bit_dot(v.subspace.blades, self.algebra.negatives))

	@cache
	def identity(self, output: SubSpace) -> "Operator":
		"""Identity operator"""
		identity = np.identity(len(output), self.algebra.blade_dtype)
		return self.build(identity, (output, output.subspace))
	def diagonal(self, output: SubSpace, weights) -> "Operator":
		"""Diagonal scaling operator"""
		kernel = np.diag(weights)
		return self.build(kernel, (output, output.subspace))
	def _complement(self, v: SubSpace, blades, signs) -> Operator:
		"""Construct an operator that maps the subspace/operator to a complementary set of blades"""
		kernel = SparseTensor.from_cayley((v.subspace,), blades, signs)
		return Operator.build(kernel, (v, v.subspace.complement()))

	@cache
	def select(self, i: SubSpace, o: SubSpace) -> Operator:
		"""selection output subspace. think of better name?"""
		identity = np.identity(self.algebra.n_blades, self.algebra.blade_dtype)
		# FIXME: construction from dense full identity scales poorly!
		# NOTE: dont squeeze here; broadcasting to zero blades is part of intended functionality
		return Operator.build(
			identity.take(i.subspace.blades, axis=0).take(o.subspace.blades, axis=1),
			(i.subspace, o.subspace)
		)
	@cache
	def restrict(self, i: SubSpace, o: SubSpace) -> Operator:
		"""Restrict i to o; drop terms not in o"""
		return self.select(i, o.intersection(i))

	@cache
	def reverse(self, v: SubSpace) -> Operator:
		"""Reverse the order of all basis blades"""
		reverse_sign = parity_to_sign(self.algebra.grade(v.subspace.blades) // 2)
		return self.diagonal(v, reverse_sign)

	@cache
	def involute(self, v: SubSpace) -> Operator:
		"""Reverse the signs on all basis blades"""
		involute_sign = self.algebra.involute(v.subspace.blades)
		return self.diagonal(v, involute_sign)

	@cache
	def conjugate(self, v: SubSpace) -> Operator:
		"""Reverse the signs and order of all basis blades"""
		# FIXME: could make anti version of this, using anti-reverse
		return self.involute(self.reverse(v))

	@cache
	def right_dual(self, v: SubSpace) -> Operator:
		"""Compute right pss complement; d(v) = v * I
		"""
		return self.product(v, self.algebra.subspace.pseudoscalar())
	@cache
	def left_dual(self, v: SubSpace) -> Operator:
		"""Compute left complement; d(v) = I * v"""
		return self.product(self.algebra.subspace.pseudoscalar(), v)
	@cache
	def right_complement(self, v: SubSpace) -> Operator:
		"""Compute right complement; d(v) = v * I
		"""
		blades, swaps = self.algebra.cayley(v.subspace.blades, self.algebra.subspace.pseudoscalar().blades)
		signs = parity_to_sign(swaps) * self.negatives(v.subspace)
		return self._complement(v, blades, signs)
	@cache
	def left_complement(self, v: SubSpace) -> Operator:
		"""Compute left complement; d(v) = I * v"""
		blades, swaps = self.algebra.cayley(self.algebra.subspace.pseudoscalar().blades, v.subspace.blades)
		signs = parity_to_sign(swaps) * self.negatives(v.subspace)
		return self._complement(v, blades, signs)

	@cache
	def right_complement_dual(self, v: SubSpace) -> Operator:
		"""Dual d(x) such that v * d(v) = I
		See: eq 124 of PGA4CS"""
		blades, _ = self.algebra.cayley(self.algebra.subspace.pseudoscalar().blades, v.subspace.blades)
		_, swaps = self.algebra.cayley(v.subspace.blades, blades)
		signs = parity_to_sign(swaps)
		return self._complement(v, blades, signs)
	@cache
	def left_complement_dual(self, v: SubSpace) -> Operator:
		"""Dual d(x) such that d(v) * v = I
		See: eq 124 of PGA4CS"""
		blades, _ = self.algebra.cayley(self.algebra.subspace.pseudoscalar().blades, v.subspace.blades)
		_, swaps = self.algebra.cayley(blades, v.subspace.blades)
		signs = parity_to_sign(swaps)
		return self._complement(v, blades, signs)

	@cache
	def left_hodge(self, v: SubSpace) -> Operator:
		"""Dual d(x) such that v * d(v) = sign * I
		where sign = e * ~e
		and e = v.nondegenerate()
		"""
		return self.left_complement_dual(self.diagonal(v, self.negatives(v)))
	@cache
	def right_hodge(self, v: SubSpace) -> Operator:
		"""Dual d(x) such that v * d(v) = sign * I
		where sign = e * ~e
		and e = v.nondegenerate()
		"""
		return self.right_complement_dual(self.diagonal(v, self.negatives(v)))

	def unary_inverse(self, operator: Operator) -> Operator:
		"""Generate the inverse `inv` of a unary operator 'op', such that inv(op(x)) == x"""
		assert operator.arity == 1
		kernel = np.linalg.inv(operator.kernel).astype(operator.kernel.dtype)
		return Operator.build(kernel, operator.axes[::-1])

	@cache
	def right_hodge_inverse(self, v: SubSpace) -> Operator:
		# FIXME: inversion should happen before binding...
		#  how can we make this more elegant?
		return self.unary_inverse(self.right_hodge(v.subspace.complement()))._build((v,))
	@cache
	def left_hodge_inverse(self, v: SubSpace) -> Operator:
		return self.unary_inverse(self.left_hodge(v.subspace.complement()))._build((v,))

	dual = right_hodge
	dual_inverse = right_hodge_inverse


	@cache
	def nondegenerate(self, v: SubSpace) -> Operator:
		"""Select blades that do not involve a null basis vector"""
		return self.select(v, v.subspace.nondegenerate())
	@cache
	def degenerate(self, v: SubSpace) -> Operator:
		"""Select blades that do involve a null basis vector"""
		return self.select(v, v.subspace.degenerate())

	# binary operators start here
	@cache
	def make_product(self, l: SubSpace, r: SubSpace, formula) -> Operator:
		"""Product with selected grades.
		Grade selection is not a regular linear operation
		so we need to apply it before binding any other operators
		"""
		# defacto output subspace will be deduced by squeeze below
		o = self.algebra.subspace.full()
		prod_blades, prod_sign = self.algebra.product(l.subspace.blades, r.subspace.blades)
		kernel = SparseTensor.from_cayley((l.subspace, r.subspace), prod_blades, prod_sign)
		return Operator(kernel, (l.subspace, r.subspace, o)).\
			grade_selection(formula).\
			_build((l, r))\
			.squeeze()

	@cache
	def geometric_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Geometric product"""
		return self.make_product(l, r, lambda l, r, o: True)
	gp = product = geometric_product

	# grade selection products
	@cache
	def wedge_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Wedge product, or outer product"""
		return self.make_product(l, r, lambda l, r, o: (l + r) == o)
	op = outer = wedge = outer_product = wedge_product
	@cache
	def inner_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Symmetric inner product; or inner for short"""
		# NOTE: eric feels inner should be called dot?
		# http://terathon.com/blog/wedge-products-dot-products-scalars-and-norms/
		return self.make_product(l, r, lambda l, r, o: np.abs(l - r) == o)
	ip = inner = inner_product
	@cache
	def scalar_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Scalar product (l * r)<0>; only grade-0 part of geometric product"""
		return self.make_product(l, r, lambda l, r, o: 0 == o)
	sp = dp = dot = scalar = dot_product = scalar_product

	@cache
	def bivector_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Bivector product (l * r)<2>; only grade-2 part of geometric product
		occurs quite often in code, so it gets a precompiled version here
		"""
		gp = self.geometric_product(l, r)
		# FIXME: make delayed binding version of restrict
		return self.restrict(gp.subspace, self.algebra.subspace.bivector()).bind(gp)

	@cache
	def left_contraction_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Left contraction product"""
		#  https://enkimute.github.io/ganja.js/examples/coffeeshop.html#IYtlGz1Ld&fullscreen
		return self.make_product(l, r, lambda l, r, o: (r - l) == o)
	lc = left_contract = left_contraction_product
	@cache
	def right_contraction_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Right contraction product"""
		return self.make_product(l, r, lambda l, r, o: (l - r) == o)
	rc = right_contract = right_contraction_product

	# notes: interior products as defined here
	#   https://projectivegeometricalgebra.org/wiki/index.php?title=Interior_products
	@cache
	def left_interior_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute *l v r"""
		return self.anti_wedge(self.left_complement(l), r).squeeze()
	lip = left_interior = left_interior_product
	@cache
	def right_interior_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute l v r*"""
		return self.anti_wedge(l, self.right_complement(r)).squeeze()
	rip = right_interior = right_interior_product
	# @cache
	# def self_right_interior_product(self, l: SubSpace) -> Operator:
	# 	"""Compute l v r*"""
	# 	return self.right_interior_product(l, l).symmetry((0, 1), +1)

	# anti-products
	def antify(self, op, *inputs: Tuple[SubSpace]) -> Operator:
		"""Compute *op(l*, v*); wrap the operator in left/right dualizatiom"""
		return self.left_complement(op(*(self.right_complement(i) for i in inputs))).squeeze()
		# not sure we want complement here; isnt hodge more general?
		# return self.right_hodge_inverse(op(*(self.right_hodge(i) for i in inputs))).squeeze()

	@cache
	def anti_geometric_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute *(r* l*); """
		# # FIXME: any symbols for this in utf16? vdot; v*?
		return self.antify(self.geometric_product, l, r)
	agp = anti_product = anti_geometric_product
	@cache
	def anti_wedge_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute *(r* ^ l*)

		Note: this differs from regressive product, in that regressive uses dual in its definitions
		which results in some differences in sign
		"""
		return self.antify(self.wedge_product, l, r)
	aop = anti_outer = anti_wedge = anti_outer_product = anti_wedge_product

	@cache
	def anti_inner_product(self, l, r):
		"""Compute *(r* . l*)"""
		return self.antify(self.inner_product, l, r)
	aip = anti_inner = anti_inner_product

	@cache
	def anti_scalar_product(self, l, r):
		"""Compute *(r* . l*)"""
		return self.antify(self.scalar_product, l, r)
	asp = adp = anti_dot = anti_dot_product = anti_scalar = anti_scalar_product

	@cache
	def anti_left_interior_product(self, l: SubSpace = None, r: SubSpace = None) -> Operator:
		"""Compute *l ^ r"""
		return self.antify(self.left_interior_product, l, r)
	alip = anti_left_interior = anti_left_interior_product

	@cache
	def anti_right_interior_product(self, l: SubSpace = None, r: SubSpace = None) -> Operator:
		"""Compute l ^ r*"""
		return self.antify(self.right_interior_product, l, r)
	arip = anti_right_interior = anti_right_interior_product

	@cache
	def regressive_product(self, l: SubSpace = None, r: SubSpace = None) -> Operator:
		"""Like the anti-wedge, but using right complement on both sides"""
		return self.dual_inverse(
			self.wedge(
				self.dual(l),
				self.dual(r)
			)
		).squeeze()
	rp = regressive = regressive_product


	# @cache
	# def bulk_right_complement(self, l: SubSpace) -> Operator:
	# 	return self.bulk(self.right_complement(l))
	# @cache
	# def weight_right_complement(self, l: SubSpace) -> Operator:
	# 	return self.weight(self.right_complement(l))
	# @cache
	# def bulk_left_complement(self, l: SubSpace) -> Operator:
	# 	return self.bulk(self.left_complement(l))
	# @cache
	# def weight_left_complement(self, l: SubSpace) -> Operator:
	# 	return self.weight(self.left_complement(l))

	# FIXME: https://projectivegeometricalgebra.org/wiki/index.php?title=Commutators
	def make_commutator(self, op, sign):
		def bind(l, r):
			return (op(l, r) + ((sign * op(r, l)).swapaxes(0, 1))).div(2).squeeze()
		return bind
	def commutator_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Commutator of the geometric product"""
		return self.make_commutator(self.product, -1)(l, r)
	commutator = commutator_product
	def anti_commutator_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Anti-commutator of the geometric product"""
		return self.make_commutator(self.product, +1)(l, r)
	anti_commutator = anti_commutator_product

	def commutator_anti_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Commutator of the geometric anti-product"""
		return self.make_commutator(self.anti_product, -1)(l, r)
	commutator_anti = commutator_anti_product
	def anti_commutator_anti_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Anti-commutator of the geometric anti-product"""
		return self.make_commutator(self.anti_product, +1)(l, r)
	anti_commutator_anti = anti_commutator_anti_product

	@cache
	def cross_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute (l ^ r)*"""
		# FIXME: is this meaningfull in general? I think not
		return self.dual(self.wedge(l, r)).squeeze()

	@cache
	def reverse_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""l * ~r"""
		return self.product(l, self.reverse(r))
	@cache
	def anti_reverse_product(self, l: SubSpace, r: SubSpace) -> Operator:
		"""l * ~r"""
		return self.antify(self.reverse_product, l, r)

	@cache
	def symmetric_reverse_product(self, x: SubSpace) -> Operator:
		"""x * ~x"""
		return self.reverse_product(x, x).symmetry((0, 1), +1)

	@cache
	def squared(self, v: SubSpace) -> Operator:
		"""x * x"""
		return self.product(v, v).symmetry((0, 1), +1)

	# FIXME: should we introduce dual/anti norms as well?
	# @cache
	# def norm_squared(self, v: SubSpace) -> Operator:
	# 	"""<x * ~x>1"""
	# 	# should produce only nonnegative scalar diagonal contributions, if metric is non-negative
	# 	return self.scalar_product(v, self.reverse(v)).symmetry((0, 1), +1)
	@cache
	def dual_norm_squared(self, v: SubSpace) -> Operator:
		"""norm_squared(*v)"""
		return self.norm_squared(self.dual(v))

	@cache
	def study_conjugate(self, subspace) -> "Operator":
		assert subspace in self.algebra.subspace.study()
		sign = parity_to_sign(subspace.grades() // 4)
		return self.diagonal(subspace, sign)
	@cache
	def study_norm_squared(self, subspace) -> "Operator":
		return self.scalar_product(subspace, self.study_conjugate(subspace))


	# def bulk_norm_squared(self, v: SubSpace) -> Operator:
	# 	# return self.scalar_product(v, self.reverse(v))
	# 	return self.norm_squared(self.bulk(v))
	# def weight_norm_squared(self, v: SubSpace) -> Operator:
	# 	# return self.anti_scalar_product(v, self.reverse(v))
	# 	return self.norm_squared(self.weight(v))
	# def projected_geometric_norm_squared(self, v):
	# 	# FIXME: this is not a multilinear op; move to vector
	# 	return self.bulk_norm_squared(v) / self.weight_norm_squared(v)

	@cache
	def full_sandwich(self, R: SubSpace, v: SubSpace) -> Operator:
		"""Compute (R * v) * ~R, the full sandwich product

		The output subspace of the sandwich product contains both the input subspace grades,
		as well as those grades modulo +4. Typically, the full sandwich product is only of academic interest,
		and its probably the grade-preserving sandwich product you are looking for

		As a practical example, the full sandwich product of a 1-vector in 3d cga with a motor, will be of grade [1, 5],
		however if the motor being sandwiched is indeed satisfies the requirements of a motor,
		the product will be grade preserving and the grade-5 part will be numericailly zero.
		"""
		return self.product(self.product(R, v), self.reverse(R)).symmetry((0, 2), +1)
		# return self.product(R, self.product(v, self.reverse(R))

	# @cache
	# def violating_sandwich(self, R: SubSpace, v: SubSpace) -> Operator:
	# 	"""Compute (R * v) * ~R
	#
	# 	Only retain the grade-violating part of the sandwich product
	# 	"""
	# 	grades = np.unique(v.grades())
	# 	return self.full_sandwich(R, v).grade_selection(lambda l, m, r, o: ~np.any(o[..., None] == grades, axis=-1))

	@cache
	def sandwich(self, R: SubSpace, v: SubSpace) -> Operator:
		"""Compute (R * v) * ~R

		Only retain the grade-preserving part of the sandwich,
		which is usually what we are interested in, since proper motors composed of bireflections,
		are in fact exactly grade preserving under sandwiching
		"""
		grades = np.unique(v.grades())
		return self.full_sandwich(R, v).grade_selection(lambda l, m, r, o: np.any(o[..., None] == grades, axis=-1))

	@cache
	def reverse_sandwich(self, R: SubSpace, v: SubSpace) -> Operator:
		"""Compute (~R * v) * R"""
		return self.sandwich(R.reverse(), v)

	@cache
	def transform(self, R: SubSpace, v: SubSpace) -> Operator:
		"""Compute (~R * v) * R, while preserving the outermorphism
		That is, transform(R, v) ^ transform(R, u) == transform(R, v ^ u)
		Note; R is assumed normalized/motorized
		"""
		parity = match(R.grades() % 2)  # this throws unless all blades in the subspace have a grade of the same parity
		S = self.sandwich(R, v)
		sign_mask = parity_to_sign(v.grades() * parity)
		return Operator(kernel=S.kernel * sign_mask[None, :, None], axes=S.axes)
	@cache
	def inverse_transform(self, R: SubSpace, v: SubSpace) -> Operator:
		"""Compute (R * v) * ~R, while preserving the outermorphism
		Note; R is assumed normalized/motorized
		"""
		return self.transform(R.reverse(), v)

	@cache
	def inertia(self, l: SubSpace, r: SubSpace) -> Operator:
		"""Compute inertia operator; l.regressive(l x r)"""
		# FIXME: it is tempting to enforce symmetry here, but it does not improve sparsity,
		#  and does force us to use a float kernel, so nevermind
		return self.regressive(l, self.commutator(l, r))#.symmetry((0, 1), +1)


	# # projections according to erik lengyel
	# @cache
	# def project(self, l: SubSpace, r: SubSpace) -> Operator:
	# 	"""Project r onto l"""
	# 	return self.anti_wedge(self.wedge(self.weight_left_complement(l), r), l).symmetry((0, 2), +1)
	#
	# @cache
	# def anti_project(self, l: SubSpace, r: SubSpace) -> Operator:
	# 	"""Project r onto l"""
	# 	return self.wedge(self.anti_wedge(self.weight_left_complement(l), r), l).symmetry((0, 2), +1)
