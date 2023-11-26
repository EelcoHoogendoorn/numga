import numpy as np
import numpy_indexed as npi

from numga.flyweight import FlyweightMixin
from numga.algebra.algebra import Algebra
from numga.algebra.bitops import biterator
from numga.subspace.namespaces import InNamespace, IsNamespace, SelectNamespace, RestrictNamespace
from numga.util import cache, cached_property, match


# FIXME: GA4CS recommends maintaining the sign along with the blade; could do that here?
#  perhaps becomes important when supporting non-diagonal metrics?
# FIXME: some texts seem to prefer element over blade; not sure which is more ideomatic
class SubSpace(FlyweightMixin):
	"""Describes a subspace of a full geometric algebra,
	by means of an array of bit-blades, denoting the blades present in the subspace
	"""
	def __init__(self, algebra: Algebra, blades: np.array):
		# parent algebra
		self.algebra = algebra
		self.inside = InNamespace(self)
		self.equals = IsNamespace(self)
		self.select = SelectNamespace(self)
		self.restrict = RestrictNamespace(self)
		assert npi.all_unique(blades)
		# lets throw up some defences against accidental mutations of our backing storage
		self.blades = blades.copy()
		self.blades.flags.writeable = False
		# print('constructing a subspace!')
		# print(self)

	def grades(self):
		"""Grade of each blade; number of basis vectors present in each blade

		Returns
		-------
		np.ndarray, [n_blades], uint8
			grade of each blade
		"""
		return self.algebra.grade(self.blades)

	def grade(self) -> int:
		"""Get the grade of the subspace; raises if no single grade exists, and this is a mixed-grade subspace"""
		return match(self.grades())

	def __len__(self):
		return len(self.blades)
	@property
	def is_blade(self):
		return len(self) == 1

	def __repr__(self):
		# return self.pretty_str
		return f'{self.pretty_str}: {self.named_str}'
		# return str(self.bit_blades())

	def bit_blades(self):
		"""Represent elements array in unpacked bit format, [n_elements, dimensions]"""
		# np.unpack?
		return np.rollaxis(
			np.array(list(biterator(self.blades, self.algebra.n_dimensions))),
			0
		)
	@cached_property
	def bit_str(self) -> str:
		"""bit string representation"""
		if len(self.blades) == 0:
			return ''
		names = np.array([''] * len(self)) # special case scalar to '1'
		for b in biterator(self.blades, self.algebra.n_dimensions):
			names = np.char.add(names, b.astype(str))
		return ','.join(names)
	@cached_property
	def named_str(self) -> str:
		"""string representation based on basis names"""
		if len(self.blades) == 0:
			return ''
		names = np.char.multiply('1', (self.blades == 0) * 1) # special case scalar to '1'
		for n, b in zip(
				self.algebra.description.basis_names,
				biterator(self.blades, self.algebra.n_dimensions)):
			names = np.char.add(names, np.char.multiply(n, b))
		return ','.join(names)
	@cached_property
	def pretty_str(self):
		"""Return a human-readable label for a subspace.
		Very useful for keeping our debug prints easily parsable"""
		s = self.algebra.subspace
		matches = [
			'empty',
			'scalar',
			'pseudoscalar',
			'vector',
			'bivector',     # note bivectors get precedence over antivectors in spaces where they coincide
			'antivector',
			'antibivector',
			'trivector',
			'antitrivector',
			'quadvector',
			'antiquadvector',

			'scalar_pseudoscalar',
			'study',
			'even_grade',
			'odd_grade',
			'reflection',
			'bireflection',
			'trireflection',
			'quadreflection',
			'self_reverse',
			'nonscalar',
			'multivector',
		]
		for m in matches:
			try:
				if self == getattr(s, m)():
					return f'is_{m}'
			except:
				pass
		for m in matches:
			try:
				if self in getattr(s, m)():
					return f'in_{m}'
			except:
				pass


	# FIXME: provide complete subspace construction syntax, with naming analogous to operators

	# FIXME: these compete with getattr operator construction syntax, perhaps? not working atm anyway tho...
	#  perhaps forego subspace construction in favor of operator construction; pretty easy to write V.wedge(B).subspace, after all.
	#  alternatively, write V.operator.wedge(B), if operator is desired?
	#  note we are hung up on requiring context previously
	#  just optionally pass in context? only invoked when constructing operators
	@cache
	def complement(self) -> "SubSpace":
		"""Complementary set of blades, such that self * self.complement() == I"""
		# FIXME: also use operators for these? less efficient i suppose?
		return self.algebra.subspace.from_blades(np.bitwise_xor(
			self.blades,
			self.algebra.subspace.pseudoscalar().blades)
		)
	dual = complement

	def _zeros(self):
		"""bool for each blade in the subspace, if it contains a degenerate basis vector"""
		return self.algebra.bit_dot(self.blades, self.algebra.zeros)
	@cache
	def degenerate(self) -> "SubSpace":
		"""Select degenerate subspace"""
		return self.algebra.subspace.from_blades(self.blades[self._zeros() > 0])
	@cache
	def nondegenerate(self) -> "SubSpace":
		"""Select non-degenerate subspace"""
		return self.algebra.subspace.from_blades(self.blades[self._zeros() == 0])
	@cache
	def scalar_degenerate(self) -> "SubSpace":
		"""Degenerate subspace of self, plus scalars"""
		return self.algebra.subspace.scalar().union(self.degenerate())

	# FIXME: better to generate these from getattr i suppose.
	#  could place them in .operator. namespace?
	#  current namespace is getting rather cluttered
	@cache
	def product(self, other) -> "SubSpace":
		return self.algebra.operator.product(self, other).subspace
	# @cache
	# def division(self, other) -> "SubSpace":
	# 	"""do recursive hitzer logic"""
	# 	return self.product(other.inverse())
	# @cache
	# def inverse(self) -> "SubSpace":
	# 	"""do recursive hitzer logic. subspace inverse of self lives in
	# 	note; this is an 'estimate'; the returned subspace may be larger than the actual inverse
	# 	"""
	# 	# return self.algebra.subspace.scalar() / self

	@cache
	def wedge(self, other) -> "SubSpace":
		return self.algebra.operator.wedge(self, other).subspace
	@cache
	def inner(self, other) -> "SubSpace":
		return self.algebra.operator.inner(self, other).subspace
	# def reject(self, other):
	# 	return self.difference()

	@cache
	def squared(self) -> "SubSpace":
		return self.algebra.operator.squared(self).subspace
	@cache
	def cubed(self) -> "SubSpace":
		return self.algebra.operator.cubed(self).subspace
	@cache
	def symmetric_reverse(self) -> "SubSpace":
		return self.algebra.operator.symmetric_reverse_product(self).subspace
	@cache
	def symmetric_involute(self) -> "SubSpace":
		return self.algebra.operator.symmetric_involute_product(self).subspace
	@cache
	def symmetric_conjugate(self) -> "SubSpace":
		return self.algebra.operator.symmetric_conjugate_product(self).subspace
	@cache
	def symmetric_scalar_negation(self) -> "SubSpace":
		return self.algebra.operator.symmetric_scalar_negation_product(self).subspace
	@cache
	def symmetric_pseudoscalar_negation(self) -> "SubSpace":
		return self.algebra.operator.symmetric_pseudoscalar_negation_product(self).subspace

	@cache
	def reverse_product(self, other) -> "SubSpace":
		return self.algebra.operator.reverse_product(self, other).subspace
	@cache
	def involute_product(self, other) -> "SubSpace":
		return self.algebra.operator.involute_product(self, other).subspace
	@cache
	def conjugate_product(self, other) -> "SubSpace":
		return self.algebra.operator.conjugate_product(self, other).subspace
	@cache
	def scalar_negation_product(self, other) -> "SubSpace":
		return self.algebra.operator.scalar_negation_product(self, other).subspace
	@cache
	def pseudoscalar_negation_product(self, other) -> "SubSpace":
		return self.algebra.operator.pseudoscalar_negation_product(self, other).subspace
	@cache
	def anti_reverse_product(self, other) -> "SubSpace":
		return self.algebra.operator.anti_reverse_product(self, other).subspace

	# NOTE: lets not overload operators here; gets confusing
	# FIXME: use npi?
	@cache
	def intersection(self, other: "SubSpace") -> "SubSpace":
		"""Intersection of elements in both subspaces"""
		return self.algebra.subspace.from_blades(
			list(set(self.blades).intersection(set(other.blades))),
		)
	@cache
	def union(self, other: "SubSpace") -> "SubSpace":
		"""Union of elements in both subspaces"""
		return self.algebra.subspace.from_blades(
			list(set(self.blades).union(set(other.blades))),
		)
	@cache
	def difference(self, other: "SubSpace") -> "SubSpace":
		"""Difference of elements in both subspaces"""
		return self.algebra.subspace.from_blades(
			list(set(self.blades) - set(other.blades)),
		)
	@cache
	def is_subspace(self, other: "SubSpace") -> bool:
		"""True if all blades of other are contained in self
		Contains a shortcircuit to optimize the common case of identical subspaces being tested
		"""
		return (self == other) or set(self.blades).issubset(set(other.blades))
	@cache
	def select_subspace(self, other: "SubSpace") -> "SubSpace":
		return other.union(self.restrict_subspace(other))
	@cache
	def restrict_subspace(self, other: "SubSpace") -> "SubSpace":
		return self.intersection(other)

	def __contains__(self, other: "SubSpace") -> bool:
		return other.is_subspace(self)

	def slice_subspace(self, indices) -> "SubSpace":
		"""slice out part of subspace using indices relative to the indices of the subspace"""
		return self.algebra.subspace.from_blades(self.blades[indices])
	def __index__(self, idx):
		return self.slice_subspace(idx)

	def to_relative_indices(self, blades):
		"""convert blades to indices that index the subspace
		self.blades[indices] = blades
		anti-slice, if you will
		"""
		return npi.indices(self.blades, blades)

	@cached_property
	def is_subalgebra(self) -> bool:
		"""Check that this subspace represents a subalgebra under the action of the geometric product"""
		return self.product(self) in self

	@cached_property
	def minimal_subalgebra(self) -> "SubSpace":
		"""Find the smallest closed subalgebra this subspace is a part of
		Useful in finding square roots? not sure; cant find sqrt of -1 in the scalars...
		"""
		subspace, previous = self, None
		while subspace != previous:
			previous = subspace
			subspace = subspace.union(self.product(subspace))
		return subspace
	@cached_property
	def minimal_exponential(self) -> "SubSpace":
		"""Find the smallest closed subalgebra this subspace is a part of under exponentiation
		Can be used to deduce subspace exponential lives in?
		"""
		subspace, previous = self.union(self.algebra.subspace.scalar()), None
		while subspace != previous:
			previous = subspace
			subspace = subspace.union(subspace.squared())
		return subspace

	@cache
	def is_n_simple(self, n):
		"""A subspace is n-simple if it maps to a scalar under n self-involution products"""
		if n == 0:
			return self.inside.scalar()
		return (
				self.is_squared_n_simple(n) or
				self.is_reverse_n_simple(n) or
				self.is_involute_n_simple(n) or
				self.is_conjugate_n_simple(n) or
				self.is_scalar_negation_n_simple(n) or
				self.is_pseudoscalar_negation_n_simple(n)
		)

	@cached_property
	def simplicity(self) -> int:
		"""Return minimum number of products found to reduce this subspace to a scalar"""
		# FIXME: our use of simplicity is an abuse of terminology
		#  simple means can be factored as wedge of one vectors (which means it squares scalar)
		#  not the other way around

		for n in range(self.algebra.n_dimensions // 2 + 2):
			if self.is_n_simple(n):
				return n
		return None

	# a subspace is simple in any of the specific senses below if its simplicity reduces under the given operator
	@cache
	def is_squared_n_simple(self, n) -> bool:
		return self.squared().is_n_simple(n-1)
	@cache
	def is_reverse_n_simple(self, n) -> bool:
		return self.symmetric_reverse().is_n_simple(n-1)
	@cache
	def is_involute_n_simple(self, n) -> bool:
		return self.symmetric_involute().is_n_simple(n-1)
	@cache
	def is_conjugate_n_simple(self, n) -> bool:
		return self.symmetric_conjugate().is_n_simple(n-1)
	@cache
	def is_scalar_negation_n_simple(self, n) -> bool:
		return self.symmetric_scalar_negation().is_n_simple(n-1)
	@cache
	def is_pseudoscalar_negation_n_simple(self, n) -> bool:
		return self.symmetric_pseudoscalar_negation().is_n_simple(n-1)


	@cache
	def symmetric_alt_product(self) -> "SubSpace":
		# of = self.algebra.operator
		# # op = of.compose_symmetry_ops(self, of.reverse, of.conjugate)
		# op = of.compose_symmetry_ops(self, of.reverse, of.scalar_negation)
		# op = of.complete_op(op)
		op = self.algebra.operator.inverse_factor_completed_alt(self)
		return op.subspace
	@cache
	def is_alt_n_simple(self, n) -> bool:
		return self.symmetric_alt_product().is_n_simple(n-2)



	@cached_property
	def is_degenerate_scalar(self) -> bool:
		# rename to nonscalar_degenerate?
		return self == self.algebra.subspace.scalar().union(self.degenerate())


	# NOTE: the below are kinda useless since introducing .equals / .inside syntax
	@cached_property
	def is_degenerate(self) -> bool:
		return self == self.degenerate()
	@cached_property
	def in_even(self) -> bool:
		return self in self.algebra.subspace.even_grade()
	@cached_property
	def in_odd(self) -> bool:
		return self in self.algebra.subspace.odd_grade()
	@cached_property
	def is_even(self) -> bool:
		return self == self.algebra.subspace.even_grade()
	@cached_property
	def is_odd(self) -> bool:
		return self == self.algebra.subspace.odd_grade()
	@cached_property
	def is_empty(self) -> bool:
		return self == self.algebra.subspace.empty()
	@cached_property
	def in_self_reverse(self) -> bool:
		"""check that this subspace is self-reverse"""
		return self in self.algebra.subspace.self_reverse()
	@cached_property
	def is_study(self) -> bool:
		"""Check that this is a generalized study number"""
		return self.is_scalar_negation_n_simple(1)


	# FIXME: this is a bit of a hack. can we make subspace a super-type of our type-hierarchy instead?
	#  better yet is if all subspaces would be wrapped in an operator; including a nullary subspace operator,
	#  in the trivial case
	@property
	def subspace(self):
		"""For unifying syntax in operator construction"""
		return self

	def __getattr__(self, operator) -> "Operator":
		"""Subspace can also act as an operator factory.
		Allowing us to write V.wedge(B) as alternative syntax to
		algebra.operator.wedge(V, B)

		"""
		# FIXME: add is_ in_ cached attributes via getattr mechanism?
		#  or add .is. .in. subnamespaces?
		try:
			op = getattr(self.algebra.operator, operator)
		except AttributeError:
			raise AttributeError(f'No operator named `{operator}` defined in operator factory')
		def bind(*args):
			# FIXME: ternary operators require more specialized binding.
			#  perhaps not require; but thats what we do in multivector
			return op(self, *args)
		return bind

	# def __call__(self, values):
	# 	"""Can we instantiate subspace with values? Not without giving subspace access to context
	# 	what would that look like in dependency injection context?
	# 	"""


