
import numpy as np

from numga.algebra.description import AlgebraDescription
from numga.algebra.bitops import minimal_uint_type, bit_pack, bit_count, parity_to_sign
from typing import Dict
from numga.util import cached_property


class Algebra:
	"""Generate algebra by associating bit-patterns of integers with blade patterns

	This implementation uses vectorized numpy code,
	and implements the combinatorial logic using bitwise operations,
	thus is likely one of the more performant implementations possible in python.
	Though id be happy to be proven wrong on that.
	"""

	def __init__(self, signature: Dict[str, int]):
		if isinstance(signature, AlgebraDescription):
			self.description = signature
		else:
			self.description = AlgebraDescription.from_str(signature)

		self.blade_nbytes, self.blade_dtype = minimal_uint_type(self.n_dimensions)

	@staticmethod
	def from_str(str):
		return Algebra(AlgebraDescription.from_str(str))

	@staticmethod
	def from_pqr(p, q, r):
		return Algebra(AlgebraDescription.from_pqr(p, q, r))

	@property
	def n_dimensions(self):
		return self.description.n_dimensions
	@property
	def n_grades(self):
		return self.description.n_grades
	@property
	def n_blades(self):
		return self.description.n_blades
	def __len__(self):
		return self.description.n_blades

	def __mul__(self, other):
		return Algebra(self.description * other.description)

	@cached_property
	def negatives(self) -> np.ndarray:
		return bit_pack(self.description.negatives).astype(self.blade_dtype)
	@cached_property
	def positives(self) -> np.ndarray:
		return bit_pack(self.description.positives).astype(self.blade_dtype)
	@cached_property
	def zeros(self) -> np.ndarray:
		return bit_pack(self.description.zeros).astype(self.blade_dtype)

	@cached_property
	def pseudo_scalar_squared(self) -> int:
		"""Square of the pseudoscalar of this algebra"""
		pss = self.subspace.pseudoscalar()
		_, signs = self.product(pss.blades, pss.blades)
		return int(signs[0, 0])

	def complement(self, a):
		"""Complement of a set of blades"""
		pss = self.subspace.pseudoscalar()
		return np.bitwise_xor(a, pss.blades)

	def bit_dot(self, a, b):
		"""Dot-product between bit-blades; counting their shared high bits"""
		return bit_count(np.bitwise_and(a, b), nbytes=self.blade_nbytes)

	def grade(self, a):
		"""Grade of each blade; number of basis vectors present in each blade

		Parameters
		----------
		a: np.ndarray, [...], blade_dtype
			basis blades encoded as integer bitpatterns

		Returns
		-------
		np.ndarray, [...], uint8
			grade of each blade in a
		"""
		return bit_count(a, nbytes=self.blade_nbytes)

	def cayley(self, a, b):
		"""Cayley table of the geometric product a * b

		Parameters
		----------
		a: np.ndarray, [...], blade_dtype
			basis blades encoded as integer bitpatterns
		b: np.ndarray, [...], blade_dtype
			basis blades encoded as integer bitpatterns

		Returns
		-------
		cayley: np.ndarray, [...], blade_dtype
			bitpatterns of the blades that the product maps to
		swaps: np.ndarray, [...], np.int8
			minimal number of pairwise swaps required to reorder the product of basis vectors,
			back to a sorted canonical form

		"""
		# these are the basis vectors that remain; easy peasy
		cayley = np.bitwise_xor(a, b)

		# Count the swaps required to normalize the order of the blades in the product
		def downshift(e):
			for i in range(self.n_dimensions - 1):
				e = np.left_shift(e, 1)
				yield e
		# bit in second arg needs to step over all bits present in upper tri to fulfill all swaps
		# we are fully vectorized over n_elements, which is what matters most,
		# but we vectorize over one of the bit axes too in our registers
		swaps = sum(self.bit_dot(a, shifted_b) for shifted_b in downshift(b))
		return cayley, swaps

	def involute(self, a):
		"""Tack a minus onto every basis vector

		Parameters
		----------
		a: np.ndarray, [n], blade_dtype
			basis blades encoded as integer bit-patterns

		Returns
		-------
		signs: np.ndarray, [n], np.int8
			signs of the involute operation
		"""
		return parity_to_sign(self.grade(a))

	def reverse(self, a):
		"""Calculate the equivalent sign flip incurred by reversing the order of all basis vectors,
		and permuting them back to their canonical sorted order.
		This requires grade // 2 swaps

		Parameters
		----------
		a: np.ndarray, [n], blade_dtype
			basis blades encoded as integer bitpatterns

		Returns
		-------
		signs: np.ndarray, [n], np.int8
			signs of the reverse operation
		"""
		return parity_to_sign(self.grade(a) // 2)

	def product(self, a, b):
		"""Construct geometric product between elements of a and b

		Parameters
		----------
		a: np.ndarray, [a], blade_dtype
			basis blades encoded as integer bitpatterns
		b: np.ndarray, [b], blade_dtype
			basis blades encoded as integer bitpatterns

		Returns
		-------
		cayley: np.ndarray, [a, b], blade_dtype
			bitpatterns of the blades that the product maps to
		signs: np.ndarray, [a, b], np.int8
			signs of the product
		"""

		# these are the basis vectors that occur on both sides of the product,
		# and will be eliminated through the metric
		doubles = np.bitwise_and.outer(a, b)
		# how often do we hit negative metric?
		negatives = self.bit_dot(doubles, self.negatives)
		# do we hit any degenerate parts of the metric?
		zeros = self.bit_dot(doubles, self.zeros)

		cayley, swaps = self.cayley(a[:, None], b[None, :])
		return cayley, parity_to_sign(negatives + swaps) * (zeros == 0)


	# high level interface; should these be part of context object instead?
	# feels kinda out of place here; otoh kinda nice to be able to do algebbraic symbolic manipulations,
	# without needing all the crap that comes with a backend specfic context.
	@cached_property
	def subspace(self) -> "SubSpaceFactory":
		"""Access the subspace factory

		Subspaces are used in various places as keys to caches; hence to realize those caching benefits,
		it is important that all subspaces are created via this cached property
		"""
		from numga.subspace.factory import SubSpaceFactory
		return SubSpaceFactory(self)

	@cached_property
	def operator(self) -> "OperatorFactory":
		"""Nice place to nest and cache all our operator related stuff"""
		from numga.operator.factory import OperatorFactory
		return OperatorFactory(self)
