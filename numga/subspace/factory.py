from functools import cached_property
from typing import List

import numpy as np
import numpy_indexed as npi

from numga.algebra.algebra import Algebra
from numga.algebra.bitops import bit_count
from numga.flyweight import FlyweightFactory
from numga.subspace.subspace import SubSpace
from numga.util import cache


class SubSpaceFactory(FlyweightFactory):
	"""Namespace for subspace constructors,

	and a place to provide caching / flyweight mechanism

	"""
	def __init__(self, algebra: Algebra, order='fancy'):
		super(SubSpaceFactory, self).__init__()
		self.algebra = algebra
		self.order = order
		self.grades = np.arange(self.algebra.n_dimensions + 1, dtype=np.uint8)  # enumeration of all grades

	def order_blades(self, blades):
		"""Put an array of blades in canonical order, as defined by this factory"""
		blades = np.array(blades, dtype=self.algebra.blade_dtype)
		if self.order == 'blade':
			blades.sort()
		if self.order == 'grade':
			# first sort on grade; then on blades within that, to provide determinism since blades are unique
			idx = np.lexsort((blades, self.algebra.grade(blades)))
			blades = blades[idx]
		if self.order == 'fancy':
			# FIXME: try to get bivectors and zero-metrics in a specific order; but open questions still
			#  now the trivectors are kinda flipped around in 3dpga
			grade = self.algebra.grade(blades).astype(np.int16)
			# grade = bit_count(np.bitwise_and(blades, np.bitwise_not(self.algebra.zeros)))
			special = bit_count(np.bitwise_and(blades, self.algebra.zeros))
			# order = np.where(
			# 	grade % 2,
			# 	-(blades),  # swap sorting order for higher order blades
			# 	blades,
			# )
			order = np.where(
				(blades * 2 >= self.algebra.n_dimensions),
				# grade * 2 <= self.algebra.dimensions,
				blades,
				blades * -1  # swap sorting order for higher order blades
			)
			order = blades
			# order = grade + (blades * 2 >= self.algebra.dimensions)
			idx = np.lexsort((order, self.algebra.grade(blades)))
			blades = blades[idx]
		return blades

	@cache
	def __getattr__(self, subspace: str) -> SubSpace:
		"""Attribute subspace construction syntax
		Example: 'xy_zt' produces a bivector with these two components
		"""
		names = self.algebra.description.basis_names
		basis = self.basis()
		return self.from_blades(np.array([
			sum(basis[names.index(b)].blades for b in blade)[0]
			for blade in subspace.split('_')
		]))

	@cache
	def basis(self) -> List[SubSpace]:
		# FIXME: generators?
		"""Construct a list of all canonical basis subspaces, as they appear in the 1-vector"""
		return [self.blade(b) for b in self.vector().blades]

	@cached_property
	def blades_by_grade(self) -> List[np.ndarray]:
		# NOTE: this 'greedy' way of precomputing all subspaces doesnt scale to very high dimensions
		blades = np.arange(self.algebra.n_blades)
		grades, blades = npi.group_by(self.algebra.grade(blades), blades)
		assert np.array_equal(grades, self.grades)
		return np.array(blades, dtype=object)

	def from_blades(self, blades: List[int]) -> SubSpace:
		blades = self.order_blades(blades)
		key = blades.tobytes()
		value = lambda: SubSpace(self.algebra, blades)
		return self.factory_construct(key, value)   # construct a subspace and register with the flyweight factory

	def from_grades(self, grades: List[int]) -> SubSpace:
		return self.from_blades(np.concatenate(self.blades_by_grade[grades]))
	def from_bit_str(self, bits: str) -> SubSpace:
		return self.from_blades([int(b[::-1], 2) for b in bits.split(',')])

	@cache
	def k_vector(self, k: int) -> SubSpace:
		return self.from_grades([k])
	@cache
	def blade(self, blade: int) -> SubSpace:
		return self.from_blades([blade])

	# NOTE: capitalize these subspace constructors?
	def scalar(self) -> SubSpace:
		return self.k_vector(0)
	def vector(self) -> SubSpace:
		return self.k_vector(1)
	def bivector(self) -> SubSpace:
		return self.k_vector(2)
	def trivector(self) -> SubSpace:
		return self.k_vector(3)
	def quadvector(self) -> SubSpace:
		return self.k_vector(4)

	def pseudoscalar(self) -> SubSpace:
		return self.k_vector(-1)
	def antiscalar(self) -> SubSpace:
		return self.k_vector(-1)
	def antivector(self) -> SubSpace:
		return self.k_vector(-2)
	def antibivector(self) -> SubSpace:
		return self.k_vector(-3)
	def antitrivector(self) -> SubSpace:
		return self.k_vector(-4)
	def antiquadvector(self) -> SubSpace:
		return self.k_vector(-5)

	@cache
	def multivector(self) -> SubSpace:
		return self.from_grades(...)
	full = multivector
	@cache
	def empty(self) -> SubSpace:
		return self.from_blades([])

	@cache
	def even_grade(self) -> SubSpace:
		return self.from_grades(self.grades[0::2])
	# FIXME: this is abuse of terminology; subspace isnt a motor per se
	motor = even_grade
	self_involute = even_grade
	@cache
	def odd_grade(self) -> SubSpace:
		return self.from_grades(self.grades[1::2])

	@cache
	def k_reflection(self, k: int) -> SubSpace:
		return self.scalar() if k == 0 else self.k_reflection(k-1).product(self.vector())
	def reflection(self):
		return self.k_reflection(1)
	def bireflection(self):
		return self.k_reflection(2)
	def trireflection(self):
		return self.k_reflection(3)
	def quadreflection(self):
		return self.k_reflection(4)

	@cache
	def self_reverse(self) -> SubSpace:
		"""Subspace of multivectors that are self-reverse; that is, x == x.reverse()"""
		return self.from_grades(self.grades[(self.grades // 2 % 2) == 0])
		# return self.from_blades(self.blades[self.algebra.reverse(self.blades) == 1])
	@cache
	def study(self) -> SubSpace:
		"""Study numbers; or grades of [0, 4]"""
		return self.self_reverse().intersection(self.quadreflection())
	@cache
	def mod4(self) -> SubSpace:
		"""Subspace of generalized study numbers;
		output space of motor * motor.reverse()
		even-grade and self-reverse
		"""
		return self.from_grades(self.grades[0::4])


	@cache
	def degenerate(self) -> "SubSpace":
		"""Degenerate subspace"""
		blades = self.full().blades
		zeros = self.algebra.bit_dot(blades, self.algebra.zeros)
		return self.from_blades(blades[zeros > 0])
	@cache
	def scalar_degenerate(self) -> "SubSpace":
		"""Degenerate subspace plus scalars; """
		return self.degenerate().union(self.scalar())
	@cache
	def nondegenerate(self) -> "SubSpace":
		"""Create non-degenerate subspace"""
		return self.full().difference(self.degenerate())
