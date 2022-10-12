from typing import List

from numga.context import AbstractContext
from numga.multivector.multivector import AbstractMultiVector
from numga.util import cache


class MultiVectorFactory:
	"""Groups together some convenience functionality to allocate concrete MultiVector types
	"""
	def __init__(self, context: AbstractContext):
		self.context = context

	@property
	def algebra(self):
		return self.context.algebra
	@property
	def mvtype(self):
		return self.context.mvtype
	@property
	def dtype(self):
		return self.context.dtype

	def __call__(self, subspace=None, values=None):
		return self.multivector(values, subspace)

	@cache
	def __getattr__(self, blade: str) -> "MultiVector":
		"""Construct a basis blade according to the given blade string"""
		basis = self.basis()
		import operator
		from functools import reduce
		return reduce(
			operator.xor,       # wedge together the basis 1-blades
			(basis[i] for i in self.algebra.description.parse_tokens(blade))
		)

	def basis(self) -> List[AbstractMultiVector]:
		"""Construct a list of canonical basis 1-blades"""
		return [
			self.multivector(values=[1], subspace=b)
			for b in self.algebra.subspace.basis()
		]

	def blade(self, blade: int, s: int = 1):
		return self.mvtype.construct(
			values=[s],
			subspace=self.algebra.subspace.blade(blade),
			context=self.context
		)

	def empty(self):
		return self.multivector(subspace=self.algebra.subspace.empty())
	def scalar(self, values=None):
		return self.k_vector(0, values)
	def vector(self, values=None):
		return self.k_vector(1, values)
	def bivector(self, values=None):
		return self.k_vector(2, values)
	def trivector(self, values=None):
		return self.k_vector(3, values)
	def pseudoscalar(self, values=None):
		return self.k_vector(-1, values)
	def antivector(self, values=None):
		return self.k_vector(-2, values)
	def antibivector(self, values=None):
		return self.k_vector(-3, values)
	def antitrivector(self, values=None):
		return self.k_vector(-4, values)

	def k_vector(self, k: int, values=None):
		return self.multivector(values, self.algebra.subspace.k_vector(k))

	def rotor(self, values=None):
		return self.multivector(values, self.algebra.subspace.even_grade())
	def motor(self, values=None):
		return self.multivector(values, self.algebra.subspace.even_grade())

	def multivector(self, values=None, subspace=None):
		if subspace is None:
			# full algebra default
			subspace = self.algebra.subspace.full()
		return self.mvtype.construct(
			values=values,
			subspace=subspace,
			context=self.context,
		)
