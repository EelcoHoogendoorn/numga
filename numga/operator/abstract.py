import itertools
from functools import cached_property
from typing import Tuple, Dict

import numpy as np

from numga.multivector.multivector import AbstractMultiVector
from numga.operator.operator import Operator
from numga.util import cache


class AbstractConcreteOperator:
	"""Interface to wrap an abstract linear operator,
	with a concrete implementation in a specific framework

	Abstract in the sense that this should not be instantiated
	concrete in the sense that its an interface for a specific framework :)
	"""
	# FIXME: does concrete operator need to support a notion of broadcasting?
	#  if we want to generalize the notion of partial apply, i think so
	def __init__(self, context: "AbstractContext", operator: Operator):
		# need access to the context to allocate output multivectors
		self.context = context
		self.operator = operator

	def copy(self, operator):
		# FIXME: copying these will drop cached props; this is mostly undesirable
		#  do we need some kind of operatorinfo object independent of both backing storage/slicing thereof,
		#  as well as independent from the abstract algebraic operator?
		#  could push them into abstract operator, but some of these things are backend specific,
		#  like precomputed einsum strings and the like
		#  we encounter this problem when slicing inertia tensions from bodies
		#  in constraint solver. arguably we shouldnt do this anyway;
		#  they dont change over iterations so only need to dynamicay slice dynamic state.
		#  but thats a seperate issue, really
		return type(self)(self.context, operator)

	@property
	def algebra(self):
		return self.operator.algebra
	@property
	def kernel(self):
		return self.operator.kernel
	@property
	def output(self):
		return self.operator.output
	@property
	def inputs(self):
		return self.operator.inputs
	@property
	def arity(self):
		return self.operator.arity

	def __call__(self, *inputs: Tuple[AbstractMultiVector]) -> AbstractMultiVector:
		"""Bind all inputs"""
		raise NotImplementedError

	def partial(self, inputs: Dict[int, AbstractMultiVector]) -> "AbstractConcreteOperator":
		"""Bind a subset of inputs"""
		raise NotImplementedError

	# some generic utilities that tend to be useful in derived concrete types
	@cached_property
	def broadcasting_shapes(self):
		"""Precompute shape tuples for proper broadcasting with the kernel,
		if using a direct dense product approach

		Returns
		-------
		Tuple[Tuple[int, ...], ...]
		"""
		def shape(i):
			s = [1] * self.kernel.ndim
			s[i] = self.kernel.shape[i]
			return tuple(s)
		return tuple(shape(i) for i in range(self.arity))

	@cache
	def precompute_einsum_partial(self, axes: Tuple[int, ...]) -> str:
		"""Build einsum expression for (partial) application of input arguments

		Parameters
		----------
		axes : Tuple[int, ...]
			sequence of integers denoting which axes are meant to be included in the product

		Returns
		-------
		einsum_expr: str
			numpy-valid einsum expression
		"""
		# kernel characters
		from numga.util import chars
		k = chars[:self.arity + 1]
		# input axis characters
		i = tuple(self.broadcast_einsum(chars[a]) for a in axes)
		# output axes characters
		o = (c for c in k if c not in i)
		i = ','.join(i)
		o = self.broadcast_einsum(''.join(o))
		expr = f'{self.broadcast_einsum(k)},{i}->{o}'
		return expr

	@staticmethod
	def broadcast_einsum(a: str) -> str:
		"""Overridable behavior for specifying broadcasting in einsum expression"""
		return a

	@cached_property
	def intensity(self) -> int:
		"""Number of operations"""
		return np.count_nonzero(self.kernel)
	@cached_property
	def sparsity(self) -> float:
		"""Fraction in the range [0-1]"""
		return self.intensity / self.kernel.size

	# FIXME: make partial variant?
	@cached_property
	def precompute_sparse(self):
		"""A nice quadruply nested tuple, encoding our operations

		Returns
		-------
		For each row of the output subspace,
		return a tuple containing all individual product terms contributing to that output,
		where each product term is encoded as tuple (axes, scalar),
		where axes is a tuple of integers of length equal to the number of inputs,
		encoding which part of the subspace of each input participates in this product term
		"""
		# FIXME: should generalize this to also handle sparse partial application!
		# FIXME: what about broadcasting axes in the kernel; like inertia tensor?
		#  zero pattern should be check as reduced over all broadcasting axes
		#  and should return array rather than scalar
		kernel = self.kernel
		*si, so = kernel.shape
		rows = tuple(
			tuple(
				(tuple(ii), kernel[ii + (io,)])
				for ii in itertools.product(*[range(i) for i in si])
				# FIXME: np.nonzero and some vectorized approach might be superior
				if kernel[ii + (io,)]
			)
			for io in range(so)
		)
		# enumerate rows, and drop rows that are not written to
		return tuple(
			(i, row)
			for i, row in enumerate(rows)
			if len(row)
		)

	def squeeze(self):
		return self.copy(self.operator.squeeze())
