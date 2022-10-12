from functools import cached_property
from typing import Tuple, Dict

import numpy as np

from numga.backend.numpy.multivector import NumpyMultiVector
from numga.operator.abstract import AbstractConcreteOperator
from numga.operator.operator import Operator
from numga.subspace.subspace import SubSpace


class NumpyOperator(AbstractConcreteOperator):

	@property
	def shape(self):
		"""Shape of the kernel, modulo subspace axis"""
		return self.kernel.shape[:-(self.arity+1)]
	# @property
	# def kernel_shape(self):
	# 	"""Shape of the kernel, modulo subspace axis"""
	# 	return self.kernel.shape[self.arity:]

	def __getitem__(self, idx):
		# index the broadcasting axes; note; cant use slicing syntax for grade/subspace selectin
		# NOTE: this is a candidate for moving into super class? at least common to numpy/jax
		slice = self.copy(self.operator.copy(self.kernel[idx]))
		# assert self.kernel_shape == slice.kernel_shape, 'Dont slice off any kernel axes!'
		return slice

	def concatenate(self, other, axis):
		assert self.operator.axes == other.operator.axes
		kernel = (np.concatenate([self.operator.kernel, other.operator.kernel], axis=axis))
		return self.copy(self.operator.copy(kernel))


	def broadcast_allocate(self, inputs: Tuple[NumpyMultiVector], output=None) -> NumpyMultiVector:
		"""Allocation for a set of inputs, with multivector components as last axis,
		and broadcasting axes on the left"""
		# assert self.operator.inputs == tuple(i.subspace for i in inputs)
		output = output or self.output
		shape = np.broadcast_shapes(self.shape, *(i.shape for i in inputs)) + (len(output), )
		return self.context.multivector(
			values=self.context.allocate_array(shape),
			subspace=output
		)

	@staticmethod
	def broadcast_einsum(a: str) -> str:
		"""Broadcasting to the left in our einsum expressions"""
		return f'...{a}'

	def partial(self, inputs: Dict[int, NumpyMultiVector], expr=None) -> "NumpyEinsumOperator":
		# NOTE: after partial application we tend to be dealing with dense kernels, so einsum operator is likely best fit
		expr = expr or self.precompute_einsum_partial(tuple(inputs.keys()))
		# FIXME: would want to use same form of preallocation here too;
		#  hinges on generalization of our operator/mv unification i suppose
		#  prealloc wouldnt work with custom kernel I suppose
		kernel = np.einsum(expr, self.kernel, *(i.values for i in inputs.values()), optimize=True)
		return NumpyEinsumOperator(
			self.context,
			# FIXME: is it ok to use Operator class in this manner, or should we keep it reserved to non-broadcasting kernels?
			Operator(
				kernel,
				(a for i, a in enumerate(self.operator.axes) if i not in inputs)
			)
		)
	def sum(self, axis):
		"""Sum over a broadcastable axis"""
		# FIXME: check that it is broadcastable? len(axes) dimensions left over on the right?
		return NumpyEinsumOperator(
			self.context,
			self.operator.copy(self.context.coerce_array(self.kernel.sum(axis)))
		)
	def inverse(self):
		# only unary operators are invertible in the LA-sense
		# to be invertable the matrix should also be square and full rank;
		# but well let np.linalg.inv be the judge of that
		assert self.arity == 1
		# inverse operators almost always dense operators?
		return NumpyEinsumOperator(
			self.context,
			Operator(
				kernel=self.context.coerce_array(np.linalg.inv(self.kernel)),
				axes=(self.operator.axes[1], self.operator.axes[0]),
			)
		)
	def transpose(self):
		assert self.arity == 1
		return type(self)(
			self.context,
			Operator(
				kernel=np.transpose(self.kernel, (-1, -2)),
				axes=(self.operator.axes[1], self.operator.axes[0]),
			)
		)
	# def quadratic(self, l, r=None):
	# 	"""Compute quadratic form of unary operator; l*O*r"""
	# 	assert self.arity == 1
	# 	r = l if r is None else r
	# 	output = self.broadcast_allocate((l, r), self.algebra.subspace.scalar())
	# 	np.einsum(
	# 		'...ij,...i,...j->...',
	# 		self.kernel, l.values, r.values,
	# 		out=output.values[..., 0],
	# 		optimize=True
	# 	)
	# 	return output
	# def __mul__(self, other):
	# 	# FIXME:
	# 	operator = self.operator.product(other.subspace)
	# 	operator.bind()


class NumpyEinsumOperator(NumpyOperator):
	"""Implement multilinear operator using einsum on dense kernel in numpy

	This is the best implementation in numpy for small subspaces or unbatched multivectors
	"""

	@cached_property
	def precompute(self) -> str:
		q = self.precompute_einsum_partial((tuple(range(self.arity))))
		# FIXME: only want to see this called initially, really
		# #
		# print(q)
		# print(self.operator.axes)
		return q

	def __call__(self, *inputs: Tuple[NumpyMultiVector]) -> NumpyMultiVector:
		if any(isinstance(inp, SubSpace) for inp in inputs):
			# FIXME: this is hacky; need this for all operator types,
			#  and better adressed with unified MVOperator class...
			return self.partial({i: inp for i, inp in enumerate(inputs) if not isinstance(inp, SubSpace)})
		output = self.broadcast_allocate(inputs)
		return self.context.multivector(
			values=np.einsum(
				self.precompute,
				self.kernel,
				*(i.values for i in inputs),
				out=output.values,
				optimize=True
			),
			subspace=self.output,
		)


class NumpySparseOperator(NumpyOperator):
	"""Unroll a (generally sparse) kernel into computations on individual nonzero entries

	For small batches or small subspaces, this is unlikely to be the most efficient option;
	Though for large batches and large subspaces, it is likely the opposite.
	Good thing you have a framework for easily experimenting with whats best in your situation!
	"""

	def __call__(self, *inputs: Tuple[NumpyMultiVector]) -> NumpyMultiVector:
		""""""
		expr = self.operator.precompute_einsum_prod
		output = self.broadcast_allocate(inputs)

		for oi, term in self.precompute_sparse:
			# sadly, we cannot pass the preallocated slice repeatedly to accumulate inputs; einsum zeros output operand
			output.values[..., oi] = sum(
				# use einsum to loop-fuse individual product terms
				# for unary operators there should be no gain,
				# but for binary operations it saves us the scalar term,
				# and for n-ary operations it should be a clear gain,
				# but I have not yet benchmarked this; einsum overhead might be substantial?
				np.einsum(expr, scalar, *(inp.values[..., ii] for inp, ii in zip(inputs, idx)), optimize=True)
				# math.prod((inp[ii, ...] for inp, ii in zip(inputs, idx)), start=scalar)
				for (idx, scalar) in term
			)
		return output
