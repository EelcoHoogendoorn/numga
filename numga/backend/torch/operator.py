
import math
from typing import Tuple, Dict

import torch

from numga.multivector.helper import IndexHelper
from numga.backend.torch.multivector import TorchMultiVector
from numga.operator.operator import Operator
from numga.operator.abstract import AbstractConcreteOperator


class TorchOperator(AbstractConcreteOperator):

	@property
	def shape(self):
		"""Shape of the kernel, modulo subspace axis"""
		return tuple(self.kernel.shape)[:-(self.arity+1)]
	# @property
	# def kernel_shape(self):
	# 	"""Shape of the kernel, modulo subspace axis"""
	# 	return self.kernel.shape[self.arity:]

	def broadcast_allocate(self, inputs: Tuple[TorchMultiVector], output=None) -> TorchMultiVector:
		"""Allocation for a set of inputs, with multivector components as last axis,
		and broadcasting axes on the left"""
		# assert self.operator.inputs == tuple(i.subspace for i in inputs)
		output = output or self.output
		shape = torch.broadcast_shapes(self.shape, *(i.shape for i in inputs)) + (len(output), )
		return self.context.multivector(
			values=self.context.allocate_array(shape),
			subspace=output
		)

	@staticmethod
	def broadcast_einsum(a: str) -> str:
		"""Broadcasting to the left in our einsum expressions"""
		return f'...{a}'


	def __getitem__(self, idx):
		try:
			idx = torch.asarray(idx)
		except:
			pass
		k = self.kernel[idx]
		slice = self.copy(self.operator.copy(k))
		return slice
	@property
	def at(self):
		return IndexHelper(lambda k: self.copy(self.operator.copy(k)), self.kernel, self.context)

	def concatenate(self, other, axis):
		assert self.operator.axes == other.operator.axes
		kernel = (torch.cat([self.operator.kernel, other.operator.kernel], dim=axis))
		return self.copy(self.operator.copy(kernel))


	def partial(self, inputs: Dict[int, TorchMultiVector]) -> "TorchEinsumOperator":
		# NOTE: after partial application we tend to be dealing with dense kernels,
		# so einsum operator is likely best fit as output. but perhaps not always the case?
		# infact inertia tensors and their inverses often have substantial numerical sparsity
		#
		# FIXME: we should specialize this for sparse operators though; do sparse partial, producing einsum op
		expr = self.precompute_einsum_partial(tuple(inputs.keys()))
		# FIXME: cache putting on device
		kernel = self.context.coerce_array(self.kernel)

		return TorchEinsumOperator(
			self.context,
			# NOTE: we wrap the new kernel here in symbolic operator type,
			# since thats what the interface expects... but feels kinda abusive?
			Operator(
				torch.einsum(expr, kernel, *(i.values for i in inputs.values())),
				(a for i, a in enumerate(self.operator.axes) if i not in inputs)
			)
		)

	def sum(self, axis):
		"""Sum over a broadcastable axis"""
		# FIXME: check that it is broadcastable? len(axes) dimensions left over on the right?
		# FIXME: make negative axes work as expected?
		return TorchEinsumOperator(
			self.context,
			self.operator.copy(self.context.coerce_array(self.kernel.sum(dim=axis)))
		)
	def inverse(self):
		# only unary operators are invertible in the LA-sense
		# to be invertable the matrix should also be square and full rank;
		# but well let linalg.inv be the judge of that
		assert self.arity == 1
		# inverse operators almost always dense operators?
		return TorchEinsumOperator(
			self.context,
			Operator(
				kernel=self.context.coerce_array(torch.linalg.inv(self.kernel)),
				axes=(self.operator.axes[1], self.operator.axes[0]),
			)
		)


class TorchDenseOperator(TorchOperator):
	"""Implement product using plain broadcasting."""

	def __init__(self, *args, **kwargs):
		super(TorchDenseOperator, self).__init__(*args, **kwargs)
		# precompute reshape operations
		self.shapes = self.broadcasting_shapes

	def __call__(self, *inputs: Tuple[TorchMultiVector]) -> TorchMultiVector:
		# kernel, shapes = self.precompute
		# FIXME: cache putting on device
		kernel = self.context.coerce_array(self.kernel)
		return self.context.multivector(
			values=torch.sum(
				math.prod((i.values.reshape(s) for i, s in zip(inputs, self.shapes)), start=kernel),
				dim=range(self.arity)
			),
			subspace=self.output
		)


class TorchEinsumOperator(TorchOperator):
	"""Implement product using einsum."""

	def __init__(self, *args, **kwargs):
		super(TorchEinsumOperator, self).__init__(*args, **kwargs)

	@property
	def expr(self):
		return self.precompute_einsum_partial(range(self.arity))

	def __call__(self, *inputs: Tuple[TorchMultiVector]) -> TorchMultiVector:
		# NOTE: it is assumed here broadcasting is left purely to vmap
		# FIXME: cache putting on device
		kernel = self.context.coerce_array(self.kernel)
		return self.context.multivector(
			subspace=self.output,
			values=torch.einsum(
				self.expr,
				kernel,
				*(i.values for i in inputs),
			)
		)


class TorchSparseOperator(TorchOperator):
	"""Unroll a (generally sparse) kernel into computations on individual nonzero entries"""

	def __call__(self, *inputs: Tuple[TorchMultiVector]) -> TorchMultiVector:
		output = self.broadcast_allocate(inputs)
		for oi, term in self.precompute_sparse:
			q = sum(
				math.prod((inp.values[..., ii] for inp, ii in zip(inputs, idx)), start=scalar)
				for (idx, scalar) in term
			)
			output.values[..., oi] = q
		return output
