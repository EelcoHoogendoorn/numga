"""JAX based operators

We implement both dense and sparsely executed operators.
As a simple rule of thumb, dense operators compile faster since they have smaller compute graphs,
but sparse operators require fewer FLOPs to execute, and therefore tend to run faster.
The details are application and device dependent,
but in any case this gives us an interesting debug/release compile alternatives,
since JAX compilation for large programs can be quite the development bottleneck.

While there are probably minor numerical rounding differences depending on the operator execution strategy,
otherwise the results should obviously be the same.


Note that it would be cool to run a sparsified variant of alphatensor,
to find optimal ways of executing our large sparse products.
"""

from functools import partial
import math
from typing import Tuple, Dict

import jax
import jax.numpy as jnp

from numga.multivector.helper import IndexHelper
from numga.backend.jax.multivector import JaxMultiVector
from numga.operator.operator import Operator
from numga.operator.abstract import AbstractConcreteOperator
from numga.backend.jax import pytree


class JaxOperator(AbstractConcreteOperator):

	@property
	def shape(self):
		"""Shape of the kernel, modulo subspace axis"""
		return self.kernel.shape[:-(self.arity+1)]
	# @property
	# def kernel_shape(self):
	# 	"""Shape of the kernel, modulo subspace axis"""
	# 	return self.kernel.shape[self.arity:]

	def broadcast_allocate(self, inputs: Tuple[JaxMultiVector], output=None) -> JaxMultiVector:
		"""Allocation for a set of inputs, with multivector components as last axis,
		and broadcasting axes on the left"""
		# assert self.operator.inputs == tuple(i.subspace for i in inputs)
		output = output or self.output
		shape = jnp.broadcast_shapes(self.shape, *(i.shape for i in inputs)) + (len(output), )
		return self.context.multivector(
			values=self.context.allocate_array(shape),
			subspace=output
		)

	@staticmethod
	def broadcast_einsum(a: str) -> str:
		"""Broadcasting to the left in our einsum expressions"""
		return f'...{a}'


	def __getitem__(self, idx):
		k = self.kernel[idx]
		slice = self.copy(self.operator.copy(k))
		return slice
	@property
	def at(self):
		return IndexHelper(lambda k: self.copy(self.operator.copy(k)), self.kernel, self.context)

	def concatenate(self, other, axis):
		assert self.operator.axes == other.operator.axes
		kernel = (jnp.concatenate([self.operator.kernel, other.operator.kernel], axis=axis))
		return self.copy(self.operator.copy(kernel))


	def partial(self, inputs: Dict[int, JaxMultiVector]) -> "JaxEinsumOperator":
		# NOTE: after partial application we tend to be dealing with dense kernels,
		# so einsum operator is likely best fit as output. but perhaps not always the case?
		# infact inertia tensors and their inverses often have substantial numerical sparsity
		#
		# FIXME: we should specialize this for sparse operators though; do sparse partial, producing einsum op
		expr = self.precompute_einsum_partial(tuple(inputs.keys()))
		return JaxEinsumOperator(
			self.context,
			# NOTE: we wrap the new kernel here in symbolic operator type,
			# since thats what the interface expects... but feels kinda abusive?
			Operator(
				jnp.einsum(expr, self.kernel, *(i.values for i in inputs.values())),
				(a for i, a in enumerate(self.operator.axes) if i not in inputs)
			)
		)

	def sum(self, axis):
		"""Sum over a broadcastable axis"""
		# FIXME: check that it is broadcastable? len(axes) dimensions left over on the right?
		# FIXME: make negative axes work as expected?
		return JaxEinsumOperator(
			self.context,
			self.operator.copy(self.context.coerce_array(self.kernel.sum(axis)))
		)
	def inverse(self):
		# only unary operators are invertible in the LA-sense
		# to be invertable the matrix should also be square and full rank;
		# but well let linalg.inv be the judge of that
		assert self.arity == 1
		# inverse operators almost always dense operators?
		return JaxEinsumOperator(
			self.context,
			Operator(
				kernel=self.context.coerce_array(jnp.linalg.inv(self.kernel)),
				axes=(self.operator.axes[1], self.operator.axes[0]),
			)
		)


@pytree.register
class JaxDenseOperator(JaxOperator):
	"""Implement product using plain broadcasting in JAX.
	This is identical to the approach used for quaternion multiplication in alphafold.
	The JAX compiler seems to handle this well when jitted,
	and it compiles much faster than the unrolled version,
	Though on cpu at least einsum does tend to bring runtime-performance benefits.
	Possibly this implementation is advantageous on TPU?
	"""
	__pytree_ignore__ = ('context', 'operator')

	def __init__(self, *args, **kwargs):
		super(JaxDenseOperator, self).__init__(*args, **kwargs)
		# put kernel on device
		# self.jax_kernel = np.array(self.kernel)
		# precompute reshape operations
		self.shapes = self.broadcasting_shapes

	@partial(jax.jit, static_argnums=(0,))
	def __call__(self, *inputs: Tuple[JaxMultiVector]) -> JaxMultiVector:
		# kernel, shapes = self.precompute
		shape = jnp.broadcast_shapes(*(i.shape for i in inputs))
		return self.context.multivector(
			values=jnp.sum(
				math.prod((i.values.reshape(i.shape + s) for i, s in zip(inputs, self.shapes)), start=self.kernel),
				axis=range(len(shape), len(shape)+self.arity)# FIXME: negative indexing rather than len(shape)?
			),
			subspace=self.output
		)


@pytree.register
class JaxEinsumOperator(JaxOperator):
	"""Implement product using JAX einsum.
	This seems to provide a nice balance between compilation speed and runtime speed,
	though sparse evaluation seems faster in most circumstances on cpu.
	"""
	__pytree_ignore__ = ('context', 'operator')

	def __init__(self, *args, **kwargs):
		super(JaxEinsumOperator, self).__init__(*args, **kwargs)
		# put kernel on device
		# self.jax_kernel = np.array(self.kernel)
		# precompute reshape operations
		# self.expr = self.precompute_einsum_partial(range(self.arity))

	# @property
	# def jax_kernel(self):
	# 	# put kernel on device
	# 	return jnp.array(self.kernel)
	# FIXME: add to pytree ignore?
	@property
	def expr(self):
		return self.precompute_einsum_partial(range(self.arity))

	@partial(jax.jit, static_argnums=(0,))
	def __call__(self, *inputs: Tuple[JaxMultiVector]) -> JaxMultiVector:
		return self.context.multivector(
			subspace=self.output,
			values=jnp.einsum(
				self.expr,
				self.kernel,
				*(i.values for i in inputs),
				optimize=True
			)
		)


@pytree.register
class JaxSparseOperator(JaxOperator):
	"""Unroll a (generally sparse) kernel into computations on individual nonzero entries

	This will give very good performance in jax in high dimension,
	due to its ability to perform subsequent loop fusion.
	However, compilation times are typically an order of magnitude longer,
	Than for dense jax operator execution.

	For debugging/development, dense operator execution is likely preferable
	"""
	__pytree_ignore__ = ('context', 'operator')

	# @partial(jax.jit, static_argnums=(0,))
	def __call__(self, *inputs: Tuple[JaxMultiVector]) -> JaxMultiVector:
		output = self.broadcast_allocate(inputs)
		for oi, term in self.precompute_sparse:
			q = sum(
				math.prod((inp.values[..., ii] for inp, ii in zip(inputs, idx)), start=scalar)
				for (idx, scalar) in term
			)
			output.values = output.values.at[..., oi].set(q)
		return output
