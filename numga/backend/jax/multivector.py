from jax import numpy as jnp

from numga.backend.jax import pytree
from numga.multivector.multivector import AbstractMultiVector
from numga.subspace.subspace import SubSpace


@pytree.register
class JaxMultiVector(AbstractMultiVector):
	"""Concrete realization of abstract type in JAX"""
	# override these fields as static
	subspace: SubSpace
	context: "JaxContext"

	__pytree_ignore__ = ('subspace', 'context')
	# dense contiguous backing array
	values: jnp.array

	@classmethod
	def construct(cls, context, values, subspace):
		if values is None:
			# scalar = 1, rest = 0 is the default init,
			# which is a sensible unit default in a multiplicative algebra
			values = subspace.blades == 0
		# coerce type for external given values
		values = jnp.asarray(values, dtype=context.dtype)
		# first axis should enumerate the subspace;
		# not worth it to complicate downstream code with deviations from that pattern
		# assert len(values) == len(subspace)
		assert values.shape[-1] == len(subspace)
		return cls(
			context=context,
			values=values,
			subspace=subspace
		)

	def __getitem__(self, idx):
		v = self.values[idx]
		return self.copy(values=v)
	@property
	def shape(self):
		"""Shape of the array, modulo subspace axis"""
		return self.values.shape[:-1]

	def take_along_axis(self, idx, axis):
		values = self.values
		values = jnp.take_along_axis(values, idx[..., None], axis=axis)
		return self.copy(values=values)

	def concatenate(self, other, axis):
		assert self.subspace == other.subspace
		return self.copy(jnp.concatenate([self.values, other.values], axis=axis))

	# FIXME: in the below, assert that subspace axes remain untouched?
	def sum(self, axis):
		return self.copy(values=self.values.sum(axis))
	def mean(self, axis):
		return self.copy(values=self.values.mean(axis))

	def rearrange(self, pattern):
		from einops import rearrange
		return self.copy(rearrange(self.values, pattern))
	def repeat(self, pattern, **kwargs):
		from einops import repeat
		return self.copy(repeat(self.values, pattern, **kwargs))
