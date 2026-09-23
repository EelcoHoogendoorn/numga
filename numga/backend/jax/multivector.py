import jax
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
		values = context.coerce_array(values)
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

	def inverse_la(self):
		"""Inverse of x such that x * x.inverse() == 1 == x.inverse() * x"""
		inverse_subspace = self.subspace.inverse_subspace_estimate()
		op = self.operator.product(self.subspace, inverse_subspace)
		k = op.partial({0: self}).kernel
		k = jnp.swapaxes(k, -1, -2)
		unit = op.output.blades == 0

		# r = jnp.einsum('...k,k', k, unit)
		try:
			r = jnp.linalg.solve(k, unit)	# some 10x faster in compiled jax
		except:
			def solve_one_system(a):
				return jnp.linalg.lstsq(a, unit)[0]
			r = jax.vmap(solve_one_system)(k)

		# idx, = jnp.flatnonzero(unit)    # grab index of scalar of output; zero or raises
		# r = jnp.linalg.solve(    # use least squares to solve for inverse
		# 	jnp.einsum('...ji,...ki->...jk', k, k), # k.T * k
		# 	k[..., idx],    #equal to  k.T * unit_scalar
		# )
		return self.context.multivector(values=r, subspace=inverse_subspace)

	def solve(self, rhs):
		"""Solve for y such that self * y = rhs"""
		op = self.algebra.operator.solve(self.subspace, rhs.subspace)
		k = self.operator(op).partial({0: self}).kernel
		rhs = rhs.select_subspace(op.subspace).values

		r = jnp.linalg.solve(    # use least squares
			jnp.einsum('...ji,...ki->...jk', k, k), # k.T * k
			jnp.einsum('...ji, ...i->...j', k, rhs),
		)
		return self.context.multivector(values=r, subspace=op.axes[1])
