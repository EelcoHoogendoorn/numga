import numpy as np

from numga.multivector.multivector import AbstractMultiVector


class NumpyMultiVector(AbstractMultiVector):
	"""Concrete realization of abstract type in numpy"""
	# dense contiguous backing array
	values: np.array

	@classmethod
	def construct(cls, context: "NumpyContext", values: np.ndarray, subspace: "SubSpace"):
		if values is None:
			# scalar = 1, rest = 0 is the default init,
			# which is a sensible unit default in a multiplicative algebra
			values = subspace.blades == 0
		# coerce type for external given values
		values = context.coerce_array(values)

		# last axis should enumerate the subspace
		# earlier axes are used for broadcasting, as is the most ideomatic in numpy
		# memory layout might impact performance, but is controlled seperately.
		assert values.shape[-1] == len(subspace)

		return cls(
			context=context,
			values=values,
			subspace=subspace
		)
	def __getitem__(self, idx):
		return self.copy(values=self.values[idx])
	# def __setitem__(self, idx, value):
	# 	# FIXME: should setitem return a copy?
	# 	v = self.values#.copy()
	# 	v[idx] = value.values
	# 	self.values = v
	def concatenate(self, other, axis):
		assert self.subspace == other.subspace
		return self.copy(np.concatenate([self.values, other.values], axis=axis))

	@property
	def shape(self):
		"""Shape of the array, modulo subspace axis"""
		return self.values.shape[:-1]

	def take_along_axis(self, idx, axis):
		values = self.values
		values = np.take_along_axis(values, idx[..., None], axis=axis)
		return self.copy(values=values)

	def sum(self, axis):
		return self.copy(values=self.values.sum(axis))
	def mean(self, axis):
		return self.copy(values=self.values.mean(axis))


	# FIXME: should we support general einops on base multivector?
	#  appealing; sure want ot unify these jnp type ops between numpy and JAX
	def rearrange(self, pattern):
		from einops import rearrange
		return self.copy(rearrange(self.values, pattern))
	def repeat(self, pattern, **kwargs):
		from einops import repeat
		return self.copy(repeat(self.values, pattern, **kwargs))
