import torch
from numga.multivector.multivector import AbstractMultiVector
from numga.subspace.subspace import SubSpace


class TorchMultiVector(AbstractMultiVector):
	"""Concrete realization of abstract type in torch"""
	subspace: SubSpace
	context: "TorchContext"
	# dense contiguous backing array
	values: torch.tensor

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
		# FIXME: torch indexing with ndarray behavior differs!
		try:
			idx = torch.asarray(idx)
		except:
			pass
		v = self.values[idx]
		return self.copy(values=v)

	@property
	def shape(self):
		"""Shape of the array, modulo subspace axis"""
		return tuple(self.values.size())[:-1]

	def take_along_axis(self, idx, axis):
		values = self.values
		values = torch.take_along_dim(values, idx[..., None], dim=axis)
		return self.copy(values=values)

	def concatenate(self, other, axis):
		assert self.subspace == other.subspace
		return self.copy(torch.cat([self.values, other.values], dim=axis))

	# FIXME: in the below, assert that subspace axes remain untouched?
	def sum(self, axis):
		return self.copy(values=self.values.sum(dim=axis))
	def mean(self, axis):
		return self.copy(values=self.values.mean(dim=axis))

	def rearrange(self, pattern):
		from einops import rearrange
		return self.copy(rearrange(self.values, pattern))
	def repeat(self, pattern, **kwargs):
		from einops import repeat
		return self.copy(repeat(self.values, pattern, **kwargs))
