from typing import List

from numga.multivector.multivector import AbstractMultiVector


class PythonMultiVector(AbstractMultiVector):
	"""Concrete realization of abstract type in JAX"""
	values: List

	@classmethod
	def construct(cls, context, values, subspace):
		if values is None:
			# scalar = 1, rest = 0 is the default init,
			# which is a sensible unit default in a multiplicative algebra
			values = subspace.blades == 0
		# coerce type for external given values
		values = list(float(v) for v in values)

		# first axis should enumerate the subspace;
		# not worth it to complicate downstream code with deviations from that pattern
		assert len(values) == len(subspace)

		return cls(
			context=context,
			values=list(values),
			subspace=subspace
		)


