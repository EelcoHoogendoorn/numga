"""Plain python backend.

May seem kinda silly since we have a numpy dependency in this project anyway,
but there could be a legitimate use case for it; if acting on unbatched multivectors,
in a high dimenional space, a sparse product of python scalars should beat the numpy implementation,
since the latter will be dominated by numpy call overhead.

However, its primary purpose is as a benchmarking reference point.
"""
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


