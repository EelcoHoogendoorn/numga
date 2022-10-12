import math
from typing import Tuple

from numga.backend.python.multivector import PythonMultiVector
from numga.operator.abstract import AbstractConcreteOperator


class PythonSparseOperator(AbstractConcreteOperator):
	"""Unroll a (generally sparse) kernel into computations on individual nonzero entries"""

	def __call__(self, *inputs: Tuple[PythonMultiVector]) -> PythonMultiVector:
		terms = self.precompute_sparse
		output = self.context.multivector(self.output)
		for ci, term in terms:
			output.values[ci] = sum(
				math.prod((inp.values[ii] for inp, ii in zip(inputs, idx)), start=scalar)
				for (idx, scalar) in term
			)
		return output

	# def partial(self, inputs: Dict[int, PythonMultiVector]) -> "PythonSparseOperator":
	# 	expr = self.precompute_einsum_partial(tuple(inputs.keys()))
	# 	return PythonSparseOperator(
	# 		self.context,
	# 		Operator(
	# 			np.einsum(expr, self.kernel, *(i.values for i in inputs.values())),
	# 			(a for i, a in enumerate(self.operator.axes) if i not in inputs)
	# 		)
	# 	)
