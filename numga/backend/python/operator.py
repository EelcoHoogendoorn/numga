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


class PythonCodegenOperator(AbstractConcreteOperator):
	"""quick and dirty python codegen example"""

	def __call__(self, *inputs: Tuple[PythonMultiVector]) -> PythonMultiVector:
		terms = self.precompute_sparse
		inputs = [f'i{i}' for i in range(self.operator.arity)]
		inp = ','.join(inputs)
		text = f'def foo({inp}):\n'
		def make_term(idx, scalar):
			q = [inp + f'[{ii}]' for inp, ii in zip(inputs, idx)]
			return '*'.join([str(scalar)] + q)

		def make_line(ci, term):
			return f'\to[{ci}] = ' + '+'.join(make_term(idx, scalar) for idx, scalar in term)

		return text + '\n'.join(make_line(ci, term) for ci, term in terms)

# def partial(self, inputs: Dict[int, PythonMultiVector]) -> "PythonSparseOperator":
	# 	expr = self.precompute_einsum_partial(tuple(inputs.keys()))
	# 	return PythonSparseOperator(
	# 		self.context,
	# 		Operator(
	# 			np.einsum(expr, self.kernel, *(i.values for i in inputs.values())),
	# 			(a for i, a in enumerate(self.operator.axes) if i not in inputs)
	# 		)
	# 	)
