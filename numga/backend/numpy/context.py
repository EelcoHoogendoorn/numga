import numpy as np

from numga.backend.numpy.multivector import NumpyMultiVector
from numga.backend.numpy.operator import NumpyEinsumOperator
from numga.context import AbstractContext


class NumpyContext(AbstractContext):
	"""NumpyEinsumOperator is the default operator type
	Unroled sparse operators would only make sense for large batch sizes, and high dimensional products
	"""
	def __init__(self, algebra, dtype=np.float64, otype=NumpyEinsumOperator, order='F'):
		self.order = order
		super(NumpyContext, self).__init__(
			algebra=algebra,
			dtype=dtype,
			otype=otype,
			mvtype=NumpyMultiVector
		)

	def coerce_array(self, values, dtype=None):
		return np.asarray(values, dtype=dtype or self.dtype, order=self.order)

	def allocate_array(self, shape):
		# FIXME: empty might be more appropriate, usually?
		return np.zeros(shape, dtype=self.dtype, order=self.order)

	def set_array(self, arr, idx, values):
		arr = arr.copy()
		arr[idx] = values
		return arr


	# math forward functions.
	# FIXME: should we wrap them in multivectors, or work in backend-specific arrays??
	def arctan2(self, x1, x2):
		return np.arctan2(x1, x2)
	def arctan(self, x):
		return np.arctan(x)
	def arccos(self, x):
		return np.arccos(x)
	def arcsinh(self, x):
		return np.arcsinh(x)

	def sqrt(self, x):
		return np.sqrt(x)
	def log(self, x):
		return np.log(x)
	def exp(self, x):
		return np.exp(x)
	def abs(self, x):
		return np.abs(x)

	def pow(self, x1, x2):
		return np.pow(x1, x2)

	def where(self, cond, a, b):
		return np.where(cond, a, b)

	def trigonometry(self, t):
		"""Implement trigonometry required for exponentiation for a specific backend"""
		norm = np.sqrt(t + 0j)
		cosh = np.cosh(norm)
		sinh = np.where(norm==0+0j, 1, np.sinh(norm) / norm)
		return cosh.real, sinh.real

	def nan_to_num(self, x, nan):
		return np.nan_to_num(x, nan=nan)
	def min(self, x, **kwargs):
		return np.min(x, **kwargs)
	def argmin(self, x, **kwargs):
		return np.argmin(x, **kwargs)
	def all(self, x, **kwargs):
		return np.all(x, **kwargs)
	def isnan(self, x):
		return np.isnan(x)
	def logical_xor(self, x, y):
		return np.logical_xor(x, y)
	def logical_or(self, x, y):
		return np.logical_or(x, y)

