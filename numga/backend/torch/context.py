import torch

from numga.backend.torch.operator import TorchEinsumOperator
from numga.backend.torch.multivector import TorchMultiVector
from numga.context import AbstractContext


class TorchContext(AbstractContext):
	def __init__(self, algebra, otype=TorchEinsumOperator, dtype=torch.float32, device='cpu'):
		super(TorchContext, self).__init__(
			algebra=algebra,
			otype=otype,
			dtype=dtype,
			mvtype=TorchMultiVector
		)
		self.device = device

	def coerce_array(self, values, dtype=None):
		return torch.asarray(values, dtype=dtype or self.dtype, device=self.device)

	def allocate_array(self, shape):
		# FIXME: empty might be more appropriate, usually?
		return torch.zeros(shape, dtype=self.dtype, device=self.device)

	def set_array(self, arr, idx, values):
		try:
			idx = torch.asarray(idx)
		except:
			pass

		arr = torch.clone(arr)
		arr[idx] = values
		return arr



	def where(self, cond, a, b):
		return torch.where(cond, a, b)

	def nan_to_num(self, x, nan):
		return torch.nan_to_num(x, nan=nan)
	def abs(self, x):
		return torch.abs(x)

	def sqrt(self, x):
		return torch.sqrt(x)
	def log(self, x):
		return torch.log(x)
	def exp(self, x):
		return torch.exp(x)
	def abs(self, x):
		return torch.abs(x)

	def pow(self, x1, x2):
		return torch.pow(x1, x2)

	def min(self, x, axis):
		return torch.min(x, dim=axis)
	def argmin(self, x, axis):
		return torch.argmin(x, dim=axis)
	def all(self, x, axis):
		return torch.all(x, dim=axis)
	def isnan(self, x):
		return torch.isnan(x)
	def logical_xor(self, x, y):
		return torch.logical_xor(x, y)
	def logical_or(self, x, y):
		return torch.logical_or(x, y)
