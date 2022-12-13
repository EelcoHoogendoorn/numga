from jax import numpy as jnp

from numga.backend.jax.operator import JaxEinsumOperator
from numga.backend.jax.multivector import JaxMultiVector
from numga.context import AbstractContext


class JaxContext(AbstractContext):
	def __init__(self, algebra, otype=JaxEinsumOperator, dtype=jnp.float32):
		super(JaxContext, self).__init__(
			algebra=algebra,
			otype=otype,
			dtype=dtype,
			mvtype=JaxMultiVector
		)

	def coerce_array(self, values, dtype=None):
		return jnp.asarray(values, dtype=dtype or self.dtype)

	def allocate_array(self, shape):
		# FIXME: empty might be more appropriate, usually?
		return jnp.zeros(shape, dtype=self.dtype)

	def set_array(self, arr, idx, values):
		return arr.at[idx].set(values)

	# def trigonometry(self, t):
	# 	"""Implement trigonometry required for exponentiation for a specific backend"""
	# 	norm = jnp.sqrt(t + 0j)
	# 	cosh = jnp.cosh(norm)
	# 	sinh = jnp.where(norm==0+0j, 1, jnp.sinh(norm) / norm)
	# 	return cosh.real, sinh.real

	# plain array operation forwarding. can this be more elegant?
	def where(self, cond, a, b):
		return jnp.where(cond, a, b)
	def nan_to_num(self, x, nan):
		return jnp.nan_to_num(x, nan=nan)
	def abs(self, x):
		return jnp.abs(x)

	def sqrt(self, x):
		return jnp.sqrt(x)
	def log(self, x):
		return jnp.log(x)
	def exp(self, x):
		return jnp.exp(x)
	def abs(self, x):
		return jnp.abs(x)

	def pow(self, x1, x2):
		return jnp.pow(x1, x2)

	def min(self, x, **kwargs):
		return jnp.min(x, **kwargs)
	def argmin(self, x, **kwargs):
		return jnp.argmin(x, **kwargs)
	def all(self, x, **kwargs):
		return jnp.all(x, **kwargs)
	def isnan(self, x):
		return jnp.isnan(x)
	def logical_xor(self, x, y):
		return jnp.logical_xor(x, y)
	def logical_or(self, x, y):
		return jnp.logical_or(x, y)
	# where = jnp.where