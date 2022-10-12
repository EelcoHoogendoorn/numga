
# FIXME: adopt .at syntax at base level? makes it easier to unify jax/numpy code

class SetterHelper:
	"""Helper objects to extend .at[].set syntax of jax arrays to jax multivectors and jax operators"""
	def __init__(self, helper, idx):
		self.helper = helper
		self.idx = idx
	def set(self, values):
		return self.helper.set(self.idx, values)


class IndexHelper:
	def __init__(self, copy, arr):
		self.copy = copy    # callable to propagate the mutation to the object owning arr.
		self.arr = arr      # underlying jnp.array to mutate
	def __getitem__(self, item):
		"""This is the whole point of this construct; to capture indexing syntax while being able to return the copy"""
		return SetterHelper(self, item)
	def set(self, idx, values):
		from numga.multivector.multivector import AbstractMultiVector
		if isinstance(values, AbstractMultiVector):
			values = values.values
		return self.copy(self.arr.at[idx].set(values))
