"""Sub-namespace objects to be used inside the multivector object,

This provides some de-crowding of an otherwise quite crowded namespace,
and permits some syntactic sugar like:
	bivector = multivector.select[2]
"""

from numga.subspace.subspace import SubSpace


class SelectNamespace:
	def __init__(self, multivector):
		self.multivector = multivector
		self.factory = multivector.context.subspace

	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		if isinstance(subspace, SubSpace):
			return self.multivector.select_subspace(subspace)
		else:
			return lambda *args: self.multivector.select_subspace(subspace(*args))

	def __getitem__(self, item):
		item = list(item) if isinstance(item, tuple) else [item]
		subspace = self.factory.from_grades(item)
		return self.multivector.select_subspace(subspace)


class RestrictNamespace:
	def __init__(self, multivector):
		self.multivector = multivector
		self.factory = multivector.context.subspace

	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		if isinstance(subspace, SubSpace):
			return self.multivector.restrict_subspace(subspace)
		else:
			return lambda *args: self.multivector.restrict_subspace(subspace(*args))

	def __getitem__(self, item):
		item = list(item) if isinstance(item, tuple) else [item]
		subspace = self.factory.from_grades(item)
		return self.multivector.restrict_subspace(subspace)
