"""Sub-namespace objects to be used inside the subspace object,
this provides some de-crowding of an otherwise quite crowded namespace,
and permits some syntactic sugar like:
	bivector = multivector.select[2]
"""



class InNamespace:
	def __init__(self, subspace):
		self.subspace = subspace
		self.factory = subspace.algebra.subspace

	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		from numga.subspace.subspace import SubSpace
		if isinstance(subspace, SubSpace):
			return self.subspace in subspace
		else:
			return lambda *args: self.subspace in subspace(*args)

	def __call__(self, subspace):
		return self.subspace in subspace


class IsNamespace:
	def __init__(self, subspace):
		self.subspace = subspace
		self.factory = subspace.algebra.subspace

	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		from numga.subspace.subspace import SubSpace
		if isinstance(subspace, SubSpace):
			return self.subspace == subspace
		else:
			return lambda *args: self.subspace == subspace(*args)

	def __call__(self, subspace):
		return self.subspace == subspace


class SelectNamespace:
	def __init__(self, subspace):
		self.subspace = subspace
		self.factory = subspace.algebra.subspace

	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		from numga.subspace.subspace import SubSpace
		if isinstance(subspace, SubSpace):
			return self.subspace.select_subspace(subspace)
		else:
			return lambda *args: self.subspace.select_subspace(subspace(*args))

	def __getitem__(self, item):
		subspace = self.factory.from_grades(item)
		return self.subspace.select_subspace(subspace)

	def __call__(self, subspace):
		return self.subspace.select_subspace(subspace)


class RestrictNamespace(SelectNamespace):
	def __getattr__(self, item):
		subspace = getattr(self.factory, item)
		from numga.subspace.subspace import SubSpace
		if isinstance(subspace, SubSpace):
			return self.subspace.restrict_subspace(subspace)
		else:
			return lambda *args: self.subspace.restrict_subspace(subspace(*args))
