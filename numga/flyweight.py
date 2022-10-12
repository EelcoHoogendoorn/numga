"""Simple implementation of a flyweight pattern

This is used to make subspace objects trivially comparable and hashable
"""

class FlyweightMixin:
	def __hash__(self):
		return id(self)
	def __eq__(self, other):
		return self is other


class FlyweightFactory:
	def __init__(self):
		self.flyweight_pool = {}
	def factory_construct(self, key, value) -> FlyweightMixin:
		if key in self.flyweight_pool:
			return self.flyweight_pool[key]
		else:
			value = value()
			self.flyweight_pool[key] = value
			return value
