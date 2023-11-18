

class DynamicDispatch:
	"""Register methods that dynamically dispatch
	according to static type-like attributes of its arguments

	This is an abstract base class; specific implementations should averride the 'attribute' method
	"""

	def __init__(self, doc=""):
		self.cache = {}
		self.options = []
		self.__doc__ = doc

	def attribute(self, arg):
		raise NotImplementedError

	def match(self, subspaces):
		for condition, func in self.options:
			if condition(*subspaces):
				self.cache[subspaces] = func
				return func
		raise Exception('No matching implementation found')

	def __call__(self, *args):
		attributes = tuple(self.attribute(a) for a in args)
		try:
			func = self.cache[attributes]
		except:
			func = self.match(attributes)
		return func(*args)

	def __get__(self, instance, owner):
		"""Need this so we can register these objects as methods; and get the self arg passed in to us
		Might not be desirable for free functions though; keep it optional?
		"""
		from types import MethodType
		return MethodType(self, instance) if not instance is None else self

	def register(self, condition=lambda *s: True, position=None):
		"""Add an implementation to our list of options"""
		position = len(self.options) if position is None else position
		def wrap(func):
			"""This wrapper gets called by the @ syntax"""
			self.options.insert(position, (condition, func))
			return func
		return wrap


class SubspaceDispatch(DynamicDispatch):
	"""Register methods that dynamically dispatch
	according to the subspace of a multivector"""
	def attribute(self, arg):
		return getattr(arg, 'subspace')
