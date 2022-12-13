"""Provide a context object to encapsulate all state associated with an algebra
and its associated objects"""
import abc

from numga.algebra.algebra import Algebra


# class ConcreteSubspace(SubSpace):
# 	"""
# 	Make subspace subclass with access to the context?
# 	That will allow us to construct concrete operators directly from subspaces
#
# 	# FIXME: or is it better to just subclass algebra and add mv-allocation features there?
# 	"""


class ConcreteOperatorFactory:
	"""Caches the creation of concrete operator instances
	from abstract operator instances

	NOTE: this is more a warehouse than a factory.
	also, whats the point? the symbolic operators are cached on the algebra already
	the concrete operator types have a trivial init.
	at best we are enabling recycling of some cached properties on the concrete operator type
	"""

	def __init__(self, context):
		self.context = context
		self.cache = {}

	def __call__(self, abstract_op):
		return self.context.otype(self.context, abstract_op)

	def __getattr__(self, operator_name: str):
		"""Wrap construction of an abstract operator in a concrete subtype

		Callable that will instantiate a concrete operator type in a cached manner
		"""
		def func(*args) -> "AbstractConcreteOperator":
			# defer the getattr to the abstract operator factory
			# note: this implies we need a 1-1 match between the operators defined in both
			# the abstract and concrete operator factories
			# note that we include the type of operator as well, to cover the case where otype is dynamic
			params = operator_name, args, str(self.context.otype)
			try:
				return self.cache[params]
			except:
				abstract_op = getattr(self.context.algebra.operator, operator_name)(*args)
				concrete_op = self(abstract_op)
				self.cache[params] = concrete_op
				return concrete_op
		return func


class AbstractContext:
	"""Abstract context object; to be overridden by backend specific logic"""

	def __init__(self, algebra: Algebra, dtype, mvtype, otype):
		self.algebra = algebra if isinstance(algebra, Algebra) else Algebra(algebra)
		self.dtype = dtype
		self.mvtype = mvtype    # provide these in the subclasses?
		self.otype = otype      # concrete operator type
		# FIXME: make these dependencies injectable
		self.operator_factory = ConcreteOperatorFactory(self)
		from numga.multivector.factory import MultiVectorFactory
		self.multivector_factory = MultiVectorFactory(self)
		# force registration of extension methods on multivector type
		# FIXME: why here?
		from numga.multivector import extension

	@property
	def multivector(self):
		return self.multivector_factory
	@property
	def operator(self):
		return self.operator_factory
	@property
	def subspace(self):
		return self.algebra.subspace


	@abc.abstractmethod
	def set_array(self, idx, value):
		"""Backend specific array mutation"""
		raise NotImplementedError


	# backend needs to provide some elementary math implementations
	@abc.abstractmethod
	def sqrt(self, x):
		pass
	@abc.abstractmethod
	def pow(self, x, y):
		pass
	@abc.abstractmethod
	def exp(self):
		pass
	@abc.abstractmethod
	def log(self):
		pass
	@abc.abstractmethod
	def nan_to_num(self, x, nan):
		pass

	@abc.abstractmethod
	def cos(self, x):
		pass
	@abc.abstractmethod
	def sin(self, x):
		pass
	@abc.abstractmethod
	def atan(self):
		pass
	@abc.abstractmethod
	def arctan2(self):
		pass
