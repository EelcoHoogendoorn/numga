import abc

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.namespaces import SelectNamespace, RestrictNamespace
from numga.subspace.subspace import SubSpace
from numga.context import AbstractContext
from numga.multivector.helper import IndexHelper, SetterHelper


class AbstractMultiVector:
	"""Abstract multivector type;
	To be used as a base class for implementations in specific backends

	Note that there is essentially no GA-logic in this class at all;
	just boilerplate to forward overloaded operators on the multivector object,
	to the relevant multi-linear-operators, where the actual magic happens.

	Lie-group related functionality is added as dynamic dispatch extension method
	"""
	# derived classes should have a values attribute, that defines the backing storage
	values = None
	# algebraic (sub)space this multi-vector is part of
	# this ought to be viewed as an extension of the type of the multivector object,
	# since it is an immutable property that strongly influences the method dispatch of the object
	subspace: SubSpace
	# context this multivector belongs to; so it can access operator cache,
	# and allocate proper output types and all that good stuff
	context: AbstractContext

	def __init__(self, context, subspace, values):
		self.context = context
		self.subspace = subspace
		self.values = values

	@property
	def algebra(self) -> "Algebra":
		return self.subspace.algebra
	@property
	def operator(self) -> "OperatorFactory":
		return self.context.operator

	@classmethod
	def construct(cls, values, subspace) -> "AbstractMultiVector":
		"""A place to provide subtype specific type coercions and so on"""
		raise NotImplementedError

	def copy(self, values=None, subspace=None) -> "AbstractMultiVector":
		return self.construct(
			values=self.values.copy() if values is None else values,
			subspace=self.subspace if subspace is None else subspace,
			context=self.context,
		)

	@property
	def at(self):
		return IndexHelper(lambda v: self.copy(values=v), self.values, self.context)

	def __repr__(self):
		return f"{self.subspace.pretty_str}: <{self.subspace.named_str}>\n{self.values}"

	# FIXME: register below boilerplate in automated fashion?
	#  or forward to operator factory in getattr fashion?
	# unary ops
	def dual(self):
		"""Hodge dual"""
		return self.operator.right_hodge(self.subspace)(self)
	def dual_inverse(self):
		"""Hodge dual inverse"""
		return self.operator.right_hodge_inverse(self.subspace)(self)
	def reverse(self):
		"""~ operator, or blade-reversal"""
		return self.operator.reverse(self.subspace)(self)
	def involute(self):
		"""Involution, or blade-negation"""
		return self.operator.involute(self.subspace)(self)
	def conjugate(self):
		"""Conjugation; or combined reversion and involution"""
		return self.operator.conjugate(self.subspace)(self)

	# binary ops
	def product(self, other: "AbstractMultiVector") -> "AbstractMultiVector":
		"""Binary geometric product between self and other of given symmetry"""
		return self.operator.product(self.subspace, other.subspace)(self, other)
	def anti_product(self, other: "AbstractMultiVector") -> "AbstractMultiVector":
		"""Binary geometric product between self and other of given symmetry"""
		return self.operator.anti_product(self.subspace, other.subspace)(self, other)
	def wedge(self, other):
		return self.operator.wedge(self.subspace, other.subspace)(self, other)
	outer = wedge
	def anti_wedge(self, other):
		return self.operator.anti_wedge(self.subspace, other.subspace)(self, other)
	def regressive(self, other):
		return self.operator.regressive(self.subspace, other.subspace)(self, other)
	def cross(self, other):
		return self.operator.cross_product(self.subspace, other.subspace)(self, other)

	def inner(self, other):
		return self.operator.inner(self.subspace, other.subspace)(self, other)
	def anti_inner(self, other):
		return self.operator.anti_inner(self.subspace, other.subspace)(self, other)
	def scalar_product(self, other):
		return self.operator.scalar_product(self.subspace, other.subspace)(self, other)
	def bivector_product(self, other):
		return self.operator.bivector_product(self.subspace, other.subspace)(self, other)
	def join(self, other):
		return self.operator.join(self.subspace, other.subspace)(self, other)
	def meet(self, other):
		return self.operator.meet(self.subspace, other.subspace)(self, other)
	def commutator(self, other):
		return self.operator.commutator(self.subspace, other.subspace)(self, other)
	def commutator_anti(self, other):
		return self.operator.commutator_anti_product(self.subspace, other.subspace)(self, other)
	def anti_commutator_anti(self, other):
		return self.operator.anti_commutator_anti_product(self.subspace, other.subspace)(self, other)
	def anti_commutator(self, other):
		return self.operator.anti_commutator_product(self.subspace, other.subspace)(self, other)

	def commutator_product(self, other):
		return self.operator.commutator_product(self.subspace, other.subspace)(self, other)
	def commutator_anti_product(self, other):
		return self.operator.commutator_anti_product(self.subspace, other.subspace)(self, other)
	def anti_commutator_anti_product(self, other):
		return self.operator.anti_commutator_anti_product(self.subspace, other.subspace)(self, other)
	def anti_commutator_product(self, other):
		return self.operator.anti_commutator_product(self.subspace, other.subspace)(self, other)

	def left_contract(self, other):
		return self.operator.left_contract(self.subspace, other.subspace)(self, other)
	def right_contract(self, other):
		return self.operator.right_contract(self.subspace, other.subspace)(self, other)


	def squared(self):
		return self.operator.squared(self.subspace)(self, self)
	# def norm_squared(self):
	# 	return self.operator.norm_squared(self.subspace)(self, self)
	# def weight_norm_squared(self):
	# 	return self.operator.weight_norm_squared(self.subspace)(self, self)
	# def bulk_norm_squared(self):
	# 	return self.operator.bulk_norm_squared(self.subspace)(self, self)

	def reverse_product(self, other):
		return self.operator.reverse_product(self.subspace, other.subspace)(self, other)
	def anti_reverse_product(self, other):
		return self.operator.anti_reverse_product(self.subspace, other.subspace)(self, other)
	def symmetric_reverse_product(self):
		return self.operator.symmetric_reverse_product(self.subspace)(self, self)

	# our ternary operators
	def sandwich(self, other):
		# FIXME: allow other to be a subspace; then do partial application?
		#  one extra if of runtime overhead...
		#  also raises question if this should be supported on other ops.
		#  what if we just support subspaces in the sig of operator.__call__, genreally?
		#  some if overhead in numpy but doesnt matter in jax anyway. think i like that
		return self.operator.sandwich(self.subspace, other.subspace)(self, other, self)
	def reverse_sandwich(self, other):
		return self.operator.reverse_sandwich(self.subspace, other.subspace)(self, other, self)
	def full_sandwich(self, other):
		return self.operator.full_sandwich(self.subspace, other.subspace)(self, other, self)
	def transform(self, other):
		return self.operator.transform(self.subspace, other.subspace)(self, other, self)
	def reverse_transform(self, other):
		return self.operator.reverse_transform(self.subspace, other.subspace)(self, other, self)
	def inverse_factor(self):
		return self.operator.inverse_factor(self.subspace)(self, self, self)

	def project(self, other):
		return self.operator.project(self.subspace, other.subspace)(self, other, self)

	def degenerate(self):
		return self.operator.degenerate(self.subspace)(self)
	def nondegenerate(self):
		return self.operator.nondegenerate(self.subspace)(self)


	# NOTE: forming these throwaway helper objects for marginally cleaner syntax is kinda sketch
	#  but hey jax doesnt care
	@property
	def select(self):
		"""Subspace selection

		The returned multivector contains all components inside the selected subspace.
		Those components not present in the input multivector are treated as implicit zeros.

		Example
		-------
		motor.restrict[1] -> 1-vector of zeros
		"""
		return SelectNamespace(self)
	@property
	def restrict(self):
		"""Subspace restriction

		The returned multivector contains all components present in the selected subspace,
		intersected with the subspace components of the input multivector.

		Example
		-------
		motor.restrict[1] -> empty vector
		"""
		return RestrictNamespace(self)

	def select_subspace(self, subspace):
		return self.operator.select(self.subspace, subspace)(self)
	def restrict_subspace(self, subspace):
		return self.operator.restrict(self.subspace, subspace)(self)
	def select_grade(self, k):
		return self.select_subspace(self.algebra.subspace.k_vector(k))
	def restrict_grade(self, k):
		return self.operator.restrict(self.subspace, self.algebra.subspace.k_vector(k))(self)

	def combine(self, other, sign):
		"""fused-multiply-add on multivectors of arbitrary subspaces"""
		other = self.upcast(other)

		if self.subspace == other.subspace:
			return self.copy(values=self.values + other.values * sign)
		else:
			combined = self.subspace.union(other.subspace)

			# fuse sign into operator; is this actually superior to letting fused mul add handle things?
			# probably best to add cached sign bit to select operator
			return self.context.multivector(
				values=self.select_subspace(combined).values + other.select_subspace(combined).values * sign,
				subspace=combined
			)

	# operator overloads
	def __neg__(self):
		return self.copy(-self.values)
	def __pos__(self):
		return self

	def __invert__(self):
		"""~ operator denotes reverse"""
		return self.reverse()

	def __mul__(self, other):
		other = self.upcast(other)
		return self.product(other)
	def __rmul__(self, other):
		other = self.upcast(other)
		return other.product(self)

	def __xor__(self, other):
		return self.wedge(other)
	def __and__(self, other):
		return self.regressive(other)
	def __or__(self, other):
		return self.inner(other)

	def __add__(self, other):
		return self.combine(other, sign=+1)
	def __radd__(self, other):
		return self.combine(other, sign=+1)

	def __sub__(self, other):
		other = self.upcast(other)
		return self.combine(other, sign=-1)
	def __rsub__(self, other):
		other = self.upcast(other)
		return other.combine(self, sign=-1)

	def upcast(self, x: object) -> "Scalar":
		"""Upcast general objects to scalar multivectors"""
		if isinstance(x, type(self)):
			return x
		x = self.context.coerce_array(x)
		x = self.context.multivector.scalar(x[..., None])
		return x

	def _divide(self, other):
		# FIXME: should we use same dynamic dispatch functionality here?
		if other.subspace.equals.scalar():
			# shortcut the scalar case, so we dont end up doing self * (1/other)
			return self.copy(values=self.values / other.values)
		else:
			return self * other.inverse()

	def la_inverse(self):
		"""linear algebra based inverse"""
		raise NotImplementedError('implementation is backend specific')

	def __truediv__(self, other):
		other = self.upcast(other)
		return self._divide(other)
	def __rtruediv__(self, other):
		# FIXME: we have an odd failure case in simple_log. we can divide a float by a scalar, but not an array by a scalar
		#  does array division bind first or something?
		other = self.upcast(other)
		return other._divide(self)


	# FIXME: some libraries overload shift as contraction
	#  since we tend to use sandwiches a lot more, we prefer to use our limited operators for that
	def __rshift__(self, other):
		return self.sandwich(other)
	def __lshift__(self, other):
		return self.reverse_sandwich(other)

	# partially bound ternary->unary operators
	# would be preferrable if we could just pass subspace into the relevant operator,
	# to construct partially bound operators
	def sandwich_map(self, v: SubSpace = None) -> "AbstractConcreteOperator":
		"""Encode sandwich operation as linear map over the subspace v

		This can be used to map a quaternion to the equivalent rotation matrix,
		Or a motor to an affine matrix

		Note that this type of partial binding is possible for any operator,
		but it makes most sense for ternary operators like these,
		where binding to the quadratic argument first may serve as a useful precomputation,
		if we intend to process a large number of elements of the subspace v with this operation.
		"""
		return self.operator.sandwich(
			self.subspace,
			self.algebra.subspace.vector() if v is None else v
		).partial({0: self, 2: self})

	def project_map(self, v: SubSpace = None) -> "AbstractConcreteOperator":
		"""Encode projection operation as linear map over the subspace `v`

		`self` here is the thing we are projecting on; v the subspace to be projected
		"""
		return self.operator.project(
			self.subspace,
			self.algebra.subspace.vector() if v is None else v,
		).partial({0: self, 2: self})

	def inertia_map(self, b: SubSpace = None) -> "AbstractConcreteOperator":
		"""Construct unary operator, that maps rates to momenta, for the point in self.
		P v (P x B)
		"""
		return self.operator.inertia(
			self.subspace,
			self.algebra.subspace.bivector() if b is None else b,
		).partial({0: self, 1: self})

	# some basic elementwise math functionality
	def sqrt(self) -> "AbstractMultiVector":
		return self.copy(values=self.values ** 0.5)
	def invsqrt(self) -> "AbstractMultiVector":
		return self.copy(values=self.values ** -0.5)

	def nan_to_num(self, nan):
		return self.copy(self.context.nan_to_num(self.values, nan))
	def abs(self):
		return self.copy(self.context.abs(self.values))

	def cos(self):
		return self.copy(values=self.context.cos(self.values))
	def sin(self):
		return self.copy(values=self.context.sin(self.values))
