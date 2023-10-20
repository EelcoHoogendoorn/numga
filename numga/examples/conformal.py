"""Conformal geometric algebra

Note, this code is mostly a pile of disorganized experiments.
Not quite sure yet on the cleanest way to provide out-of-the-box CGA functionality extension.
Two different approaches are implemented here at the moment,
one based on dynamic injection of attributes,
another based on encapsulation of the original context object.

Use dynamic mixins?
Or just copy the entire structure, with dynamically created subclass?

does the dynamic method dispatch paradigm help here?
"""

from numga.util import cached_property


def extend_cls(base_obj, mixin_cls, prefix):
	base_cls = base_obj.__class__
	return type(prefix + base_cls.__name__, (mixin_cls, base_cls), {})


def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    name = cls.__name__ + base_cls_name
    obj.__class__ = type(name, (base_cls, cls),{})
    return obj


from numga.algebra.algebra import Algebra
from numga.util import summation
from numga.subspace.factory import SubSpaceFactory


class ConformalSubspaceFactory(SubSpaceFactory):
	def init(self):
		*X, p, n = self.algebra.subspace.basis()
		X = summation(X)
		# N = p.wedge(n)
		self.p = lambda: p
		self.n = lambda: n
		self.euclidian = lambda: X
		# self.N = lambda: N


from numga.multivector.factory import MultiVectorFactory


class ConformalMultiVectorFactory(MultiVectorFactory):
	def init(self):
		*X, p, n = self.basis()
		# definitions as used in Dorst
		ni = (p + n)
		no = (n - p) / 2
		N = ni ^ no  # this spans the null plane

		self.ni = lambda: ni
		self.no = lambda: no
		self.N = lambda: N
		self.euclidian = lambda x: self(subspace=self.algebra.subspace.euclidian(), values=x)


class ConformalContextMixin:
	def init_conformal(self):
		self.ni = self.multivector.ni()
		self.no = self.multivector.no()
		self.N = self.multivector.N()

	def embed_point(self, P, r=0):
		"""Embed a Euclidian point in CGA space"""
		# definitions as used in Dorst
		p = self.multivector.euclidian(P)
		# FIXME: implement as translation of origin?
		return ((self.ni * (p.norm_squared() - r*r) / 2 + self.no) + p)

	@cached_property
	def unembed(self):
		V = self.subspace.vector()
		# FIXME: need to add syntactic sugar to be able to construct operators from subspaces
		#  get to this syntax: op = (V ^ self.N) * self.N
		op = self.algebra.operator.wedge_product(V, self.N.subspace)
		op = self.algebra.operator.geometric_product(op, self.N.subspace)
		op = self.operator(op).partial({1: self.N, 2: self.N}).squeeze()
		return op
	def point_position(self, p):
		return self.unembed(p)

	def split_points(self, T):
		assert T.subspace in self.subspace.bivector()
		b = (T.norm_squared() * -1).sqrt()
		F = T / b
		P = (F + 1) / +2
		Pb = (F + -1) / -2
		Tn = T | self.ni
		# Lasenby et al. - A Covariant Approach to Geometry using Geometric A.pdf 4.11
		# use geo prod here; but 3-vec part is zero anyway?
		B = P | Tn
		A = Pb | Tn
		assert A.subspace in self.subspace.vector()
		assert B.subspace in self.subspace.vector()
		return A, B

	def normalize(self, o):
		"""normalize a CGA object o such that o.dot(ni) == -1"""
		# FIXME: create fused op for denominator
		return o / (-o.reverse().inner(self.ni))

	def project(self, a, b):
		return (a ^ self.ni) >> (b * b)



def conformalize_subspace_factory(context):
	*_, p, n = context.subspace.basis()

	context.subspace.rate = lambda: p
	context.subspace.n = lambda: n
	context.subspace.N = lambda: p.union(n)
	context.subspace.euclidian = lambda: context.subspace.vector().difference(context.subspace.N())


def conformalize_multivector_factory(context):
	*_, p, n = context.multivector.basis()
	# definitions as used in Dorst
	ni = (p + n)
	no = (n - p) / 2
	N = ni ^ no  # this spans the null plane

	context.multivector.ni = lambda: ni
	context.multivector.no = lambda: no
	context.multivector.N = lambda: N
	context.multivector.euclidian = lambda x: context.multivector(subspace=context.subspace.euclidian(), values=x)


def Conformalize(context):
	# FIXME: are there guaranteed no lingering references to old algebra?
	#  in any case, previously constructed subspaces wil be rendered invalid
	context.algebra = context.algebra * Algebra('p+n-')
	conformalize_subspace_factory(context)
	conformalize_multivector_factory(context)
	extend_instance(context, ConformalContextMixin)
	context.init_conformal()
	return context


class Conformal:

	@staticmethod
	def hyperbolic_algebra() -> "Algebra":
		return Algebra('p+n-')

	@staticmethod
	def algebra(description: str):
		"""Construct algebra with an added positive and negative dimension;
		"""
		algebra = Algebra(description) * Conformal.hyperbolic_algebra()
		*X, p, n = algebra.subspace.basis()

		# FIXME: is this type of monkey patching legit?
		#  or should be subclass the SubspaceFactory,
		#  and dependency-inject it into the algebra; prob cleaner...
		# algebra.subspace.p = lambda: p
		# algebra.subspace.n = lambda: n
		algebra.subspace.N = lambda: p.union(n)
		algebra.subspace.euclidian = lambda: algebra.subspace.vector().difference(algebra.subspace.N())

		return algebra

	def __init__(self, context):
		self.context = context

		*X, p, n = context.multivector.basis()
		# definitions as used in Dorst
		ni = (p + n)
		no = (n - p) / 2
		N = ni ^ no  # this spans the null plane

		context.multivector.ni = lambda: ni
		context.multivector.no = lambda: no
		context.multivector.N = lambda: N
		context.multivector.euclidian = lambda x: context.multivector(subspace=context.subspace.euclidian(), values=x)

		self.ni = ni
		self.no = no
		self.N = N
		# self.X = X

	def embed_point(self, P, r=0):
		"""Embed a Euclidian point in CGA space"""
		# definitions as used in Dorst
		p = self.context.multivector.euclidian(P)
		return ((self.ni * (p.norm_squared() - r*r) / 2 + self.no) + p)

	def point_position(self, p):
		assert p.subspace.inside.vector()
		"""reject parts containing N, keeping only euclidian components"""
		# FIXME: map back to parent algebra?
		pos = (p ^ self.N) * self.N
		assert pos.subspace in self.context.subspace.euclidian()
		return pos

	def split_points(self, T):
		"""split point-pair bivector into two vector-valued points"""
		assert T.subspace.inside.bivector()
		b = (T.norm_squared() * -1).sqrt()
		F = T / b
		# FIXME: is this related to the invariant decomposition?
		S = self.context.multivector.scalar([[1], [-1]])
		P = (F * S + 1) / 2
		Tn = T.inner(self.ni)
		# Lasenby et al. - A Covariant Approach to Geometry using Geometric A.pdf 4.11
		#  uses geo prod here; but 3-vec part is zero anyway?
		AB = P.inner(Tn)
		assert AB.subspace.inside.vector()
		return AB

	def normalize(self, o):
		"""normalize a CGA object o such that o.dot(ni) == -1"""
		return o / (-o.reverse().inner(self.ni))

	def project(self, a, b):
		return (a ^ self.ni) >> (b * b)
