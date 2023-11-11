from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
from numga.multivector.multivector import AbstractMultiVector as mv


mv.square_root = SubspaceDispatch("""Square root s of x such that s * s == x
Note we only compute a single (arbitrarily chosen) square root.
Moreover, since numga only considers real GAs, 
square roots of negative scalars will return nan.

Moreover, we do not 'exhaustively' search for square roots;
for instance, the scalar -1 will have many roots in a typical GA,
formed by blades of various grades, depending on the signature;
but these implementations do not perform such an exhaustive search
""")
@mv.square_root.register(lambda s: s.inside.scalar())
def scalar_square_root(s: Scalar) -> Scalar:
	return s.sqrt()
@mv.square_root.register(lambda s: s.is_study and s.restrict.nonscalar().is_degenerate)
def study_degenerate_square_root(s: Study) -> Study:
	"""This applies to generalized study numbers with degenerate nonscalar"""
	s0, ns = s.restrict.scalar(), s.restrict.nonscalar()
	sqrt = s0.sqrt()
	# need nan_to_num to handle numerical zero case
	ci = (2 * sqrt).inverse().nan_to_num(0)
	return sqrt + ns * ci
@mv.square_root.register(lambda s: s.is_study)
def study_square_root(s: Study) -> Study:
	"""This applies to generalized study numbers"""
	s0, ns = s.restrict.scalar(), s.restrict.nonscalar()
	c = ((s0 + s.study_norm()) / 2).sqrt()
	ci = (c * 2).inverse().nan_to_num(0)
	return c + ns * ci


mv.inverse_square_root = SubspaceDispatch("""invsqrt(x), such that invsqrt(x) * x * invsqrt(x) = 1""")
@mv.inverse_square_root.register(lambda s: s.equals.scalar())
def scalar_inverse_square_root(s: Scalar) -> Scalar:
	return s.copy(s.values ** (-0.5))
# FIXME: there should be a fused optimized version of this for study numbers?
# @mv.inverse_square_root.register(lambda s: s.inside.study())
@mv.inverse_square_root.register()
def default_inverse_square_root(x):
	return x.square_root().inverse()
	# FIXME: we might get different result depending on order of ops with non-normalizable motors
	# return x.inverse().square_root()


# NOTE: we put this under motor prefix, since this formula only works for normalized input
mv.motor_square_root = SubspaceDispatch("""Square root s of x such that s * s == x. Input motor assumed normalized!""")
@mv.motor_square_root.register(lambda s: s.inside.even_grade())
def motor_square_root(m: Motor):
	return (m + 1).normalized()
mv.motor_geometric_mean = SubspaceDispatch("""Geometric mean of two motors. Input motors assumed normalized!""")
@mv.motor_geometric_mean.register(lambda l, r: l.inside.even_grade() and r.inside.even_grade())
def motor_geometric_mean(l: Motor, r: Motor):
	return (l + r).normalized()
