"""
Norms and normalizations

Note: we should generalize study and reverse norms
to normalizations wrt arbitrary involutions

Also, should be adding a set of anti-norms
"""

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
from numga.multivector.multivector import AbstractMultiVector as mv


# NOTE: these methods have a study_ prefix;
# since otherwise theyd shadow equally named functionality with subtly different outcomes
mv.study_norm_squared = SubspaceDispatch("""Study norm squared""")
@mv.study_norm_squared.register(lambda s: s.is_study)
def study_norm_squared(s: Study) -> Scalar:
	return s.context.operator.study_norm_squared(s.subspace)(s, s)
mv.study_norm = SubspaceDispatch("""Study norm""")
@mv.study_norm.register(lambda s: s.is_study)
def study_norm(x: Study) -> Scalar:
	return x.study_norm_squared().sqrt()


mv.norm_squared = SubspaceDispatch("""
	Norm squared; note it does not always yield a positive value, 
	or a scalar for that matter! To the mathematical purist, 
	this isnt really a norm, but we are going by prevailing GA convention here
	""")
@mv.norm_squared.register(lambda s: s.is_reverse_n_simple(1))
def reverse_simple_norm_squared(x) -> Scalar:
	return x.symmetric_reverse_product()
@mv.norm_squared.register(lambda s: s.inside.even_grade())
def reverse_motor_norm_squared(x: Motor):
	return x.symmetric_reverse_product()
@mv.norm_squared.register()
def reverse_default_norm_squared(x):
	return x.symmetric_reverse_product()


mv.norm = SubspaceDispatch("""Note; the same naming caveats apply as to the squared-norm...""")
@mv.norm.register(lambda s: s.inside.scalar())
def reverse_scalar_norm(x):
	return x.abs()
@mv.norm.register()
def reverse_default_norm(x):
	return x.norm_squared().square_root()

# Note: should we call this reversed-normalized, to make the operation explicit? to contrast with study/scalar-neg norms?
mv.normalized = SubspaceDispatch("""Normalisation, such that x ~x == 1""")
# @mv.normalized.register(lambda s: s.is_study)
# def study_normalized(s):
# 	# FIXME: this is a lie; does not yield x * ~x = 1
# 	return s / s.study_norm()
@mv.normalized.register(lambda s: s.symmetric_reverse().equals.empty())
def reverse_empty_normalized(x):
	raise Exception("Unable to normalize null objects")
@mv.normalized.register(lambda s: s.symmetric_reverse().equals.scalar())
def reverse_scalar_normalized(x):
	"""Binds to 2d and 3d even grade, as well as other simple objects"""
	s = x.symmetric_reverse_product()
	s = s.inverse_square_root()
	return s * x
@mv.normalized.register(lambda s: s.symmetric_reverse().is_study)
def reverse_study_normalized(x):
	"""Binds to 4d and 5d even grade, as well as many other objects"""
	s = x.symmetric_reverse_product()
	s = s.square_root().inverse()
	return s * x
