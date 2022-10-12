"""
This module provides extension methods to the multivector type,
that are not simple linear operators generic to all signatures,
but have some reason to be dispatched to different implementations depending on their subspace

This involves mostly norms, square roots, inverses, logarithms and exponentials.
These are all implemented here for dimensions < 6, in a backend and signature agnostic manner.


Note that the current log/exp implementations are based on a brute-force bisection method.
More optimized variants can and have been conceived for various signatures and memory layouts;
but we provide these numerical algorithms as a useful baseline implementation.

These implementations provide uniform numerical precision in all quadrants, plus well behaved differentiability,
unlike some optimized implementations that employ various trigonometric approaches.
These implementations should offer reasonable performance especially on hardware accelerators,
since they avoid trigonometry, nan-handling and branching altogether.

We suggest to first try if these log/exp implementations actually form a bottleneck for your application,
before overriding them with more specialized approaches, and then spending months chasing down jitters
introduced by poor numerical stability. Dont ask me how I know.

For examples of how to override with more optimized implementations, see extension_optimized.py

"""

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
from numga.multivector.multivector import AbstractMultiVector as mv


mv.study_conjugate = SubspaceDispatch("""Study conjugation; tack a minus sign on grade 4 components""")
@mv.study_conjugate.register(lambda s: s.inside.study())
def study_conjugate(s: Study) -> Study:
	return s.operator.study_conjugate(s.subspace)(s)
mv.study_norm_squared = SubspaceDispatch("""Study norm squared""")
@mv.study_norm_squared.register(lambda s: s.inside.study())
def study_norm_squared(s: Study) -> Scalar:
	return s.context.operator.study_norm_squared(s.subspace)(s, s)
mv.study_norm = SubspaceDispatch("""Study norm""")
@mv.study_norm.register(lambda s: s.inside.study())
def study_norm(x: Study) -> Scalar:
	return x.study_norm_squared().square_root()


mv.norm_squared = SubspaceDispatch("""Norm squared; note it does not always yield a positive value, or a scalar for that matter! To the mathematical purist, this isnt really a norm, but we are going by prevailing GA convention here""")
@mv.norm_squared.register(lambda s: s.is_reverse_simple)
def reverse_simple_norm_squared(x) -> Scalar:
	return x.symmetric_reverse_product()
@mv.norm_squared.register(lambda s: s.inside.even_grade())
def motor_norm_squared(x: Motor):
	return x.symmetric_reverse_product()


mv.square_root = SubspaceDispatch("""Square root s of x such that s * s == x""")
@mv.square_root.register(lambda s: s.inside.scalar())
def scalar_square_root(s: Scalar) -> Scalar:
	return s.sqrt()
@mv.square_root.register(lambda s: s.inside.study())
def study_square_root(s: Study) -> Study:
	s0, s4 = s.select[0], s.restrict[4]
	c = ((s0 + study_norm(s)) / 2).square_root()
	ci = (c * 2).inverse().nan_to_num(0)
	return c + s4 * ci

# NOTE: put under motor name, since this formula only works for normalized input
mv.motor_square_root = SubspaceDispatch("""Square root s of x such that s * s == x""")
@mv.motor_square_root.register(lambda s: s.inside.motor())
def motor_square_root(m: Motor):
	return (m + 1).normalized()
mv.motor_geometric_mean = SubspaceDispatch("""Geometric mean of two motors""")
@mv.motor_geometric_mean.register(lambda l, r: l.inside.motor() and r.inside.motor())
def motor_geometric_mean(l: Motor, r: Motor):
	return (l + r).normalized()


mv.norm = SubspaceDispatch("""Note; the same naming caveats apply as to the squared-norm...""")
@mv.norm.register(lambda s: s.inside.scalar())
def scalar_norm(x):
	return x.abs()
@mv.norm.register()
def default_norm(x):
	return x.norm_squared().square_root()


mv.normalized = SubspaceDispatch("""Yeah... what does it mean anyway?""")
@mv.normalized.register(lambda s: s.equals.empty())
def empty_normalized(e):
	raise NotImplementedError
@mv.normalized.register()
def default_normalized(x):
	return x / x.norm()


mv.motor_inverse = SubspaceDispatch("""Inversion, assuming normalization""")
@mv.motor_inverse.register(lambda s: s.inside.motor())
def motor_inverse_normalized(m: Motor) -> Motor:
	return m.reverse()


mv.inverse = SubspaceDispatch("""Inversion inv(x), such that inv(x) * x = 1, not assuming normalization""")
@mv.inverse.register(lambda s: s.equals.empty())
def empty_inverse(s: Scalar) -> Scalar:
	s = s.context.multivector.scalar()
	return s / (s * 0) # note; this is a bit of a lie; inv(x) x != 1. why provide this franken-codepath again?
@mv.inverse.register(lambda s: s.equals.scalar())
def scalar_inverse(s: Scalar) -> Scalar:
	return s.copy(1 / s.values)
# @mv.inverse.register(lambda s: s.is_simple)   # reverse-simple seems to be the more general pattern; is there a real need for this?
# def simple_inverse(s):
# 	return s / s.squared()
@mv.inverse.register(lambda s: s.is_reverse_simple)
def reverse_simple_inverse(s):
	return s.reverse() / s.symmetric_reverse_product()
@mv.inverse.register(lambda s: s.inside.study())
def study_inverse(s: Study) -> Study:
	return s.study_conjugate() / s.study_norm_squared()
@mv.inverse.register(lambda s: s.inside.even_grade())
def even_inverse(m: Motor) -> Motor:
	return m.reverse() / m.norm_squared()


mv.inverse_square_root = SubspaceDispatch("""invsqrt(x), such that invsqrt(x) * sqrt(x) = 1""")
@mv.inverse_square_root.register(lambda s: s.equals.scalar())
def scalar_inverse_square_root(s: Scalar) -> Scalar:
	return s.copy(s.values ** (-0.5))
@mv.inverse_square_root.register(lambda s: s.is_reverse_simple)
def reverse_simple_inverse_square_root(s):
	return s.reverse() / s.norm()
@mv.inverse_square_root.register(lambda s: s.inside.study())
def study_inverse_square_root(s: Study) -> Study:
	return study_conjugate(s) / study_norm(s)
@mv.inverse_square_root.register()
def default_inverse_square_root(x):
	return x.square_root().inverse()



mv.exp = SubspaceDispatch("""Exponentials you know and love""")
@mv.exp.register(lambda s: s.equals.empty())
def empty_exp(e) -> Scalar:
	"""Empty is an implicit zero, which exps to one"""
	return e.context.multivector.scalar()   # FIXME: broadcasting?
@mv.exp.register(lambda s: s.equals.scalar())
def scalar_exp(s: Scalar) -> Scalar:
	"""Vanilla scalar exp"""
	return s.copy(s.context.exp(s.values))
@mv.exp.register(lambda s: s.is_degenerate)
def degenerate_exp(b: BiVector) -> Motor:
	"""If we know all basis elements square to zero, might as well use that information"""
	return b + 1
@mv.exp.register(lambda s: s.inside.bivector())
def default_exp(b: BiVector) -> "Motor":
	"""default to numerical brute force"""
	return b.exp_bisect()


mv.log = SubspaceDispatch("""Logarithm; inverse of exponential""")
@mv.log.register(lambda s: s.equals.empty())
def empty_log(e) -> Scalar:
	raise NotImplementedError   # FIXME: just return negative inf no?
@mv.log.register(lambda s: s.equals.scalar())
def scalar_log(s: Scalar) -> Scalar:
	return s.copy(s.context.log(s.values))


mv.motor_log = SubspaceDispatch("""Logarithm; inverse of exponential. Normalized motor is assumed as input""")
@mv.motor_log.register(lambda s: s.is_degenerate_motor)
def degenerate_log(m: Motor) -> BiVector:
	"""This should bind to translators"""
	return m.restrict[2]
@mv.motor_log.register(lambda m: m.inside.even_grade())
def default(m: Motor) -> BiVector:
	return m.motor_log_bisect()


mv.motor_log_linear = SubspaceDispatch("""
	Linear Logarithm; inverse of exp_linear. """)
@mv.motor_log_linear.register(lambda s: s.inside.even_grade())
def motor_log_linear(m: Motor) -> BiVector:
	return m.restrict[2]
mv.exp_linear = SubspaceDispatch("""
	Linear exponential; inverse of logarithm. 
	NOTE: output is not normalized!""")
@mv.exp_linear.register(lambda s: s.inside.bivector())
def exp_linear(b: BiVector) -> Motor:
	return b + 1


mv.motor_log_linear_normalized = SubspaceDispatch("""
	Exact inverse to exp_linear_normalized.""")
@mv.motor_log_linear_normalized.register(lambda s: s.inside.quadreflection())
def motor_log_linear_normalized(m: Motor) -> BiVector:
	# 'denormalize' the motor to have scalar == 1 and quadvector == 0
	denormalize = m.restrict.mod4().inverse()
	return m.bivector_product(denormalize)
mv.exp_linear_normalized = SubspaceDispatch("""
	Exact inverse to motor_log_linear_normalized""")
@mv.exp_linear_normalized.register(lambda s: s.inside.bivector())
def exp_linear_normalized(b: BiVector) -> Motor:
	return (b + 1).normalized()


# # FIXME: can i make kwargs work with dynamic dispatch?
mv.motor_log_bisect = SubspaceDispatch("""
	Bisection based logarithm
	Exact inverse to exp_bisect""")
@mv.motor_log_bisect.register(lambda s: s.inside.even_grade())
def motor_log_bisect(m: Motor, n=16) -> BiVector:
	for i in range(n):  # FIXME: should prefer jax looping construct in jax context!
		m = m.motor_square_root()
	return motor_log_linear_normalized(m) * (2 ** n)
mv.exp_bisect = SubspaceDispatch("""
	Bisection based exponential.
	Exact inverse to motor_log_bisect""")
@mv.exp_bisect.register(lambda s: s.inside.bivector())
def exp_bisect(b: BiVector, n=16) -> Motor:
	m = exp_linear_normalized(b / (2 ** n))
	for i in range(n):
		m = m.squared()
	return m
