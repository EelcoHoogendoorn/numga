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


# NOTE: these methods have a study_ prefix;
# since otherwise theyd shadow equally named functionality with subtly different outcomes
mv.study_norm_squared = SubspaceDispatch("""Study norm squared""")
@mv.study_norm_squared.register(lambda s: s.inside.study())
def study_norm_squared(s: Study) -> Scalar:
	return s.context.operator.study_norm_squared(s.subspace)(s, s)
mv.study_norm = SubspaceDispatch("""Study norm""")
@mv.study_norm.register(lambda s: s.inside.study())
def study_norm(x: Study) -> Scalar:
	return x.study_norm_squared().sqrt()


mv.norm_squared = SubspaceDispatch("""Norm squared; note it does not always yield a positive value, or a scalar for that matter! To the mathematical purist, this isnt really a norm, but we are going by prevailing GA convention here""")
@mv.norm_squared.register(lambda s: s.is_reverse_n_simple(1))
def reverse_simple_norm_squared(x) -> Scalar:
	return x.symmetric_reverse_product()
@mv.norm_squared.register(lambda s: s.inside.even_grade())
def motor_norm_squared(x: Motor):
	return x.symmetric_reverse_product()
@mv.norm_squared.register()
def default_norm_squared(x):
	return x.symmetric_reverse_product()


mv.square_root = SubspaceDispatch("""Square root s of x such that s * s == x""")
@mv.square_root.register(lambda s: s.inside.scalar())
def scalar_square_root(s: Scalar) -> Scalar:
	return s.sqrt()
@mv.square_root.register(lambda s: s.inside.study())
def study_square_root(s: Study) -> Study:
	s0, s4 = s.select[0], s.restrict[4]
	c = ((s0 + study_norm(s)) / 2).sqrt()   # +- study_norm should both be roots; why pick one over the other?
	ci = (c * 2).inverse().nan_to_num(0)
	return c + s4 * ci

# NOTE: we put this under motor prefix, since this formula only works for normalized input
mv.motor_square_root = SubspaceDispatch("""Square root s of x such that s * s == x. Input motor assumed normalized!""")
@mv.motor_square_root.register(lambda s: s.inside.even_grade())
def motor_square_root(m: Motor):
	return (m + 1).normalized()
mv.motor_geometric_mean = SubspaceDispatch("""Geometric mean of two motors. Input motors assumed normalized!""")
@mv.motor_geometric_mean.register(lambda l, r: l.inside.even_grade() and r.inside.even_grade())
def motor_geometric_mean(l: Motor, r: Motor):
	return (l + r).normalized()


mv.norm = SubspaceDispatch("""Note; the same naming caveats apply as to the squared-norm...""")
@mv.norm.register(lambda s: s.inside.scalar())
def scalar_norm(x):
	return x.abs()
@mv.norm.register()
def default_norm(x):
	return x.norm_squared().square_root()


mv.normalized = SubspaceDispatch("""Normalisation, such that x ~x == 1""")
@mv.normalized.register(lambda s: s.equals.empty())
def empty_normalized(e):
	raise NotImplementedError
@mv.normalized.register(lambda s: s.inside.study())
def study_normalized(s):
	return s / s.study_norm()
# Note; this does not provide 1-grade preservation for motors in dimensions >= 6!
@mv.normalized.register()
def default_normalized(x):
	return x.norm_squared().inverse_square_root() * x


mv.motor_inverse = SubspaceDispatch("""Inversion, assuming normalization""")
@mv.motor_inverse.register(lambda s: s.inside.motor())
def motor_inverse_normalized(m: Motor) -> Motor:
	return m.reverse()


mv.inverse = SubspaceDispatch("""Inversion inv(x), such that inv(x) * x = 1 = x * inv(x), not assuming normalization""")
@mv.inverse.register(lambda s: s.equals.empty())
def empty_inverse(s: "Empty") -> Scalar:
	"""note; this is a bit of a lie; inv(x) x != 1.
	why provide this franken-codepath again?
	to be more compatible with normal float semantics?"""
	s = s.context.multivector.scalar()
	return s / (s * 0)
@mv.inverse.register(lambda s: s.equals.scalar())
def scalar_inverse(s: Scalar) -> Scalar:
	return s.copy(1 / s.values)

def simplify_inverses(n):
	"""Register inversion rules based on simplifying the expression by self-involutions
	We register the simplest paths first and only go 3 levels of recursion deep;
	This suffices to invert in dimension < 6,
	and in higher dimensions we need different strats not more recursion
	see: https://arxiv.org/pdf/1712.05204.pdf
	"""
	@mv.inverse.register(lambda s: s.is_squared_n_simple(n))
	def squared_simple_inverse(s):
		"""Gives squared option precedence since it has the lowest op count"""
		return s / s.squared()
	@mv.inverse.register(lambda s: s.is_reverse_n_simple(n))
	def reverse_simple_inverse(s):
		"""This is the classic complex-number-like case"""
		return s.reverse() / s.symmetric_reverse_product()
	@mv.inverse.register(lambda s: s.is_conjugate_n_simple(n))
	def conjugate_simple_inverse(s):
		"""This is the classic complex-number-like case"""
		return s.conjugate() / s.symmetric_conjugate_product()
	@mv.inverse.register(lambda s: s.is_scalar_negation_n_simple(n))
	def scalar_negation_simple_inverse(s):
		"""This is the defacto study-inverse"""
		return s.scalar_negation() / s.symmetric_scalar_negation_product()
	@mv.inverse.register(lambda s: s.is_pseudoscalar_negation_n_simple(n))
	def pseudoscalar_negation_simple_inverse(s):
		# FIXME: need to find optimal formulation and name for this one; many options
		return s.pseudoscalar_negation() / s.symmetric_pseudoscalar_negation_product()
	@mv.inverse.register(lambda s: s.is_involute_n_simple(n))
	def involute_simple_inverse(s):
		"""Does not seem like we ever need this one"""
		return s.involute() / s.symmetric_involute_product()
# register 3 levels of recursion in inversion
for n in [1, 2, 3]:
	simplify_inverses(n)


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
@mv.exp.register(lambda s: s.equals.bivector)
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
@mv.motor_log.register(lambda s: s.inside.bireflection() and s.is_degenerate_scalar)
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
@mv.exp_bisect.register()
def exp_bisect(b: BiVector, n=16) -> Motor:
	m = exp_linear_normalized(b / (2 ** n))
	for i in range(n):
		m = m.squared()
	return m
