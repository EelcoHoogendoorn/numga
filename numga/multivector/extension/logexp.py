"""
Note that the current log/exp implementations are based on a brute-force bisection method.
More optimized variants can and have been conceived for various signatures and memory layouts [1];
but we provide these numerical algorithms as a useful baseline implementation.

These implementations provide uniform numerical precision in all quadrants, plus well behaved differentiability,
unlike some optimized implementations that employ various trigonometric approaches.
These implementations should offer reasonable performance especially on hardware accelerators,
since they avoid trigonometry, nan-handling and branching altogether.

We suggest to first try if these log/exp implementations actually form a bottleneck for your application,
before overriding them with more specialized approaches, and then spending months chasing down jitters
introduced by poor numerical stability. Dont ask me how I know.

For examples of how to override with more optimized implementations, see extension_optimized.py

References
----------
[1] Normalization, Square Roots, and the Exponential and Logarithmic Maps in Geometric Algebras of Less than 6D

"""

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
from numga.multivector.multivector import AbstractMultiVector as mv


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
@mv.exp.register(lambda s: s.inside.bivector)
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
	Logarithm approximation for motors close to 1
	Exact inverse to exp_linear_normalized.""")
@mv.motor_log_linear_normalized.register(lambda s: s.inside.quadreflection())
def motor_log_linear_normalized(m: Motor) -> BiVector:
	# 'denormalize' the motor to have scalar == 1 and quadvector == 0
	denormalize = m.restrict.self_reverse().inverse()
	return m.bivector_product(denormalize)
mv.exp_linear_normalized = SubspaceDispatch("""
	Exponential approximation valid for small bivectors
	Exact inverse to motor_log_linear_normalized""")
@mv.exp_linear_normalized.register(lambda s: s.inside.bivector())
def exp_linear_normalized(b: BiVector) -> Motor:
	return (b + 1).normalized()

mv.exp_quadratic = SubspaceDispatch("""
	Quadratic exponential approximation.
	Exact inverse to log_quadratic""")
@mv.exp_quadratic.register()
def exp_cayley(b: BiVector) -> Motor:
	r = 1 + b / 2
	return r.squared() / r.symmetric_reverse_product()
mv.motor_log_quadratic = SubspaceDispatch("""
	Quadratic logarithm approximation.
	Exact inverse to exp_quadratic""")
@mv.motor_log_quadratic.register()
def motor_log_cayley(m: Motor) -> BiVector:
	return motor_log_linear_normalized(m.motor_square_root()) * 2


# # FIXME: can i make kwargs work with dynamic dispatch?
mv.motor_log_bisect = SubspaceDispatch("""
	Bisection based logarithm
	Exact inverse to exp_bisect""")
@mv.motor_log_bisect.register(lambda s: s.inside.even_grade())
def motor_log_bisect(m: Motor, n=15) -> BiVector:
	for i in range(n):  # FIXME: should prefer jax looping construct in jax context!
		m = m.motor_square_root()
	return motor_log_cayley(m) * (2 ** n)
mv.exp_bisect = SubspaceDispatch("""
	Bisection based exponential.
	Exact inverse to motor_log_bisect""")
@mv.exp_bisect.register()
def exp_bisect(b: BiVector, n=15) -> Motor:
	m = exp_cayley(b / (2 ** n))
	for i in range(n):
		m = m.squared()
	return m

