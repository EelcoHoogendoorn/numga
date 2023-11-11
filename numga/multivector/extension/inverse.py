"""
All multivectors in d < 6 are invertable with the methods implemented here,
and for higher dimensions, the shirokov method is implemented.
The method used in d < 6 is based on recursively applying 'involution rules'
of the same kind 'classical complex number inversion' is based on.
This gives correct inversions in all d < 6 in at most 3 recursion steps,
and the known optimal algorithms for the 'relevant' subspaces.

However, in 4d it may result in the resulting inverse having numerical zero terms,
that in theory we could prove to be actual zeros at compile time,
and in 5d this in turn may result in a 3-step recursion being taken where a 2-step
would have sufficed. Further optimizing this requires symbolically simplifying expressions
over a longer fused context than just immediate product terms,
and would require a substantial increase in code complexity.
If it bothers you a grade [1,2,5] multivector inverts in 3 steps rather than 2,
its probably best for now to just register a special case inversion formula for that.
In 5d numerical zeros can appear in the inverse in some non-exotic cases,
notably grade 2 inverts to grade 2 + 0 * 4 and grade 3 to 0*1 + 3 without special attention;
These have been patched with a special case dispatch.

"""

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
from numga.multivector.multivector import AbstractMultiVector as mv


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
		"""We need an additional involution to find inverses in 5d"""
		# FIXME: need to find optimal formulation and name for this one; many options
		return s.pseudoscalar_negation() / s.symmetric_pseudoscalar_negation_product()
	@mv.inverse.register(lambda s: s.is_involute_n_simple(n))
	def involute_simple_inverse(s):
		"""Does not seem like we ever need this one given the above"""
		return s.involute() / s.symmetric_involute_product()

def simplify_inverses_register_higher_order(n):
	@mv.inverse.register(lambda s: s.is_alt_n_simple(n))
	def alt_simple_inverse(s):
		print('alt ', end='')
		op_l = s.operator.inverse_factor_alt(s)
		# op_r = s.operator.inverse_factor_alt_completed(s)
		q = op_l(s,s,s) # if we know we reduce to a scalar we can reuse this computation
		return q / q.scalar_product(s)

def register_special_case_5d():
	# register these two special hand optimized cases for 5d
	# dont like these single grade cases getting unnecessary zeros otherwise
	@mv.inverse.register(lambda s: s.inside.bivector())
	def bivector_inverse_5d(b: BiVector) -> BiVector:
		return b.bivector_product(-(b.symmetric_reverse_product().inverse()))

	@mv.inverse.register(lambda s: s.inside.trivector())
	def trivector_inverse_5d(t):
		return t.trivector_product(t.squared().inverse())


# register 3 levels of recursion in inversion
simplify_inverses(1)
# dont want these enabled by default; kinda slow
# simplify_inverses_register_higher_order(2)

register_special_case_5d()

simplify_inverses(2)
simplify_inverses(3)
