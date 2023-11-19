"""Implement some decomposition algorithms"""

from typing import Tuple

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import *
from numga.multivector.multivector import AbstractMultiVector as mv


mv.decompose_polar = SubspaceDispatch("""
	compute b = L * s; a polar decomposition into a bivector line and a study number

	L * s0, L * s4 gives two orthogonal bivectors, with commuting exponent,
	which coincide with the invariant decomposition

	In 4d space, this describes a screw motion

	References
	----------
	PGA4CS eq 68-70
	""")
@mv.decompose_polar.register(lambda s: s.inside.bivector() and s.squared().inside.study())
def decompose_polar(b: BiVector) -> Tuple[BiVector, Study]:
	# FIXME: special case the b=0 or not s.inv() case?
	s: Study = b.symmetric_reverse_product().square_root()
	si: Study = s.inverse().nan_to_num(0)
	return b.bivector_product(si), s


mv.decompose_invariant = SubspaceDispatch("""
	Decompose a bivector b into two orthogonal simple bivectors (l, r) such that,

		b = l + r
		l.inner(r) = 0
		(l*l).subspace == (r*r).subspace == subspace.scalar

	References
	----------
	Normalization, Square Roots, and the Exponential and Logarithmic Maps in Geometric Algebras of Less than 6D
	""")
@mv.decompose_invariant.register(lambda s: s.inside.bivector() and s.squared().inside.scalar())
def bivector_decompose_simple(b: BiVector) -> Tuple[BiVector, BiVector]:
	return b, b.context.multivector.empty()
@mv.decompose_invariant.register(lambda s: s.inside.bivector() and s.squared().is_study)
def bivector_decompose_bisimple(b: BiVector) -> Tuple[BiVector, BiVector]:
	b2 = b.squared()
	s = b2.study_conjugate() / (b2.study_norm() * 2)
	return (1/2 + s).bivector_product(b), (1/2 - s).bivector_product(b)


mv.motor_translator = SubspaceDispatch("""Translator part of motor""")
@mv.motor_translator.register(lambda m: m.inside.rotor())
def rotor_translator(r) -> "Translator":
	"""Does not have any pure translator; return the identity"""
	# FIXME: (m>>o)/o).motor_square_root() for some given origin instead?
	return r.context.multivector.scalar()   # FIXME: broadcasting?
@mv.motor_translator.register(lambda m: m.inside.motor())
def motor_translator(m) -> "Translator":
	"""Should only hit this with degenerate components present"""
	op = m.context.operator.euclidian_factorization(m.subspace)
	return 1 + op(m, m)


mv.motor_rotor = SubspaceDispatch("""Rotor part of motor""")
@mv.motor_rotor.register(lambda m: m.inside.rotor())
def rotor_rotor(r) -> "Translator":
	"""It is a pure rotor"""
	return r
@mv.motor_rotor.register(lambda m: m.inside.motor())
def motor_rotor(m) -> "Translator":
	""""""
	return m.nondegenerate()

mv.motor_split = SubspaceDispatch("""Split motor wrt a given origin""")
@mv.motor_split.register(lambda m, o: m.inside.motor() and o.inside.vector())
def motor_split(m: Motor, o: Vector) -> Tuple[Motor, Motor]:
	o = o.dual()
	# construct shortest trajectory from o to m >> o
	t = ((m >> o) / o).motor_square_root()
	return t, ~t * m