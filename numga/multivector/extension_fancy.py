from typing import Tuple

from numga.dynamic_dispatch import SubspaceDispatch
from numga.multivector.types import Scalar, BiVector, Motor, Study
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
@mv.decompose_invariant.register(lambda s: s.inside.bivector() and s.squared().inside.study())
def bivector_decompose_bisimple(b: BiVector) -> Tuple[BiVector, BiVector]:
	b2 = b.squared()
	s = b2.study_conjugate() / (b2.study_norm() * 2)
	return (1/2 + s).bivector_product(b), (1/2 - s).bivector_product(b)
