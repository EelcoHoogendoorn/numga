"""Cant construct a subspace without an algebra
perhaps more fruitful to keep subspace tests in algebra module
"""


from numga.algebra.algebra import Algebra

def test_basic():
	algebra = Algebra.from_pqr(2, 0, 0)

	# complex numbers are isomorphic to the even subalgebra of 200
	C = algebra.subspace.even_grade()
	assert C.is_subalgebra
	assert not C.in_self_reverse
	assert C in C
	assert C == C
	print()
	print(C)
	print(C.bit_str)
