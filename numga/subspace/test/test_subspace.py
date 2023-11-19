"""Cant construct a subspace without an algebra
perhaps more fruitful to keep subspace tests in algebra module
"""


from numga.algebra.algebra import Algebra
from numga.multivector.test.util import *


def test_basic():
	algebra = Algebra.from_pqr(2, 0, 0)

	# complex numbers are isomorphic to the even subalgebra of 200
	C = algebra.subspace.even_grade()
	assert C.is_subalgebra
	assert not C.in_self_reverse
	assert C in C
	assert C == C

	assert C.restrict.vector().equals.empty()
	assert C.select.vector().equals.vector()
	print()
	print(C)
	print(C.bit_str)


def test_mimimal_subalgebra():
	algebra = Algebra.from_pqr(5, 0, 0)

	assert(algebra.subspace.scalar().union(algebra.subspace.ab).is_subalgebra)
	for grades in all_grade_combinations(algebra):
		S = algebra.subspace.from_grades(grades)
		print()
		print(grades)
		# print(S)
		s = S.minimal_subalgebra
		assert s.is_subalgebra
		print(s.pretty_str)
		print(list(np.unique(s.grades())))
		s = S.minimal_exponential
		print(list(np.unique(s.grades())))


def test_stuff():
	ga = Algebra.from_pqr(3, 0, 1)

	assert ga.subspace.rotor() != ga.subspace.motor()
	assert ga.subspace.rotor().product(ga.subspace.translator()) == ga.subspace.motor()
