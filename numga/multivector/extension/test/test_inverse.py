import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *


@pytest.mark.parametrize(
	'descr', [
		(1, 0, 0), (0, 1, 0),
		(2, 0, 0), (1, 1, 0), (0, 2, 0),
		(3, 0, 0), (2, 1, 0), (1, 2, 0),
		(4, 0, 0), (3, 1, 0), (2, 2, 0),
		(5, 0, 0), (4, 1, 0), (3, 2, 0)
	],
	# 'descr', [(4, 0, 0)],
)
def test_inverse_exhaustive(descr):
	"""Test general inversion cases for all multivector grade combos in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 10
	print()
	print(descr)

	for grades in all_grade_combinations(ga.algebra):
		V = ga.subspace.from_grades(list(grades))
		print()
		x = random_subspace(ga, V, (N,))
		i = x.inverse()
		print()
		print('zero grades: ', np.unique(i.subspace.grades()[np.all(np.abs(i.values) < 1e-6, axis=0)]))
		check_inverse(x, i, atol=1e-11)
		# print()
		print('S',V.simplicity,'G', list(grades), list(np.unique(i.subspace.grades())))
		# print()


def test_inverse_simplicifation_failure():
	"""succssive involute products sometimes fail to simplify fully.
	this results in extra recursion and poorer high dim generalization,
	and also sometimes dummy zero output grades
	"""
	ga = NumpyContext(Algebra.from_pqr(5,0,0))
	V = ga.subspace.from_grades([1,2,5])
	assert V.simplicity == 3    # in reality its two but we lack the symbolic logic to see it
	x = random_subspace(ga, V, (1,))
	# can still invert correctly in 3 steps tho
	check_inverse(x, x.inverse())

	y = x.symmetric_reverse_product()
	z = y.symmetric_pseudoscalar_negation_product()
	assert z.subspace == ga.subspace.from_grades([0, 5])
	assert np.allclose(z.select[5].values, 0)

	# second-order optimized hitzer term does reduce to scalar
	op = ga.operator.inverse_factor_completed_alt(V)
	assert op.output.equals.scalar()

	# V = ga.subspace.from_grades([2])
	# assert V.simplicity == 2    # need two steps; but can do without the extra zeros
	# x = random_subspace(ga, V, (1,))
	# i = x.inverse()
	# assert i.subspace == ga.subspace.from_grades([2, 4])
	# check_inverse(x, i)
	# assert np.allclose(i.select[4].values, 0)
	# # second-order optimized hitzer term does reduce to scalar
	# op = ga.operator.inverse_factor_completed_alt(V)
	# assert op.output.equals.scalar()


def test_inverse_6d():
	"""test recursive inverse handles some 6d example
	"""
	ga = NumpyContext('x+y+z+a+b+c+')
	mv = ga.multivector
	x = 1 + mv.xy + mv.ab + mv.xyzabc
	i = x.inverse_la()
	print(i)    # note this particular 4-component multivector has an 8-component inverse
	check_inverse(x, i)

	i = x.inverse()
	print(i)
	check_inverse(x, i)
