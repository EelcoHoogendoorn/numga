import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1),# (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), #(2, 1, 0),# (1, 1, 1),
	(4, 0, 0), (3, 0, 1), #(3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), #(4, 1, 0), #(3, 1, 1), (3, 2, 0),
])
def test_study_sqrt(descr):
	"""Test square roots of generalized study numbers
	This also works with negative signatures,
	except we are lacking a way to generate random multivectors with guaranteed real roots
	"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	print()
	print(descr)
	print()

	for grades in all_grade_combinations(algebra):
		s = context.subspace.from_grades(grades)
		x = random_subspace(context, s, (10,))
		# this skips some study numbers we could take roots of
		# but this is an easy way to limit ourselves to study numbers that do have real roots
		x = x.symmetric_reverse_product()
		if x.subspace.is_study:
			print(x.subspace)
			sqrt = x.square_root()
			# print(sqrt)
			r = sqrt.squared() - x
			npt.assert_allclose(r.values, 0, atol=1e-9)
			print('PASS')
