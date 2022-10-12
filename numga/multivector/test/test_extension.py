
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import random_n_reflection, random_motor


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
])
def test_study(descr):
	"""test basic properties of study numbers"""
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)

	S = ga.subspace.study()
	b = random_n_reflection(ga, 2)
	s = b.symmetric_reverse_product()
	assert s.subspace.inside(S)
	print(s)
	zero = s * 0

	print('norm')
	print(s.study_norm())

	print('square root')
	rs = s.square_root()
	print(rs)
	r = rs * rs - s
	npt.assert_allclose(r.values, 0, atol=1e-9)
	rs = zero.square_root()
	print(rs)
	npt.assert_allclose(rs.values, 0, atol=1e-9)

	print('inverse')
	Is = s.inverse()
	print(Is)
	r = Is * s - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)

	print('inverse square root')
	Irs = s.inverse().square_root()

	npt.assert_allclose(s.inverse_square_root().values, Irs.values, atol=1e-9)

	print(Irs)
	r = s * Irs * Irs - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1), (1, 0, 2), (1, 2, 0),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1),
])
def test_bisect(descr):
	"""test roundtrip qualities of bisection based log and exp"""
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)

	for i in range(10):
		m = random_motor(ga)

		# check exact inverse props
		r = m.motor_log().exp()
		npt.assert_allclose(m.values, r.values, atol=1e-9)

