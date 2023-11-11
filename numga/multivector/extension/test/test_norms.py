
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
])
def test_study(descr):
	"""test basic properties of study numbers"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	# study numbers constructed from products of the even subalgebra
	# for general randomly sampled study numbers, square roots might not exist in GAs over the real numbers
	# as an example, in d<4, we have degenerate scalar study numbers; and negative scalars wont fly
	# this is fine though, since we are only using study number functionality for motor normalization
	m = random_non_motor(context, shape=(10, ))
	s = m.symmetric_reverse_product()
	assert s.subspace.is_study

	print('inverse')
	Is = s.inverse()
	r = Is * s - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)

	print('square root')
	rs = s.square_root()
	r = rs * rs - s
	npt.assert_allclose(r.values, 0, atol=1e-9)
	zero = s * 0
	rs = zero.square_root()
	npt.assert_allclose(rs.values, 0, atol=1e-9)

	print('inverse square root')
	Irs = s.inverse().square_root()

	Irs2 = s.square_root().inverse()
	npt.assert_allclose(Irs2.values, Irs.values, atol=1e-6)

	r = s * Irs * Irs - 1
	npt.assert_allclose(r.values, 0, atol=1e-9)


def test_motor_normalize_6d(descr=[6,0,0]):
	"""Note that in 6d we do not have a known motor normalization algorithm"""
	algebra = Algebra.from_pqr(*descr)
	context = NumpyContext(algebra)

	m = random_non_motor(context, shape=(10, ))
	s = m.symmetric_reverse_product()
	assert not s.subspace.is_study

	with pytest.raises(Exception):
		m.normalized()
