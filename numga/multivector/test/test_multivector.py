import numpy as np
import pytest
import numpy.testing as npt
from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext
from numga.multivector.test.util import *


def test_basic():
	ga = NumpyContext('x+y+w0')
	V = ga.subspace.vector()
	V = ga.subspace.from_bit_str('100,010,001')
	v = ga.multivector.vector(values=np.random.normal(size=len(V)))
	print(v)
	print(+v)
	print(-v)
	print(v - 1)
	print(1 - v)
	print(v / 1)
	print(1 / v)
	print((1 / v) * v)


@pytest.mark.parametrize('descr', [
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0)
])
def test_5d_rotor_quality(descr):
	"""Test that R~R=1 suffices for rotor grade preservation in 5d"""
	ga = NumpyContext(Algebra.from_pqr(*descr))
	N = 100
	v = random_subspace(ga, ga.subspace.vector(), (N,))
	r = random_non_motor(ga, (N,))
	r = r.normalized()  # enforce R~R=1
	q = r.full_sandwich(v)  # use full sandwich which will yield grade-5 elements
	assert_close(q.select[5], 0)

