import pytest
from numpy import testing as npt

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext
from numga.multivector.test.util import random_motor


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), #(0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 0, 2), #(2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), #(2, 2, 1),
])
def test_invariant_decomposition(descr):
	print()
	print(descr)
	algebra = Algebra.from_pqr(*descr)
	ga = NumpyContext(algebra)
	from numga.multivector import extension_fancy
	for i in range(10):
		b = random_motor(ga).select[2]

		print('decompose')
		l, r = b.decompose_invariant()
		# if ga.algebra.pseudo_scalar_squared == 0:
		# 	assert r.subspace.is_degenerate
		q = l.inner(r)
		npt.assert_allclose(q.values, 0, atol=1e-9)
		q = l.squared() # test they square to a scalar
		npt.assert_allclose(q.values[1:], 0, atol=1e-9)
		q = r.squared()
		npt.assert_allclose(q.values[1:], 0, atol=1e-9)

		npt.assert_allclose(b.values, (l+r).values, atol=1e-9)

		L = l.exp()
		R = r.exp()
		# test that component bivectors commute under exponentiation
		npt.assert_allclose((L * R).values, (R * L).values)
