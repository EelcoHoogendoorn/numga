import pytest
import numpy.testing as npt

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext as Context
from numga.multivector.numerical_normalize import normalize_motor, inverse_sandwich

from numga.multivector.test.util import *


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
	(6, 0, 0), (5, 0, 1), (5, 1, 0), (4, 1, 1), (4, 2, 0),
	(7, 0, 0), (6, 0, 1), (6, 1, 0), (5, 1, 1), (5, 2, 0),
])
def test_numerical_inverse_sandwich(descr):
	"""Test numerical inverse sandwich

	Note that this works just fine in dimensions >= 6
	Regardless, it no longer works as a basis for motor normalisation in those high dimensional spaces
	"""
	algebra = Algebra.from_pqr(*descr)
	context = Context(algebra)
	e = random_non_motor(context, shape=(10, ))

	# study numbers constructed as the symmetric reverse product of elements of the even grade subalgebra,
	# should admit an inverse sandwich solve
	s = e.symmetric_reverse_product()
	isr = inverse_sandwich(s, n_iter=10)
	assert_close(isr >> s, 1)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), #(1, 1, 0),# (0, 1, 1),
	(3, 0, 0), (2, 0, 1), #(2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), #(3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), #(4, 1, 0), (3, 1, 1), (3, 2, 0),
	# (6, 0, 0), (5, 0, 1), #(5, 1, 0), (4, 1, 1), (4, 2, 0),
	# (7, 0, 0), (6, 0, 1), #(6, 1, 0), (5, 1, 1), (5, 2, 0),
])
def test_normalize_sum(descr):
	"""Test normalization of a sum of motors"""
	algebra = Algebra.from_pqr(*descr)
	context = Context(algebra)
	e = random_motor(context, (2, 10)).sum(0)

	if context.algebra.description.n_dimensions < 5:
		# grade 1 preservation comes for free in low dimensions
		v0, v1 = motor_properties(e)
		npt.assert_allclose(v1, 0, atol=1e-9)

	m = normalize_motor(e, scalar_norm=True, inner=10, outer=3)
	# for this special sum-of-motors case, renormalization passes all tests up to dim < 8 !
	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	npt.assert_allclose(v1, 0, atol=1e-9)


@pytest.mark.parametrize('descr', [
	(2, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1),
	(3, 0, 0), (2, 0, 1), (2, 1, 0), (1, 1, 1),
	(4, 0, 0), (3, 0, 1), (3, 1, 0), (2, 1, 1), (2, 2, 0),
	(5, 0, 0), (4, 0, 1), (4, 1, 0), (3, 1, 1), (3, 2, 0),
	(6, 0, 0), (5, 0, 1), (5, 1, 0), (3, 1, 1), (3, 2, 0),
])
def test_normalize_random(descr):
	"""Test normalization of a pure random motor"""
	algebra = Algebra.from_pqr(*descr)
	context = Context(algebra)

	e = random_non_motor(context, shape=(10,), n=4)
	m = normalize_motor(e, inner=10, outer=1)

	v0, v1 = motor_properties(m)
	npt.assert_allclose(v0, 0, atol=1e-9)
	if context.algebra.description.n_dimensions < 6:
		# our renormalization procedure fails to grade-preserve 1 vecs
		# starting from pure random numbers in dimensions >= 6
		npt.assert_allclose(v1, 0, atol=1e-9)


def test_normalize_plot():
	"""Plot the convergence behavior of a grid of random motors being normalized in 6d"""
	algebra = Algebra.from_pqr(6, 0, 0)
	context = Context(algebra)

	N = 100
	if True:
		m = random_subspace(context, context.subspace.motor(), (2,))
	else:
		m = random_motor(context, (2,))
	m = m / m.symmetric_reverse_product().select[0].sqrt() * 2

	R = 2
	r = np.linspace(-R, +R, N, endpoint=True)
	m = m[0] * r[:, None] + m[1] * r[None, :]

	n = normalize_motor(m, inner=6, outer=1, scalar_norm=False)

	v0, v1 = motor_violations(n)
	v0 = np.linalg.norm(v0.values, axis=-1)
	v1 = np.linalg.norm(v1.values, axis=-1)

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 2)
	p0 = ax[0].imshow(np.log(v0))
	plt.colorbar(p0, ax=ax[0])
	p1 = ax[1].imshow(np.log(v1))
	plt.colorbar(p1, ax=ax[1])

	a = np.linspace(0, np.pi*2, 100)
	ax[0].plot(np.cos(a) * N / 2 / R + N/2, np.sin(a) * N / 2 / R + N/2, c='k')
	ax[1].plot(np.cos(a) * N / 2 / R + N/2, np.sin(a) * N / 2 / R + N/2, c='k')
	plt.show()
