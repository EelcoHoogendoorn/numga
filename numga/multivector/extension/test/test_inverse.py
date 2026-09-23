from traceback import print_tb

import numpy as np
import pytest

from numga.backend.numpy.context import NumpyContext
from numga.algebra.algebra import Algebra
import numpy.testing as npt

from numga.multivector.test.util import *
import contexttimer

@pytest.mark.parametrize(
	'descr', [
		(1, 0, 0), (0, 1, 0),
		(2, 0, 0), (1, 1, 0), (0, 2, 0),
		(3, 0, 0), (2, 1, 0), (1, 2, 0),
		(4, 0, 0), (3, 1, 0), (2, 2, 0),
		(5, 0, 0), (4, 1, 0), (3, 2, 0)
	],
	# 'descr', [(4, 1, 0)],
)
def test_inverse_exhaustive(descr):
	"""Test general inversion cases for all multivector grade combos in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 10
	print()
	print(descr)

	for count, grades in enumerate(all_grade_combinations(ga.algebra)):
		V = ga.subspace.from_grades(list(grades))
		print()
		x = random_subspace(ga, V, (N,))
		i = x.inverse()
		print()
		print('zero grades: ', np.unique(i.subspace.grades()[np.all(np.abs(i.values) < 1e-6, axis=0)]))
		check_inverse(x, i, atol=1e-11)
		# print()
		print('S',V.simplicity,'G', list(grades), list(np.unique(i.subspace.grades())))
		print(count)
		# print()

@pytest.mark.parametrize(
	# 'descr', [
	# 	(1, 0, 0), (0, 1, 0),
	# 	(2, 0, 0), (1, 1, 0), #(0, 2, 0),
	# 	(3, 0, 0), (2, 1, 0), #(1, 2, 0),
	# 	(4, 0, 0), (3, 1, 0), #(2, 2, 0),
	# 	(5, 0, 0), (4, 1, 0), #(3, 2, 0)
	# ],
	'descr', [(7, 0, 0)],
)

def test_inverse_13_shirokov(descr):
	"""trireflection can be inverted without zeros"""
	np.random.seed(0)  # fix seed to prevent chasing ever changing outliers
	from numga.backend.numpy.operator import NumpySparseOperator
	ga = NumpyContext(Algebra.from_pqr(6,0,0))#, otype=NumpySparseOperator)

	N = 1000
	print()
	print(descr)
	import time
	for grades in [[1,3]]:
		V = ga.subspace.from_grades([1, 3])
		print()
		x = random_subspace(ga, V, (N,))
		with contexttimer.Timer() as t:
			q = x.symmetric_reverse_product()
			qq = q.symmetric_scalar_negation_product()
			print(qq.subspace)
			return
			i = q.inverse_shirokov()
			ii = x.reverse_product(i)
		print('indirect inverse', t.elapsed)

		try:
			_ = q.inverse()
			print('rec fine')
		except:
			print('rec fail')

		print(q.subspace)
		print()
		print(i.subspace)
		print(i[0])
		print(ii.subspace)
		# print(ii)
		# print(x*ii)
		with contexttimer.Timer() as t:
			i = x.inverse_shirokov()
		print('direct inverse', t.elapsed)

		print(x*i)


@pytest.mark.parametrize(
	# 'descr', [
	# 	(1, 0, 0), (0, 1, 0),
	# 	(2, 0, 0), (1, 1, 0), #(0, 2, 0),
	# 	(3, 0, 0), (2, 1, 0), #(1, 2, 0),
	# 	(4, 0, 0), (3, 1, 0), #(2, 2, 0),
	# 	(5, 0, 0), (4, 1, 0), #(3, 2, 0)
	# ],
	'descr', [(7, 0, 0)],
)
def test_inverse_exhaustive_shirokov(descr):
	"""Test general inversion cases for all multivector grade combos in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 1
	print()
	print(descr)

	with contexttimer.Timer() as t:
		for grades in all_grade_combinations(ga.algebra):
			# if len(grades) > 2:
			# 	return
			V = ga.subspace.from_grades(list(grades))
			V = ga.subspace.full()
			# if not V.subspace.inside.self_reverse():
			# 	continue
			# if len(grades) > 3:
			# 	continue
			# print()
			x = random_subspace(ga, V, (N,))
			#
			# q = x.symmetric_reverse_product()
			# q = q.symmetric_scalar_negation_product()
			# print(grades)
			# print(np.unique(q.subspace.grades()))
			# print(np.unique(q.subspace.inverse_subspace_estimate().grades()))

			# continue

			if False:
				q = x.symmetric_reverse_product()
				q = q.squared()
				# qq = qq.symmetric_conjugate_product()
				f = np.all(np.abs(q.values) < 1e-6, axis=0)
				z = np.unique(q.subspace.grades()[f])
				o = np.unique(q.subspace.grades()[~f])
				# print(grades)
				print('sr grades', o)
				# continue

			# try:
			# 	q = x.inverse()
			# 	print('rec fine')
			# except:
			# 	pass

			# q = x.symmetric_reverse_product()
			# qi = q.inverse_la()
			# i = ~x * qi
			i = x.inverse_la()

			# f = np.all(np.abs(i.values) < 1e-6, axis=0)
			# z = np.unique(i.subspace.grades()[f])
			# o = np.unique(i.subspace.grades()[~f])
			# print('input grades', list(grades))
			# print('output grades', o)
			# print('zero grades: ', z)
			# print(i*x)
			check_inverse(x, i, atol=1e-6)
			# if grades[0]==5:
			# 	return
			# print()
	print(t.elapsed)


@pytest.mark.parametrize(
	# 'descr', [
	# 	(1, 0, 0), (0, 1, 0),
	# 	(2, 0, 0), (1, 1, 0), #(0, 2, 0),
	# 	(3, 0, 0), (2, 1, 0), #(1, 2, 0),
	# 	(4, 0, 0), (3, 1, 0), #(2, 2, 0),
	# 	(5, 0, 0), (4, 1, 0), #(3, 2, 0)
	# ],
	'descr', [(5, 0, 0)],
)
def test_inverse_accuracy(descr):
	"""Test general inversion cases for all multivector grade combos in dimension < 6"""
	np.random.seed(0)   # fix seed to prevent chasing ever changing outliers
	ga = NumpyContext(Algebra.from_pqr(*descr))

	N = 10
	print()
	print(descr)

	with contexttimer.Timer() as t:
		for grades in all_grade_combinations(ga.algebra):
			V = ga.subspace.from_grades(list(grades))
			x = random_subspace(ga, V, (N,))

			print(grades)

			ii = x.inverse()
			ei = x * ii - 1
			print('rec', (ei.values**2).mean())

			# il = x.inverse_la()
			# el = x * il - 1
			# print('la', (el.values**2).mean())
	print(t.elapsed)

	with contexttimer.Timer() as t:
		for grades in all_grade_combinations(ga.algebra):
			V = ga.subspace.from_grades(list(grades))
			x = random_subspace(ga, V, (N,))

			print(grades)

			# ii = x.inverse()
			# ei = x * ii - 1
			# print('rec', (ei.values**2).mean())

			il = x.inverse_la()
			el = x * il - 1
			print('la', (el.values**2).mean())
	print(t.elapsed)




def test_inverse_simplicifation_failure_bivec():
	"""succssive involute products sometimes fail to simplify fully.
	this results in extra recursion and poorer high dim generalization,
	and also sometimes dummy zero output grades
	"""
	ga = NumpyContext(Algebra.from_pqr(3,0,1))
	V = ga.subspace.from_grades([2])
	assert V.simplicity == 2    # in reality its two but we lack the symbolic logic to see it
	x = random_subspace(ga, V, (1,))
	# can still invert correctly in 3 steps tho
	check_inverse(x, x.inverse())

	y = x.symmetric_reverse_product()
	z = y.symmetric_scalar_negation_product()
	print(y)
	print(z)
	print(x)
	# assert z.subspace == ga.subspace.from_grades([0])
	# assert np.allclose(z.select[5].values, 0)


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
	# x = random_motor(ga, (1,))
	i = x.inverse_la()
	print(i)    # note this particular 4-component multivector has an 8-component inverse
	check_inverse(x, i)

	i = x.inverse()
	print(i)
	check_inverse(x, i)


def test_inverse_4d():
	"""test 4d case
	"""
	ga = NumpyContext('x+y+z+a+')
	mv = ga.multivector
	x = mv.xy + mv.za
	y=(x.reverse_product(x))
	print(y.symmetric_reverse_product())
	print(y.symmetric_scalar_negation_product())
	print(y.squared())

	i = x.inverse()
	print(i)    # note this particular 4-component multivector has an 8-component inverse
	check_inverse(x, i)
