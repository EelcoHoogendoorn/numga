import numpy as np
import pytest

from numga.algebra.algebra import Algebra


def test_dual():
	print()
	dual = Algebra('x0')
	# print(dual.subspace.x)
	full = dual.subspace.full()

	print(full)
	print(dual.subspace.empty())
	print(dual.subspace.scalar())

	c, s = dual.product(full.blades, full.blades)
	print(c)
	print(s)


def test_complex():
	complex = Algebra('x+y+')

	full = complex.subspace.full()
	even = complex.subspace.even_grade()
	assert even.is_reverse_simple
	c, s = complex.product(even.blades, even.blades)
	print(c)
	print(s)

	print()
	print(complex.subspace.empty())
	print(complex.subspace.scalar())


def test_quat():
	r3 = Algebra('x+y+z+')
	# print(r3.subspace.yx)
	quat, full = r3.subspace.even_grade(), r3.subspace.full()
	print()
	assert quat.is_subalgebra
	assert quat.is_reverse_simple
	print(full.bit_blades())
	print(full.blades)
	print(quat.bit_blades())
	print(quat.blades)
	c, s = r3.product(quat.blades, quat.blades)
	print(c)
	print(s)


def test_2d_pga():
	ga = Algebra('w0x+y+')
	full = ga.subspace.full()
	even_grade = ga.subspace.even_grade()
	assert even_grade in full
	assert even_grade.is_subspace(full)

	scalar = ga.subspace.scalar()
	vector = ga.subspace.vector()
	bivector = ga.subspace.bivector()
	rotation_vector = bivector.nondegenerate()
	translation_vector = bivector.degenerate()

	assert scalar.is_simple
	assert vector.is_simple
	assert bivector.is_simple
	assert rotation_vector.is_simple
	assert not rotation_vector.is_degenerate
	assert len(rotation_vector) == 1
	assert translation_vector.is_simple
	assert translation_vector.is_degenerate
	assert len(translation_vector) == 2

	assert not even_grade.is_simple
	assert even_grade.is_reverse_simple
	assert even_grade.is_subalgebra

	print(full.bit_blades())

	print(even_grade.bit_blades())

	S = ga.operator.sandwich(full, vector)
	assert S.output == vector


def test_3d_pga():
	print()

	ga = Algebra('x+y+z+w0')
	full = ga.subspace.full()
	even_grade = ga.subspace.even_grade()
	quad_reflection = ga.subspace.k_reflection(4)
	assert quad_reflection == even_grade
	assert even_grade in full
	assert even_grade.is_subspace(full)
	vector = ga.subspace.vector()
	bivector = ga.subspace.bivector()
	trivector = ga.subspace.trivector()

	quaternion = even_grade.nondegenerate()
	rotation_vector = bivector.nondegenerate()
	translation_vector = bivector.degenerate()
	# FIXME: translation motor would be translation_vector + scalar
	#  should we call it rotor/translator? and is there a natural way to construct it?

	assert vector.is_simple
	assert not bivector.is_simple
	assert bivector.is_bisimple
	assert rotation_vector.is_simple
	assert not rotation_vector.is_degenerate
	assert len(rotation_vector) == 3
	assert translation_vector.is_simple
	assert translation_vector.is_degenerate
	assert len(translation_vector) == 3
	assert trivector.is_simple
	assert not even_grade.is_simple
	assert not even_grade.is_reverse_simple
	assert even_grade.is_reverse_bisimple
	assert not quaternion.is_simple
	assert quaternion.is_reverse_simple

	print(full.bit_blades())

	assert even_grade.is_subalgebra
	print(even_grade.bit_blades())
	print(quaternion.bit_blades())



def mod_4_logic(all_grades, input_grades):
	all_grades, input_grades = np.asarray(all_grades), np.asarray(input_grades)
	return np.where(np.any(np.equal.outer(all_grades % 4, input_grades % 4), axis=1))[0]


def test_sandwich():
	"""Test output space of the sandwich product
	Note that it is only grade-preserving for k-vectors,
	for which no k-vectors of grade 4 lower or higher exist

	For general multivectors, the output should have all input grades, mod 4
	"""
	ga = Algebra('a+b+c+d+w+x+y+z+')
	# ga = Algebra('w+x+y+z+')
	input_grades = [1]
	grades = np.arange(ga.n_grades)     # array of all grades in the algebra
	output_grades = mod_4_logic(grades, input_grades)

	print(output_grades)
	# kv = ga.subspace.
	kv = ga.subspace.from_grades(input_grades)
	S = ga.operator.full_sandwich(ga.subspace.even_grade(), kv)
	print(np.unique(ga.grade(S.output.blades)))
	assert S.output == ga.subspace.from_grades(output_grades)


def test_norm_squared():
	"""Test output space of the squared norm for various multivectors, defined as (x ~x)
	[0] -> [0]
	[1] -> [0]
	[2] -> [0, 4]
	[3] -> [0, 4]
	[4] -> [0, 4, 8]
	[5] -> [0, 4]
	[6] -> [0, 4]
	[7] -> [0]
	[8] -> [0]
	[0, 2, 4, 6, 8] -> [0, 4, 8]

	Note; the full grade multivector maps to [0, 1, 4, 5, 8, 9, ...],
	since the product X~X is self-reverse
	"""
	# ga = Algebra('a+b+c+d+e+f+w+x+y+z+')
	ga = Algebra('a+b+c+d+e+w+x+y+z+')
	# ga = Algebra('a+b+w+x+y+z+')
	grade = 1
	if grade < 0:   # map negative indexed grades to positive equivalent
		grade = ga.n_grades + grade
	antigrade = ga.n_dimensions - grade   # grade that the dual would have
	grades = np.arange(ga.n_grades)     # array of all grades in the algebra
	output_grades = grades[::4][:min(grade, antigrade) // 2 + 1]
	print(output_grades)
	kv = ga.subspace.k_vector(grade)
	# kv = ga.subspace.from_grades([0, 4, 8])
	# construct symbolically simplified operator for (x * ~x)
	S = ga.operator.reverse_product(kv, kv).symmetry((0, 1), +1)
	print(np.unique(ga.grade(S.output.blades)))
	assert S.output == ga.subspace.from_grades(output_grades)


@pytest.mark.skip('takes too long')
def test_large():
	# so far ok few sec (5) runtime up to dim=13. good fraction of a minute 36.6 at dim=14; 14s now.
	# not sure there is much left to optimize; other than lazy eval
	ga = Algebra('w0x+y+a+b+c+d+e+g+h+k+l+z+t+')
	even_grade = ga.subspace.even_grade()

	from time import time
	t = time()
	cn, sn = ga.product(even_grade.blades, even_grade.blades)
	print('time: ', time() - t)

	print('mbs: ', ga.n_dimensions, cn.shape, cn.size / 1024 / 1024)
	print('cayley blades')
	print(even_grade.bit_blades())
