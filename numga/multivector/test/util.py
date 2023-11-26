import numpy as np


def random_subspace(context, subspace, shape=()):
	r = np.random.normal(size=shape + (len(subspace),))
	return context.multivector(subspace=subspace, values=r)


def random_normal_vector(context, shape=()):
	return random_subspace(context, context.subspace.vector(), shape)


def random_nonnegative_normal_vector(context, shape=()):
	"""Random vector from normal distribution with a norm >= 0"""
	r = random_normal_vector(context, shape)
	if context.algebra.description.pqr[1] > 0:
		n = r.squared()
		condition = n.values < 0
		r = r.copy(values=np.where(condition, 1 / r.values, r.values))
	return r


def random_positive_normal_vector(context, shape=(), margin=1e-3):
	"""Random vector from normal distribution with a norm > 0"""
	r = random_nonnegative_normal_vector(context, shape)
	while True:
		n = r.squared()
		condition = n.values < margin
		if not condition.any():
			break
		r = r.copy(values=np.where(condition, random_nonnegative_normal_vector(context, shape).values, r.values))
	return r


def random_reflection(context, shape=()):
	r = random_nonnegative_normal_vector(context, shape)
	return r * r.squared().inverse_square_root()


def random_bireflection(context, shape=()):
	l, r = random_reflection(context, shape), random_reflection(context, shape)
	b = l * r
	b = b * np.sign(b.values[..., 0])
	return b


def random_n_reflection(context, n, shape=()):
	return context.multivector.scalar() if n == 0 else random_n_reflection(context, n-1, shape) * random_reflection(context, shape)


def random_2n_reflection(context, n, shape=()):
	return context.multivector.scalar() if n == 0 else random_2n_reflection(context, n-1, shape) * random_bireflection(context, shape)


def random_motor(context, shape=()):
	m = random_2n_reflection(context, context.algebra.n_dimensions // 2, shape)
	m = m * np.sign(m.values[..., 0])
	return m


def random_non_motor(context, shape=(), scale=1e-2, n=4):
	"""not-quite-motor, not too far from unity"""
	m = random_subspace(context, context.subspace.even_grade(), shape=shape) * scale + 1
	for i in range(n):
		m = m.squared()
	return m


def motor_violations(m):
	context = m.context

	v0 = m.symmetric_reverse_product() - 1

	V = context.subspace.vector()
	v = random_subspace(context, V, m.shape)
	s = context.operator.full_sandwich(m.subspace, v.subspace)
	r = s(m, v, m)
	v1 = r.select_subspace(r.subspace.difference(V))
	return v0, v1


def motor_properties(m):
	v0, v1 = motor_violations(m)
	v0 = np.linalg.norm(v0.values, axis=-1)
	v1 = np.linalg.norm(v1.values, axis=-1)
	return v0, v1
	# return np.allclose(v0, 0, atol=1e-6), np.allclose(v1, 0, atol=1e-6)


def all_grade_combinations(algebra):
	all_grades = np.arange(algebra.n_grades)
	import itertools
	for r in all_grades:
		for grades in itertools.combinations(all_grades, r+1):
			yield list(grades)


def check_inverse(x, i, atol=1e-9):
	assert_close(x*i, 1, atol)
	assert_close(i*x, 1, atol)


def assert_close(a, b, atol=1e-9):
	assert np.allclose((a-b).values, 0, atol=atol)