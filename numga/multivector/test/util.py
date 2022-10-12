import numpy as np


def random_subspace(context, subspace):
	r = np.random.normal(size=len(subspace))
	return context.multivector(subspace=subspace, values=r)


def random_normal_vector(context):
	return random_subspace(context, context.subspace.vector())


def random_reflection(context):
	while True:
		r = random_normal_vector(context)
		n = r.norm()
		if all(n.values > 1e-3):
			return r / n


def random_n_reflection(context, n):
	return context.multivector.scalar() if n == 0 else random_n_reflection(context, n-1) * random_reflection(context)


def random_motor(context, negative_scalar=True):
	m = random_n_reflection(context, (context.algebra.n_dimensions // 2) * 2)

	m = m * np.sign(m.values[..., 0])

	if negative_scalar and context.algebra.description.pqr[1] == 0:
		# negative scalars are just fine without hyperbolic space messing it up
		m = m * m

	return m
