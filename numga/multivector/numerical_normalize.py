import numpy as np


def solve_newton_direct(J, r):
	d = np.linalg.solve(np.einsum('...ji,...jk -> ...ik', J, J), r)
	return np.einsum('...ij,...j -> ...i', J, d)


def inverse_sandwich(s, n_iter=6):
	"""Solve the quadratic sandwich equation
		x s ~x = 1
		for the unknown x

	Typically, s here s a Study number encountered in motor normalisation
	if this x is self-reverse, we could call x an inverse square root of s
	"""
	context = s.context

	S = s.subspace
	sandwich = context.operator.full_sandwich(S, S)
	assert sandwich.output == S
	# bind the center argument
	quadratic = sandwich.partial({1: s})
	# multiplicative unit; scalar one, zeros for the rest
	rhs = context.multivector(subspace=S)

	def newton_step(x):
		"""Iteratively solve the quadratic equations prod(x, s, x) = 1,
		for the inverse square root x,
		using a newton iteration:
		https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
		"""
		J = quadratic.partial({0: x})  # grad of the above quadratic wrt s
		lhs = J.partial({0: x})  # left hand side of above equation
		# find gauss-newton update step
		d = solve_newton_direct(J.kernel, lhs.kernel - rhs.values)
		# and apply it
		return x + context.multivector(S, -d / 2)

	x = rhs
	for i in range(n_iter):
		x = newton_step(x)
	return x


def normalize_motor(m, inner=6, outer=2, scalar_norm=True):
	"""Normalization of a motor,
	via numerical solution of the inverse-sandwich of the norm-squared of the motor

	In dimensions < 6, convergence should be guaranteed, in higher dimension, convergence is more erratic

	This procedure only guarantees grade-0 preservation
	but grade-1 vectors need not be grade preserving in dims >= 6
	"""
	if scalar_norm:
		# take out scalar norm first; can speed up convergence for x far from unity
		m = m * m.symmetric_reverse_product().select[0].inverse_square_root()
		# somehow this scaleup improves attraction basin?
		#m = m * 2

	for i in range(outer):
		s = inverse_sandwich(m.symmetric_reverse_product(), n_iter=inner)
		m = s * m
	return m
