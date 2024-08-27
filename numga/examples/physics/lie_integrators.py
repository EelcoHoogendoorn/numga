"""
Implementations of
https://pure.tue.nl/ws/portalfiles/portal/3801945/900594800955441.pdf
https://www.research.unipd.it/retrieve/e14fb26f-e9d2-3de1-e053-1705fe0ac030/Ortolan_PhD11.pdf

might want to add manual computed jacobians in here as well?
need to be able to compose concrete operators to do so
would be a good test case for such a rewrite
"""

import jax
import jax.numpy as jnp


def newton_solver(fn, n=10):
	"""Return callable that performs n newton steps on fn"""
	jac_fn = jax.jacfwd(fn)
	step = lambda i, x: x - jnp.linalg.solve(jac_fn(x), fn(x))
	return lambda x: jax.lax.fori_loop(0, n, step, x)


def newton_solver_wrap(func, init):
	"""wrap newton solve of a multivector function"""
	func_wrap = lambda x: func(init.copy(values=x)).values
	solver = newton_solver(func_wrap)
	return init.copy(values=solver(init.values))


# second order accurate cayley log/exp approximations
def exp2(b):
	return b.exp_quadratic()
def log2(r):
	return r.motor_log_quadratic()
# higher order accurate log/exp approximations
def exp4(b):
	return exp2(b / 2).squared()
def log4(r):
	return log2(r.motor_square_root()) * 2


def variational_lie_verlet(motor, rate, inertia, inertia_inv, dt, ext_forque):
	"""Variational Lie-Verlet"""
	energy = lambda rate: \
			inertia(rate).wedge(rate) * rate * (dt / 2)
	forque = lambda motor, rate: \
			ext_forque(motor, rate) - inertia(rate).commutator(rate)

	implicit = lambda rh: -rh + rate + \
		inertia_inv(forque(motor, rh) - energy(rh)) * (dt / 2)
	rate_half = newton_solver_wrap(implicit, init=rate)

	motor_new = motor * exp2(rate_half * dt / -4)

	rate_new = rate_half + \
	    inertia_inv(forque(motor_new, rate_half) + energy(rate_half)) * (dt / 2)

	return motor_new, rate_new


def explicit_lie_newmark(motor, rate, inertia, inertia_inv, dt, ext_forque):
	"""Explicit Lie-Newmark method"""
	impulse = lambda motor, rate: \
		(ext_forque(motor, rate) - inertia(rate).commutator(rate)) * (dt / 2)

	half_rate_step = inertia_inv(impulse(motor, rate))
	rate_half = rate + half_rate_step
	motor_new = exp2(rate_half * dt / 2) * motor
	# motor_new = (rate_half * dt / 2).exp() * motor

	implicit = lambda rn: \
		-rn + rate_half + inertia_inv(impulse(motor_new, rn))
	rate_new = newton_solver_wrap(implicit, rate_half + half_rate_step)

	return motor_new, rate_new


def explicit_lie_newmark_rev(motor, rate, inertia, inertia_inv, dt, ext_forque):
	"""My own crazy mix"""
	impulse = lambda motor, rate: \
		(ext_forque(motor, rate) - inertia(rate).commutator(rate)) * (dt / 2)

	half_rate_step = inertia_inv(impulse(motor, rate))
	implicit = lambda rh: \
		-rh + rate + inertia_inv(impulse(motor, rh))
	rate_half = newton_solver_wrap(implicit, rate + half_rate_step)

	motor_new = exp2(rate_half * dt / 2) * motor

	# double-sided implicit does not seem to work; need forward/backward cancellation
	# half_rate_step = inertia_inv(impulse(motor_new, rate_half))
	# implicit = lambda rn: -rn + rate_half + inertia_inv(impulse(motor_new, rn))
	# rate_new = newton_solver_wrap(implicit, rate_half + half_rate_step)

	half_rate_step = inertia_inv(impulse(motor_new, rate_half))
	rate_new = rate_half + half_rate_step

	return motor_new, rate_new


def new3(motor, rate, inertia, inertia_inv, dt, ext_forque):
	""""""
	# FIXME: works like garbage?
	motor_half = exp2(rate * dt / 4) * motor
	impulse = lambda motor, rate: \
		(ext_forque(motor, rate) - inertia(rate).commutator(rate)) * dt
	rate_new = rate + inertia_inv(impulse(motor, rate))

	rhs = exp4(rate * dt / 4) >> inertia(rate)
	implicit = lambda r: \
		exp4(r * dt / -4) >> inertia(r) - rhs
	rate_new = newton_solver_wrap(implicit, rate_new)

	motor_new = exp2(rate_new * dt / 4) * motor_half

	return motor_new, rate_new


def explicit_rk1(motor, rate, inertia, inertia_inv, dt, ext_forque):
	"""RK1 integration of lie state"""
	dr = lambda r: inertia_inv(ext_forque(motor, r) - inertia(r).commutator(r))
	from numga.examples.integrators import RK1
	rate = RK1(dr, rate, dt)
	motor = exp2(rate * dt / 2) * motor
	return motor, rate


def explicit_rk4(motor, rate, inertia, inertia_inv, dt, ext_forque):
	"""RK4 integration of lie state"""
	dr = lambda r: inertia_inv(ext_forque(motor, r) - inertia(r).commutator(r))
	from numga.examples.integrators import RK4
	rate = RK4(dr, rate, dt)
	motor = exp2(rate * dt / 2) * motor
	# motor = (rate * dt / 2).exp() * motor
	return motor, rate
