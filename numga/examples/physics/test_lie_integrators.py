import numpy as np
from numga.examples.physics.lie_integrators import *


def test_newton():
	func = lambda x: x**2 - 1
	solver = newton_solver(func)
	r= solver(jnp.array([1.5]))
	print(r)


def test_log_exp():
	from numga.backend.numpy.context import NumpyContext as Context

	ctx = Context('x+y+z+')
	b = ctx.multivector.bivector([2,0,0])
	r = exp2(b)
	print(r)
	print(log2(r))

	ctx = Context('x+y+z+w+')
	b = ctx.multivector.bivector([1,2,3,4,5,6])
	r = exp2(b)
	print(r)
	print(log2(r))



def make_n_cube(N):
	b = ((np.arange(2 ** N)[:, None] & (1 << np.arange(N))) > 0)
	return (2 * b - 1)

def make_n_rect(N):
	return make_n_cube(N) * (np.arange(N) + 1)


def test_tennis_racket():
	from jax.config import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	# works for p=2,3,4,5
	# p>3 is fascinating; some medial axes become seemingly chaotic
	# but stranger still, some medial axes actually stabilize?
	# also, different unstable axes appear to show qualitatively different behavior
	context = Context((4, 0, 0), dtype=jnp.float64)

	dt = 0.2
	runtime = 2000


	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	points = make_n_rect(nd)
	points = context.multivector.vector(points).dual()

	inertia = points.inertia_map().sum(axis=-3)
	inertia_inv = inertia.inverse()

	rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5))
	motor = context.multivector.motor() * np.ones((nb))
	kinetic = lambda rate: inertia(rate).wedge(rate)


	import functools
	e = context.multivector.empty() #* np.ones((nb))
	from numga.examples.physics.lie_integrators import variational_lie_verlet as integrator
	integrator = functools.partial(integrator, dt=dt, ext_forque=lambda m, r: e)
	integrator = jax.vmap(integrator, (0,0, None, None))
	integrator = jax.jit(integrator)

	states = []
	for i in range(int(runtime / dt)):
		motor, rate = integrator(motor, rate, inertia, inertia_inv)
		# states.append(kinetic(rate).values)
		# states.append(motor.values)
		states.append(rate.values)

	states = jnp.array(states)
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(nb, 1)
	for i in range(nb):
		ax[i].plot(states[:, i])
	plt.show()


def test_2dpga():
	"""test energy drift in 2d pga

	in the force-free case, things seem quite alright
	RK4 does a decent job; dissipative for large timesteps

	Testing of linear posistion stability raises a lot of questions though
	lie-verlet seems very broken,
	though lie-newmark seems to do ok if imperfect,
	but not much better than rk4, if not worse?

	"""
	np.random.seed(0)
	from jax.config import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	# context = Context((3, 0, 0), dtype=jnp.float64)
	context = Context('x+y+w0', dtype=jnp.float64)

	# dt = 1/4
	dt = .1
	runtime = 2000


	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	points = make_n_rect(nd)
	points = context.multivector.vector(points).dual().normalized()

	inertia = points.inertia_map().sum(axis=-3)
	inertia_inv = inertia.inverse()

	# rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5)[1])
	rate = context.multivector.bivector([1,1,1])
	# rate = context.multivector.bivector(np.random.normal(size=(nb))*0.3)
	print(rate)
	motor = context.multivector.motor()
	kinetic = lambda rate: inertia(rate).wedge(rate)


	import functools
	e = context.multivector.empty()
	# from numga.examples.physics.lie_integrators import variational_lie_verlet as integrator
	# from numga.examples.physics.lie_integrators import explicit_lie_newmark as integrator
	from numga.examples.physics.lie_integrators import explicit_rk4 as integrator
	integrator = functools.partial(integrator, dt=dt, ext_forque=lambda m, r: e)
	integrator = jax.jit(integrator)

	states = []
	energy = []
	for i in range(int(runtime / dt)):
		motor, rate = integrator(motor, rate, inertia, inertia_inv)
		energy.append(kinetic(rate).values)
		states.append(motor.values)
		# states.append(rate.values)

	# print(jnp.array(states))
	import matplotlib.pyplot as plt
	plt.plot(np.array(energy))
	plt.show()
	plt.plot(np.array(states))#-200:])
	plt.show()


def test_potential():
	"""
	test conservations props of rotor based potentials
	"""


def test_verlet():
	"""
	need to test interaction between lie integrators and verlet correction steps
	should we track rotor delta between pre and post integrate;
	and derive a rate-delta from that?
	# FIXME: initial delta attemps seem broken?

	or should we backtrace the entire forward integrator?
	what we do right now is essentially backtrace forward euler
	"""