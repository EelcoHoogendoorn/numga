import numpy as np
from numga.examples.physics.lie_integrators import *
np.random.seed(0)

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
	return (2. * b - 1.)

def make_n_rect(N):
	return make_n_cube(N) #* (np.arange(N) + 1)


def test_inertia():
	from jax import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	context = Context((3, 0, 1), dtype=jnp.float64)

	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	# points = make_n_rect(nd)
	# points = context.multivector.vector(points).dual()

	from numga.multivector.test.util import random_positive_normal_vector
	points = random_positive_normal_vector(context, (100,)).dual()
	# points = points - points.mean(axis=0)

	p = random_positive_normal_vector(context, (2, 10, 1))
	f = p[0] ^ p[1]

	# inertia = points.inertia_map().sum(axis=-3)
	# inertia_inv = inertia.inverse()
	print(f.squared())
	# q = inertia(f)
	p = random_positive_normal_vector(context, (2, 10, 1))
	f = p[0] ^ p[1]
	q = (points & points.commutator(f)).sum(axis=1)
	print(q.squared())

	# rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-3))



def test_tennis_racket():
	from jax import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	# works for p=2,3,4,5
	# p>3 is fascinating; some medial axes become seemingly chaotic
	# but stranger still, some medial axes actually stabilize?
	# also, different unstable axes appear to show qualitatively different behavior
	context = Context((3, 0, 0), dtype=jnp.float64)

	dt = 0.9
	runtime = 200


	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	points = make_n_rect(nd)
	points = context.multivector.vector(points).dual()

	inertia = points.inertia_map().sum(axis=-3)
	inertia_inv = inertia.inverse()

	rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-3))

	motor = context.multivector.motor() * np.ones((nb))
	kinetic = lambda rate: inertia(rate).wedge(rate)


	import functools
	e = context.multivector.empty() #* np.ones((nb))
	# from numga.examples.physics.lie_integrators import variational_lie_verlet as integrator
	from numga.examples.physics.lie_integrators import explicit_rk4 as integrator
	# from numga.examples.physics.lie_integrators import explicit_lie_newmark as integrator
	# from numga.examples.physics.lie_integrators import explicit_lie_newmark_rev as integrator
	integrator = functools.partial(integrator, dt=dt, ext_forque=lambda m, r: e)
	integrator = jax.vmap(integrator, (0, 0, None, None))
	integrator = jax.jit(integrator)

	states = []
	energies = []
	momenta = []
	motors = []
	for i in range(int(runtime / dt)):
		motor, rate = integrator(motor, rate, inertia, inertia_inv)
		energies.append(kinetic(rate).values)
		momenta.append(inertia(rate).values)
		motors.append(motor.values)
		# states.append(motor.values)
		# states.append((rate).values)
		states.append((motor << rate).values)

	import matplotlib.pyplot as plt
	plt.plot(jnp.array(energies)[..., 0])
	plt.figure()
	plt.plot((jnp.array(momenta)**2).sum(axis=-1))

	states = jnp.array(motors)[-1000:]
	fig, ax = plt.subplots(nb, 1)
	for i in range(states.shape[1]):
		ax[i].plot(states[:, i])
	plt.show()


def test_tennis_racket_world():
	from jax import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	# works for p=2,3,4,5
	# p>3 is fascinating; some medial axes become seemingly chaotic
	# but stranger still, some medial axes actually stabilize?
	# also, different unstable axes appear to show qualitatively different behavior
	context = Context((3, 0, 0), dtype=jnp.float64)

	dt = 0.9
	runtime = 200


	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	points = make_n_rect(nd)
	points = context.multivector.vector(points).dual()

	inertia = points.inertia_map().sum(axis=-3)
	inertia_inv = inertia.inverse()

	rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-3))
	rate = inertia(rate)
	# print(inertia.kernel)
	# return

	motor = context.multivector.motor() * np.ones((nb))
	kinetic = lambda rate: inertia(rate).wedge(rate)


	import functools
	e = context.multivector.empty() #* np.ones((nb))
	# from numga.examples.physics.lie_integrators import variational_lie_verlet as integrator
	# from numga.examples.physics.lie_integrators import explicit_rk4 as integrator
	# from numga.examples.physics.lie_integrators import explicit_lie_newmark as integrator
	from numga.examples.physics.lie_integrators import momentum_world as integrator
	integrator = functools.partial(integrator, dt=dt, ext_forque=lambda m, r: e)
	integrator = jax.vmap(integrator, (0, 0, None, None))
	integrator = jax.jit(integrator)

	states = []
	energies = []
	momenta = []
	motors = []
	for i in range(int(runtime / dt)):
		motor, rate = integrator(motor, rate, inertia, inertia_inv)
		energies.append(kinetic(rate).values)
		momenta.append(inertia(rate).values)
		motors.append(motor.values)
		# states.append((rate).values)
		states.append((motor >> rate).values)

	import matplotlib.pyplot as plt
	# plt.plot(jnp.array(energies)[..., 0])
	# plt.figure()
	# plt.plot((jnp.array(momenta)**2).sum(axis=-1))

	states = jnp.array(motors)[-10000:]
	fig, ax = plt.subplots(nb, 1)
	for i in range(states.shape[1]):
		ax[i].plot(states[:, i])
	plt.show()


class SplitMotor:
	"""split a motor in an origin-preserving (rotor) and origin-modifying (translator) part
	the origin-preserving part should contain the (potentially) high angular rates,
	whereas the origin-modifying part contains low angular frequencies,
	or zero frequency in the case of degenerate algebras

	"""
	def __init__(self, t, r):
		self.t = t
		self.r = r

	@staticmethod
	def from_motor(motor, origin):
		translator, rotor = motor.motor_split(origin)
		return SplitMotor(translator, rotor)

	def to_motor(self):
		return self.t * self.r

	def integrate(self, forque, rate):
		"""integrate forque in world space"""

	# def reverse(self):
	# 	# FIXME: no?
	# 	return SplitMotor(self.t.reverse(), self.r.reverse())
	def sandwich(self, p):
		return self.t >> (self.r >> p)
	def __rshift__(self, other):
		return self.sandwich(other)
	def __lshift__(self, other):
		return self.r << (self.t << other)

	def __truediv__(self, other):
		"""construct relative motor"""
		# FIXME: construct split motor instead
		return ~self.r * (~self.t * other.t) * other.r

	def split_forque(self, forque):
		t = forque.restrict_subspace(self.t.subspace.dual())
		r = forque.restrict_subspace(self.r.subspace.dual())
		return t, r

	def split_rate(self, rate):
		"""split rates into parts applying to either motor"""
		t_rate = rate.restrict_subspace(self.t.subspace)
		r_rate = rate.restrict_subspace(self.r.subspace)
		return t_rate, r_rate


class State:
	def __init__(self, smotor, rate):
		self.smotor = smotor
		self.rate = rate


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
	from jax import config
	config.update("jax_enable_x64", True)
	import jax.numpy as jnp
	from numga.backend.jax.context import JaxContext as Context
	# context = Context((3, 0, 0), dtype=jnp.float64)
	# context = Context('x+y+w0', dtype=jnp.float64)
	context = Context('x+y+z+', dtype=jnp.float64)
	context = Context('x+y+z+w+', dtype=jnp.float64)

	# dt = 1/4
	dt = .5
	runtime = 200


	nd = context.algebra.description.n_dimensions
	nb = len(context.subspace.bivector())

	# create a point cloud with distinct moments of inertia on each axis
	points = make_n_rect(nd)
	# points[:, -1] /= points[:, -1]
	# points[:, -1] =+ 1
	origin = context.multivector.basis()[-1].dual()
	# print(points)
	# origin = context.multivector.w.dual()
	points = context.multivector.vector(points).dual().normalized() + origin
	# origin = points.mean(axis=0)
	print(origin)


	inertia = points.inertia_map().sum(axis=-3)
	inertia_inv = inertia.inverse()
	# print(inertia.kernel)
	# return

	# rate = context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5)[1])
	rate = context.multivector.bivector([1]*len(context.subspace.bivector()))
	# rate = context.multivector.bivector(np.random.normal(size=(nb))*0.3)
	# print(rate)
	motor = context.multivector.motor()

	translator, rotor = motor.motor_split(origin)
	print(translator)
	print(rotor)
	sm = SplitMotor.from_motor(motor, origin)

	a, b = sm.split_rate(rate)
	print(a)
	print(b)
	return
	kinetic = lambda rate: inertia(rate).wedge(rate)


	import functools
	e = context.multivector.empty()
	# from numga.examples.physics.lie_integrators import variational_lie_verlet as integrator
	# from numga.examples.physics.lie_integrators import explicit_lie_newmark_rev as integrator
	from numga.examples.physics.lie_integrators import explicit_lie_newmark as integrator
	# from numga.examples.physics.lie_integrators import explicit_rk4 as integrator
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
	t = np.arange(int(runtime / dt)) * dt
	plt.plot(t, np.array(states))#-200:])
	plt.plot(t, t / np.sqrt(2))
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