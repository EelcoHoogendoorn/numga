# FIXME: make an explicit test of energy stability of evolving free rigid body

"""Test the tennis racket theorem, or medial axis theorem."""

import numpy as np


from numga.algebra.algebra import Algebra
from numga.backend.jax.context import JaxContext as Context

import jax

from numga.examples.physics.core import Body, Constraint
from numga.backend.jax.pytree import register
Body = register(Body)
Constraint = register(Constraint)


dt = 2
runtime = 2000

# works for p=2,3,4,5
# p>3 is fascinating; some medial axes become seemingly chaotic
# but stranger still, some medial axes actually stabilize?
context = Context(Algebra.from_pqr(3, 0, 0), dtype=np.float64)

def make_n_cube(N):
	b = ((np.arange(2 ** N)[:, None] & (1 << np.arange(N))) > 0)
	return (2 * b - 1)
def make_n_rect(N):
	return make_n_cube(N) * (np.arange(N) + 1)
def make_point_cloud():
	points = make_n_rect(context.algebra.description.n_dimensions)
	return context.multivector.vector(points).dual()

# create a point cloud with distinct moments of inertia
points = make_point_cloud()
# set up independent identical copies of the point cloud
# equal to the number of independent spin planes
nb = len(context.subspace.bivector())
# body = Body.from_point_cloud(points=points.repeat('... -> b ...', b=nb))
body = Body.from_point_cloud(points=points)

# set up initial conditions;
# each body gets a spin in a different plane; plus some jitter.
# body.rate = body.rate + context.multivector.bivector(np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5)
body.rate = body.rate + context.multivector.bivector((np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5)[1])
E = body.kinetic_energy()


# def wrapper(body, dt):
# 	import functools
# 	e = context.multivector.empty()
# 	integrator = functools.partial(variational_lie_verlet, dt=dt, ext_forque=lambda m, r: e)
# 	m, r = integrator(body.motor, body.rate, body.inertia, body.inertia_inv)
# 	return body.copy(motor=m, rate=r)

def wrapper(body, dt):
	import functools
	e = context.multivector.empty()
	from numga.examples.physics.lie_integrators import explicit_rk4 as integrator
	integrator = functools.partial(integrator, dt=dt, ext_forque=lambda m, r: e)
	m, r = (integrator)(body.motor, body.rate, body.inertia, body.inertia_inv)
	return body.copy(motor=m, rate=r)

# step = eln2
step = jax.jit(wrapper)
# step = jax.jit(lambda body, dt: body.integrate(dt))

states = []
for i in range(int(runtime / dt)):
	body = step(body, dt)
	if False:
		# optionally, enforce strict energy conservation
		energy_violation = body.kinetic_energy() / E
		body = body.copy(rate=body.rate / energy_violation.sqrt())

	print(i)
	states.append(body.rate)

# visualize tumbling behavior
import matplotlib.pyplot as plt
v = np.array([s.values for s in states])
# nb = 1
fig, ax = plt.subplots(nb, squeeze=False)
for i in range(nb):
	ax[i, 0].plot(v[:, i])
	# this shows initial energy; so we can distinguish medial axes from extrema
	ax[i, 0].set_ylabel(int(E.values[i]))
	ax[i, 0].set_ylim(-1.1, 1.1)
plt.show()
