"""Test the tennis racket theorem, or medial axis theorem."""

import numpy as np

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext as Context
from numga.examples.physics.core import Body


dt = 1 / 4
runtime = 200

# works for p=2,3,4.
# p=4 is fascinating; 3rd axis somehow seems stable, whereas 4th axis appears chaotic?

# somehow in 5d things blow up; limitation of the integrator?
# even when clamping the kinetic energy of each body at every timestep,
# the 5d bodies seem to work themselves into a corner of state-space,
# where the energy keeps growing without bound within a single timestep,
# even if starting from the same fixed energy at every timestep
context = Context(Algebra.from_pqr(4, 0, 0), dtype=np.float64)


def make_n_cube(N):
	b = ((np.arange(2 ** N)[:, None] & (1 << np.arange(N))) > 0)
	return (2 * b - 1)
def make_n_rect(N):
	return make_n_cube(N) * (np.arange(N) + 1)
def make_point_cloud():
	points = make_n_rect(context.algebra.description.n_dimensions)
	return context.multivector.vector(points).dual()

points = make_point_cloud()
nb = len(context.subspace.bivector())
body = Body.from_point_cloud(points=points.repeat('... -> b ...', b=nb))

# set up some initial conditions; equal push along all axes; plus jitter.
body.rate = body.rate + context.multivector.bivector(np.eye(nb) + np.random.normal(size=(nb, nb)) * 1e-5)
E = body.kinetic_energy()

states = []
for i in range(int(runtime / dt)):
	body = body.integrate(dt)
	print('e before norm')
	print(body.kinetic_energy())
	energy_violation = body.kinetic_energy() / E
	body = body.copy(rate=body.rate / energy_violation.sqrt())
	print('e after norm')
	print(body.kinetic_energy())

	print(i)
	states.append(body.motor)

# visualize tumbling behavior
import matplotlib.pyplot as plt
v = np.array([s.values for s in states])
fig, ax = plt.subplots(nb, squeeze=False)
for i in range(nb):
	ax[i, 0].plot(v[:, i])
	# this shows initial energy; so we can identify median axis
	ax[i, 0].set_ylabel(int(E.values[i]))
plt.show()
