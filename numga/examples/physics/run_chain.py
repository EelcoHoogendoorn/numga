"""XPBD swinging rigid body chain link"""

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

from numga.examples.physics.core import Body, Constraint
from numga.backend.jax.pytree import register
Body = register(Body)
Constraint = register(Constraint)

from numga.backend.jax.context import JaxContext as Context
from numga.backend.jax.operator import JaxEinsumOperator, JaxSparseOperator


# want this to work at least for 2d-3d-4d pga/elliptical; still need to add 4d elliptical rendering!
context = Context(
	# 'x+y+',
	'x+y+z+',       # for some reason this is my favorite
	# 'x+y+w0',
	# 'x+y+z+w0',   # should really make the setup a bit more interesting here; not really showing off 3d atm
	otype=JaxEinsumOperator,
	# otype=JaxSparseOperator,
	dtype=jnp.float64
)

from numga.examples.physics.setup_chain import setup_bodies
bodies, constraint_sets = setup_bodies(
	context,
	# do a springy sim in 1d, otherwise no real dynamics to look at
	compliance=1e-2 if context.algebra.n_dimensions == 2 else 1e-9
)


def step(bodies, constraint_sets, dt, unroll=10):
	"""Jax specific part of main loop, to be compiled"""
	def loop_body(_, bodies):
		return bodies.integrate(dt / unroll, constraint_sets)
	return jax.lax.fori_loop(0, unroll, loop_body, bodies)


dt = 1 / 20
runtime = 60    # runtime in seconds

import time
print('time jitting plus warmup call')
t = time.time()
func = step
# cache warmup call; to avoid leaks of cached vectors created during compilation
q = step(bodies, constraint_sets, dt)
print(q)
func = jax.jit(step)
_ = func(bodies, constraint_sets, dt)
print('compile time')
print(time.time() - t)


t = time.time()
states = []
for i in range(1000):
	bodies = func(bodies, constraint_sets, dt)
	# print(i)
	# print('energy: ', bodies.kinetic_energy().values.sum())
	if i % 10 == 0:
		states.append(bodies)
print('simulation time')
print(time.time() - t)

from numga.examples.physics.render import render
render(context, states)
