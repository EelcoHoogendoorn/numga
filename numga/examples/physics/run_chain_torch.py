"""XPBD swinging rigid body chain link"""

import torch

from numga.backend.torch.context import TorchContext as Context
from numga.backend.torch.operator import TorchEinsumOperator, TorchSparseOperator


# from numga.examples.physics.core import Body, Constraint
# Body = torch.jit.script(Body)
# Constraint = torch.jit.script(Constraint)


# want this to work at least for 2d-3d-4d pga/elliptical; still need to add 4d elliptical rendering!
context = Context(
	'x+y+',
	# 'x+y+z+',       # for some reason this is my favorite
	# 'x+y+w0',
	# 'x+y+z+w0',   # should really make the setup a bit more interesting here; not really showing off 3d atm
	otype=TorchEinsumOperator,
	# otype=TorchSparseOperator,
	dtype=torch.float64
)

from numga.examples.physics.setup_chain import setup_bodies
bodies, constraint_sets = setup_bodies(
	context,
	# do a springy sim in 1d, otherwise no real dynamics to look at
	compliance=1e-2 if context.algebra.n_dimensions == 2 else 1e-9
)


def step(bodies, constraint_sets, dt, unroll=10):
	"""Torch specific part of main loop, to be compiled"""
	for i in range(10):
		bodies = bodies.integrate(dt / unroll, constraint_sets)
	return bodies


dt = 1 / 20
runtime = 60    # runtime in seconds

import time
print('time jitting plus warmup call')
t = time.time()
func = step
# cache warmup call; to avoid leaks of cached vectors created during compilation
# q = step(bodies, constraint_sets, dt, unroll=10)
# print(q)
# func = torch.jit.script(step)
# _ = func(bodies, constraint_sets, dt)
print('compile time')
print(time.time() - t)


t = time.time()
states = []
for i in range(20):
	bodies = func(bodies, constraint_sets, dt, unroll=10)
	# print(i)
	# print('energy: ', bodies.kinetic_energy().values.sum())
	if i % 10 == 0:
		states.append(bodies)
		print(i)
print('simulation time')
print(time.time() - t)

from numga.examples.physics.render import render
render(context, states)
