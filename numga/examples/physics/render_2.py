import numpy as np
from jax import numpy as jnp


def setup_rays(context, n=200):
	"""Set up parralel rays impinging on the unit 2-sphere"""
	ones = jnp.ones((n, n))
	x = jnp.linspace(-1, +1, n)[:, None] * ones
	y = jnp.linspace(-1, +1, n)[None, :] * ones
	*axes, origin = context.multivector.basis()
	v = axes[0] * x + axes[1] * y
	z = (1 - v.dual().norm_squared()).sqrt().values[..., 0]
	# z = ones
	mask = jnp.isnan(z)
	z = jnp.nan_to_num(z, nan=0)
	p = jnp.array([x, y, z])
	p = jnp.moveaxis(p, 0, 2)
	p = jnp.array(p)
	mask = jnp.array(mask)
	return context.multivector.vector(p).dual(), mask


def render(rays, mask, bodies):
	"""Render a 2d scene, as embedded in a 3d elliptical or euclidian space"""
	context = rays.context
	assert context.algebra.n_dimensions == 3
	*axes, origin = context.multivector.basis()
	# map our rays to body local frame; [n, n, b] 1-vec
	# note: bind motor arg first, since n_bodies is much smaller than number of rays!
	# l = (bodies.motor << rays[:, :, None])
	l = bodies.motor.reverse().sandwich_map(rays.subspace)(rays[:, :, None])
	# FIXME: can also just motor origins forward instead?
	# l = (bodies.motor << rays.subspace)(rays[:, :, None])
	# length of join line from each pixel to each body
	n = (origin.dual() & l).norm().values[..., 0]
	# index of closest body
	idx = jnp.argmin(n, axis=-1)
	# local coords in closest body
	lv = jnp.take_along_axis(l.dual().values, idx[..., None, None], axis=2)[:, :, 0]
	# length of join line to closest body
	v = jnp.min(n, axis=-1)
	idx = 1 + jnp.logical_xor(lv[..., 0] > 0, lv[..., 1] > 0)
	# FIXME: set up collection of planes in local space, to decide inside/outside?
	radius = 0.05
	idx = jnp.where(v > radius, 0, idx)
	idx = jnp.where(mask, -1, idx)
	return idx
