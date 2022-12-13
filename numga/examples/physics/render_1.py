import numpy as np


def setup_rays(context, n=200):
	"""Set up parralel rays impinging on the unit 1-sphere"""
	ones = np.ones((n, n))
	x = np.linspace(-1.2, +1.2, n)[:, None] * ones
	y = np.linspace(-1.2, +1.2, n)[None, :] * ones
	v = np.array([x, y])
	p = np.moveaxis(v, 0, 2)
	return context.multivector.vector(np.array(p)).dual(), None


def render(rays, mask, bodies):
	"""Render a 1d scene, as embedded in a 2d space
	note; bodies are thickened for sake of visualization
	"""
	context = rays.context
	*axes, origin = context.multivector.basis()

	assert context.algebra.n_dimensions == 2
	ray_norm = rays.norm()
	rays = rays / ray_norm
	# map our rays to body local frame; [n, n, b] 1-vec
	# FIXME: can also just motor origins forward instead?
	# l = (bodies.motor << rays.subspace)(rays[:, :, None])
	l = bodies.motor.reverse().sandwich_map(rays.subspace)(rays[:, :, None])
	# length of join line from each pixel to each body
	# n = (origin.dual() & l).norm().values[..., 0]
	# n = (origin.dual() - l).norm().values[..., 0]
	# FIXME: motor log slows things down
	n = l.reverse_product(origin.dual()).motor_log().dual().norm().values[..., 0]
	# index of closest body
	idx = context.argmin(n, axis=-1)
	# local coords in closest body
	lv = l.dual().take_along_axis(idx[..., None], axis=2).values[:, :, 0]
	# length of join line to closest body
	v = context.min(n, axis=-1)
	idx = 1 + context.logical_xor(lv[..., 0] > 0, lv[..., 1] > 0)
	# FIXME: set up collection of planes in local space, to decide inside/outside?
	radius = 0.04
	# idx = idx.at[v > radius].set(0)
	idx = context.where(v > radius, 0, idx)
	# idx = idx.at[mask].set(-1)
	mask = context.logical_or(ray_norm.values[..., 0] < 0.95, ray_norm.values[..., 0] > 1.05)
	idx = context.where(mask, -1, idx)
	return idx


# def color(context, n, l, ray_norm):
# 	# index of closest body
# 	idx = context.argmin(n, axis=-1)
# 	# local coords in closest body
# 	lv = l.dual().take_along_axis(idx[..., None], axis=2).values[:, :, 0]
# 	# length of join line to closest body
# 	v = context.min(n, axis=-1)
# 	idx = 1 + context.logical_xor(lv[..., 0] > 0, lv[..., 1] > 0)
# 	# FIXME: set up collection of planes in local space, to decide inside/outside?
# 	radius = 0.04
# 	# idx = idx.at[v > radius].set(0)
# 	idx = context.where(v > radius, 0, idx)
# 	# idx = idx.at[mask].set(-1)
# 	mask = context.logical_or(ray_norm.values[..., 0] < 0.95, ray_norm.values[..., 0] > 1.05)
# 	idx = context.where(mask, -1, idx)
# 	return idx
