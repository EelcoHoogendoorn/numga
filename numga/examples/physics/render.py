import numpy as np


def quantizer(n=32, range=(0,255)):
	s = np.linspace(*range, endpoint=True, num=n).astype(np.uint8)
	import scipy
	tree = scipy.spatial.cKDTree(s[:, None])
	def quantize(data):
		d, i = tree.query(data.reshape(-1,1))
		return s[i].reshape(data.shape)
	return quantize


def write_animation_simulation(
	bodies,
	render,
	output_filename,
	q=33,
	**kwargs,
	# q=3	# good default. need q=2 to avoid playback artifacts in discord. q=4 has perceptible aliasing. q=3 offers a nice filesize saving
):
	"""write an animation based on a set of states"""
	animation = np.array([render(body) for body in bodies])
	import imageio.v3 as iio
	animation -= animation.min()
	animation = animation / animation.max() * 255
	# animation = np.floor(animation * 255.999).astype(np.uint8)
	animation = quantizer(n=q)(animation)

	# animation = np.left_shift(np.right_shift(animation, q), q)
	iio.imwrite(output_filename, animation, loop=0, format='GIF', **kwargs)


def render(context, states, jit=lambda x: x):
	if context.algebra.n_dimensions == 2:
		from numga.examples.physics.render_1 import setup_rays, render
		rays, mask = setup_rays(context)
		write_animation_simulation(
			states,
			jit(lambda bodies: render(rays, mask, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)
	if context.algebra.n_dimensions == 3:
		from numga.examples.physics.render_2 import setup_rays, render
		rays, mask = setup_rays(context)
		write_animation_simulation(
			states,
			jit(lambda bodies: render(rays, mask, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)

	if context.algebra.n_dimensions == 4:
		from numga.examples.physics.render_3 import setup_rays, render
		c_motor, plane, rays = setup_rays(context)
		write_animation_simulation(
			states,
			jit(lambda bodies: render(context, c_motor, plane, rays, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)
