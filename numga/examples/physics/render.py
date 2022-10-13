import jax
import numpy as np


def write_animation_simulation(
	bodies,
	render,
	output_filename,
):
	"""write an animation based on a set of states"""
	import imageio
	writer = imageio.get_writer(output_filename)
	for body in bodies:
		im = render(body)[..., None]
		# w, h, c = im.shape
		# im = im.reshape(w//2, 2, h//2, 2, c).mean(axis=(1, 3)).astype(im.dtype)
		writer.append_data(np.array(im))
	writer.close()


def render(context, states):
	if context.algebra.n_dimensions == 2:
		from numga.examples.physics.render_1 import setup_rays, render
		rays, mask = setup_rays(context)
		write_animation_simulation(
			states,
			jax.jit(lambda bodies: render(rays, mask, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)
	if context.algebra.n_dimensions == 3:
		from numga.examples.physics.render_2 import setup_rays, render
		rays, mask = setup_rays(context)
		write_animation_simulation(
			states,
			jax.jit(lambda bodies: render(rays, mask, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)

	if context.algebra.n_dimensions == 4:
		from numga.examples.physics.render_3 import setup_rays, render
		c_motor, plane, rays = setup_rays(context)
		write_animation_simulation(
			states,
			jax.jit(lambda bodies: render(context, c_motor, plane, rays, bodies)),
			output_filename=f'animation_{context.algebra.description.pqr_str}.gif'
		)
