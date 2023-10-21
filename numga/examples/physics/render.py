import numpy as np


def write_animation_simulation(
	bodies,
	render,
	output_filename,
):
	"""write an animation based on a set of states"""
	animation = np.array([render(body) for body in bodies])
	import imageio.v3 as iio
	animation -= animation.min()
	animation = animation / animation.max()
	animation = (animation * 255).astype(np.uint8)
	iio.imwrite(output_filename, animation, loop=0, format='GIF', duration=50)


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
