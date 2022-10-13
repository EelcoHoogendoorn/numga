import numpy as np
from jax import numpy as jnp

from numga.util import summation


def setup_rays(context, n=200):
	"""Set up perspective camera rays

	Returns
	-------
	camera_motor: Motor
		defines world-to-camera space
		camera_motor >> local = world
		the origin of the camera is where the rays converge
	camera_plane: Point
		point in camera local space
	camera_rays: Line
		rays in camera local space, from origin to camera plane
	"""

	*axes, origin = context.multivector.basis()
	model_space = summation(axes).subspace     # modelling space

	def translator(distance) -> "Motor":
		return ((origin / -2) ^ context.multivector(model_space, distance)).exp()

	def point_embed(distance: jnp.ndarray):
		"""Move away from the origin"""
		return translator(distance) >> origin.dual()

	scale = 1e-1
	ones = jnp.ones((n, n)) * scale
	x = jnp.linspace(-1, +1, n)[:, None] * ones
	y = jnp.linspace(-1, +1, n)[None, :] * ones
	z = ones
	p = jnp.array([x, y, z])
	p = jnp.moveaxis(p, 0, 2)
	# v = axes[0] * x + axes[1] * y
	camera_plane = point_embed(p)
	camera_origin = point_embed([0, 0, 0])

	# move the camera back from the center
	camera_motor = translator([scale*1, scale*0, scale*0-1])
	# rotate the view somewhat around the gravity vector
	camera_motor = (axes[1] ^ axes[2] / 4).exp() * camera_motor

	camera_rays = camera_plane & camera_origin

	return camera_motor, camera_plane, camera_rays.normalized()


def render(context, c_motor, c_plane, c_rays, bodies):
	"""Render a 3d scene, as embedded in a 4d elliptical or euclidian space

	currently it specializes to pga only
	would it be hard to revive elliptical? not really? just need to replace (a&b).norm type measurements,
	with (a/b).log().norm type measurements. I think? would be cool to see from perspective of constraint-generalization

	I bet this could be 10x simpler and more elegant, but I havnt bothered yet.
	"""
	assert context.algebra.n_dimensions == 4
	assert context.algebra.description.pqr == (3, 0, 1)
	*axes, origin = context.multivector.basis()
	origin_dual = origin.dual_inverse()

	# local = body.motor << global;  pulls global into body local space
	# relative motor between camera and bodies; camera_local = r_motor << body_local
	r_motor = bodies.motor.reverse() * c_motor
	# each body origin put in front of camera local space
	sphere_origin = r_motor << origin_dual

	def project(a, b):
		"""b onto a"""
		return (a.inner(b) * a)
	# project each center onto each ray; closest point of approach between ray and sphere
	closest = project(c_rays[:, :, None], sphere_origin).restrict_grade(3)
	closest = closest / closest.values[..., 0]

	# signed length along the ray, from camera plane to closest point
	closest_to_cplane_distance = (closest & c_plane[:, :, None]).norm().select_grade(0)# .inner(c_rays[:, :, None])


	radius = 0.05
	# line connecting origin to closest point
	orthogonal_line = closest & sphere_origin
	# distance difference between closest and the ray-sphere intersection point, as measured along the ray
	closest_to_intersection_distance = (radius**2 - orthogonal_line.norm_squared().select_grade(0)).sqrt()

	cplane_to_intersection_distance: "Scalar" = closest_to_cplane_distance - closest_to_intersection_distance

	# get idx of intersection point closest to cplane
	idx = jnp.argmin(jnp.nan_to_num(cplane_to_intersection_distance.values[..., 0], nan=1e9), axis=-1)

	cplane_to_intersection_distance_min = cplane_to_intersection_distance.take_along_axis(idx[..., None], axis=2)[:, :, 0]
	closest_min = closest.take_along_axis(idx[..., None], axis=2)[:, :, 0]
	closest_to_intersection_distance_min = closest_to_intersection_distance.take_along_axis(idx[..., None], axis=2)[:, :, 0]
	closest_to_cplane_distance_min = closest_to_cplane_distance.take_along_axis(idx[..., None], axis=2)[:, :, 0]

	closest_weight = cplane_to_intersection_distance_min / closest_to_cplane_distance_min
	cplane_weight = closest_to_intersection_distance_min / closest_to_cplane_distance_min
	intersection_point = c_plane * cplane_weight + closest_min * closest_weight

	# coloring
	r_motors = r_motor[None, None, ...].take_along_axis(idx[..., None], axis=2)[..., 0, :]
	# hit points in closest body local space
	local_hit_point = (r_motors >> intersection_point).dual()

	mask = jnp.all(jnp.isnan(cplane_to_intersection_distance.values[..., 0]), axis=2)
	q = local_hit_point.nondegenerate().values
	# q = np.dot(q, ico.T)
	idx = 1 + ((q > 0).sum(axis=-1) % 2)

	return (idx + 1) * (1 - mask)
