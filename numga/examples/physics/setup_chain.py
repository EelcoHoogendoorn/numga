import numpy as np
from jax import numpy as jnp

from numga.util import summation
from numga.examples.physics.core import Body, Constraint


def setup_bodies(
		context,
		fixing=True,
		n_bodies=12,
		compliance=1e-9,
		distance=9e-2,
		size=5e-2,
		damping=1e-3,
):
	"""Some really ugly code to initialize an nd chain of linked rigid bodies"""
	*axes, origin = context.multivector.basis()     # arbitrarily pick last axis as the origin
	model_space = summation(axes).subspace          # modelling space
	horizon = origin.subspace.wedge(model_space)    # this is the subspace of translation directions
	ndim = len(axes)

	def translator(distance: "Vector") -> "Motor":
		return ((origin / -2) ^ context.multivector(model_space, distance)).exp()

	def make_cube(n):
		# FIXME: its actually an octagon...
		return jnp.array(np.kron(np.diag(np.arange(n) + 1) / n, [+1, -1]).T)

	def point_embed(distance: jnp.ndarray):
		"""Move away from the origin"""
		# need the dual-inverse to avoid picking up that annoying minus sign in alternating dimensions
		return translator(distance) >> origin.dual_inverse()

	# single body for starters
	points = point_embed(make_cube(ndim) * size)
	bodies = Body.from_point_cloud(points=points)
	bodies = bodies.copy(gravity=axes[0] * 5e-2)
	bodies = bodies.copy(damping=bodies.damping + damping)

	d = jnp.arange(n_bodies) * distance
	d = jnp.array([d*0] * (ndim - 1) + [d]).T
	qs = translator(d)

	zeros = jnp.zeros(n_bodies, int)
	bodies = bodies[None][zeros]
	bodies = bodies.copy(motor=bodies.motor * qs)

	# init constraints, defining anchor points
	i = jnp.arange(n_bodies - 1)
	body_idx = jnp.array([i, i+1])
	a = jnp.array([[0] * (ndim-1) + [distance], [0] * (ndim-1) + [-distance]]) / 2
	ones = jnp.ones((1, n_bodies - 1, 1))
	a = a[:, None, :] * ones
	anchors = point_embed(a)
	ones = context.multivector.scalar(np.ones((n_bodies-1, 1)))
	constraints = Constraint(
		body_idx,
		anchors,
		compliance=ones * compliance,
	)
	# apply static constraints; zero out translation-like directions (in a way that generalizes to curved spaces)
	if fixing:
		idx = bodies.inertia_inv.operator.output.to_relative_indices(horizon.blades)
		bodies.inertia_inv = bodies.inertia_inv.at[0, :, idx].set(0)
	# partition constraints into two (red/black) sets
	return bodies, [constraints[0::2], constraints[1::2]]
