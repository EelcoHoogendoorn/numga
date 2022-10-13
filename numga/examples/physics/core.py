"""This is where the magic happens; 57 lines of code

If you seek to understand this code in depth, this is an invaluable reference:
https://bivector.net/PGADYN.html

The constraint solver approach used here is identical to XPDB; expressed in terms of GA.
https://matthias-research.github.io/pages/publications/XPBD.pdf

Not sure if it meets the requirements for 'simplicity' or 'speed' posted in this challenge;
https://matthias-research.github.io/pages/challenges/pendulum.html
Presumably those conditions applied to the algorithm, not the implementation thereof.

But as far as the implementation goes, this is how it compares:
 * Unlike the code linked above, actually implements the X in XPBD.
   That is, this is a rigid body physics engine; not just a point-particle system.
 * Dimension independent (tested in 1d 2d and 3d, but no restrictions in theory)
 * Signature independent (tested in euclidian and spherical spaces)
 * Faster, by virtue of being JAX compilable code
 * Slightly fewer LOC, comparing core physics to core physics, ignoring whitespace and comments

So I do think this implementation deserves an honorable mention, at least.

If there is any original idea contained in this code,
it is that JAX, GA, and XPBD mesh together really well.

"""

from numga.examples.physics.base import *
from numga.examples.integrators import RK4


# one may be tempted to use full log/exp here;
# but linearization actually seems to improve convergence of constraint solver
def motor_add_step(motor: Motor, step: BiVector) -> Motor:
	"""Apply a bivector step, or integrated-rate variable, to a body motor state"""
	return motor * (step * (-1/2)).exp_linear_normalized()
def motor_relative_step(old: Motor, new: Motor) -> BiVector:
	"""Retrieve a bivector step, or integrated-rate variable, from relative body motor states"""
	return new.reverse_product(old).motor_log_linear_normalized() * -(2)


class Body(BodyBase):
	def forques(self) -> Line:
		"""External forque line on each body, in body local space"""
		damping = -(self.rate * self.damping)
		gravity = self.centroid.dual() ^ (self.motor << self.gravity)
		return (damping + gravity).dual()

	def rate_derivative(self) -> BiVector:
		return self.inertia_inv(self.forques() - self.inertia(self.rate).commutator(self.rate))
	# def motor_derivative(self) -> Motor:
	# 	return (self.motor * self.rate) * (-1/2)
	# def ganja_integrate(self, dt) -> "Body":
	# 	"""Simple euler integrator"""
	# 	return self.copy(
	# 		motor=(self.motor + self.motor_derivative() * dt).normalized(),
	# 		# rate=(self.rate + self.rate_derivative() * dt),
	# 		rate=RK4(lambda r: self.copy(rate=r).rate_derivative(), self.rate, dt),
	# 	)

	def pre_integrate(self, dt: float) -> "Body":
		"""Verlet pre integration step; inertial update of the bodies, ignoring constraints"""
		rate = RK4(lambda r: self.copy(rate=r).rate_derivative(), self.rate, dt)
		motor: Motor = motor_add_step(self.motor, rate * dt)      # integrate motor by exponentiation
		return self.copy(motor=motor, rate=rate)
	def post_integrate(self, old, dt: float) -> "Body":
		"""Verlet post integration step; deduce rates from change in motors, after constraint relaxation"""
		motor = self.motor.normalized()  # NOTE: should not be required if keeping normalized between updates
		# computes rate in motor local frame; equivalent to
		#  motor << motor_relative_step(old.motor, motor)
		rate: BiVector = motor_relative_step(motor.reverse(), old.motor.reverse()) / dt
		return self.copy(rate=rate, motor=motor)           # rate is tracked in body local frame
	def integrate(self, dt: float, constraint_sets=[]) -> "Body":
		"""Verlet style integrator"""
		new = self.pre_integrate(dt)
		for c in constraint_sets:
			new = c.apply(new, dt=dt)
		return new.post_integrate(self, dt=dt)


class Constraint(ConstraintBase):
	"""A simple point-to-point constraint between rigid bodies

	I can say from experience that more complex joints can be implemented in GA with comparative elegance,
	but this demonstration is aiming for minimalism.
	"""

	def apply(self, bodies: Body, dt: float) -> Body:
		"""Relax the constraint set for one step"""
		idx = self.body_idx
		# note: pre-slice the inertia tensors since they are constant anyway?
		motors = self.apply_indexed(bodies.motor[idx], bodies.inertia_inv[idx], dt)
		return bodies.copy(motor=bodies.motor.at[idx].set(motors))

	def apply_indexed(self, motors: Motor, inertia_inv: "Operator", dt: float) -> Motor:
		"""Find updated motors such that constraint violation between anchors is minimized,
		in a least-action sense at the end of the timestep dt,
		balancing work done by the constraint, and inertial work
		"""
		anchors: Point = motors >> self.anchors         # [2, c]; transform anchors on both bodies into world space
		forque: Line = anchors[0] & anchors[1]          # [c]; their connecting line in world space
		magnitude: Scalar = forque.norm()               # [c]; constraint violation magnitude
		direction: Line = forque / (magnitude + 1e-26)  # [c]; constrain violation direction

		steps: BiVector = self.distribute_forque(
			motors << direction,    # map the total required motion to satisfy the constraint back to body local space
			magnitude,
			inertia_inv,
			self.compliance / dt**2
		)
		return motor_add_step(motors, steps)

	def distribute_forque(
			self,
			directions: Line,			   # [2, c] normalized direction lines in local space
			magnitude: Scalar,             # [c], how much to displace along normalized direction
			inertias_inv: "Unaryoperator", # [2, c] maps forque impulse lines to bivector displacements
			constraint_compliance: Scalar, # [c]
	) -> BiVector:                         # [2, c]
		"""Divide an anti-bivector impulse forque line in body local space,
		to two momentum-conserving bivector steps in body local space,
		which will move our configuration towards constraint satisfaction"""
		# reaction bivector steps, to impulse applied along this line
		steps: BiVector = inertias_inv(directions)
		# scalar inertial compliance of each body to movement in this direction
		compliances: Scalar = steps & directions
		# sum to get total compliance towards motion along this direction
		total_compliance: Scalar = constraint_compliance + compliances.sum(axis=0)
		# lagrange multiplier for each constraint
		multiplier: Scalar = magnitude / total_compliance

		return steps * self.connectivity * multiplier
