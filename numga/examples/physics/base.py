"""Boilerplate classes to support a XPBD style rigid body engine using geometric algebra"""

import numpy as np


# some type annotations
class Scalar:
	""""""
class Motor:
	"""Positions/rotation states are encoded as even-graded multivectors"""
	pass
class BiVector:
	"""rates are encoded as a bivector"""
	pass
class Point:
	"""dual to 1-vec. encodes positions"""
class Line:
	"""Dual to 2-vec; momenta, forques
	note; questionable to call momenta a line in higher dims,
	if it cant be factored into the join of two points
	in general will be an anti-bivector though
	"""

class BodyBase:
	motor: Motor                # rigid body state; global = motor >> local; local = motor << global
	rate: BiVector              # rates of change in state; to be interpreted in body-local frame

	inertia: "UnaryOperator"      # inertia map, in body local frame
	inertia_inv: "UnaryOperator"   # inverse inertia map, in body local frame


	def __init__(self, motor, rate, centroid, inertia, inertia_inv, damping, gravity):
		assert motor.subspace.equals.motor()
		assert rate.subspace.equals.bivector()
		assert gravity.subspace.inside.vector()
		assert inertia.output.equals.antibivector()
		assert inertia_inv.output.equals.bivector()
		assert damping.subspace.equals.scalar()
		context = motor.context
		self.motor = motor
		self.rate = rate
		self.inertia = inertia
		self.inertia_inv = inertia_inv
		self.damping = damping
		self.gravity = gravity
		self.centroid = centroid

	def __getitem__(self, idx):
		"""slice up body set"""
		return type(self)(
			self.motor[idx],
			self.rate[idx],
			self.centroid[idx],
			# FIXME: these slices on the operator cause a copy and thus a recompute on cached properties like the kernels.
			#  this is a design flaw. undecided what to do about it, but good to keep tabs on it
			self.inertia[idx],
			self.inertia_inv[idx],
			self.damping[idx],
			self.gravity[idx],
		)

	def concatenate(self, other):
		return type(self)(
			self.motor.concatenate(other.motor, axis=0),
			self.rate.concatenate(other.rate, axis=0),
			self.centroid.concatenate(other.centroid, axis=0),
			self.inertia.concatenate(other.inertia, axis=0),
			self.inertia_inv.concatenate(other.inertia_inv, axis=0),
			self.damping.concatenate(other.damping, axis=0),
			self.gravity.concatenate(other.gravity, axis=0),
		)

	def copy(self, motor=None, rate=None, gravity=None, damping=None) -> "Body":
		return type(self)(
			motor=motor or self.motor,
			rate=rate or self.rate,
			damping=damping or self.damping,
			gravity=gravity or self.gravity,
			centroid=self.centroid,
			inertia=self.inertia,
			inertia_inv=self.inertia_inv,
		)

	@classmethod
	def from_point_cloud(cls, points: Point) -> "BodyBase":
		"""Initialize a body from a point cloud"""
		context = points.context
		assert points.subspace.inside.antivector()
		inertia = points.inertia_map().sum(axis=-3)
		centroid = points.sum(axis=-2)

		return cls(
			motor=context.multivector.motor(),
			rate=context.multivector.bivector(),
			centroid=centroid,
			inertia=inertia,
			inertia_inv=inertia.inverse(),
			damping=context.multivector.scalar() * 0,
			gravity=context.multivector.vector()
		)

	def kinetic_energy(self) -> Scalar:
		"""Compute kinetic energy of each body"""
		r = self.rate
		I = self.inertia
		return I(r) & r     # missing a factor -1/2


class ConstraintBase:
	"""Anchors are a pair of points, describing the location of the constraint in the body local frame"""
	body_idx: np.array       # [2, n], int; body index the anchors connect to
	anchors: Point           # [2, n]       body local coordinate of anchors
	compliance: Scalar       # [n]          compliance of each constraint

	def __init__(self, body_idx, anchors, compliance):
		context = anchors.context
		assert anchors.subspace.equals.antivector()
		assert compliance.subspace.equals.scalar()
		self.body_idx = body_idx
		self.anchors = anchors
		self.compliance = compliance
		self.unique = np.unique(body_idx).size == body_idx.size
		# numerically encode how constraints connect to anchors; from one to the other
		self.connectivity = context.multivector.scalar([[+1 / 2], [-1 / 2]])[:, None]
		assert self.connectivity.shape == (2, 1)

	def __getitem__(self, idx):
		"""slice up constraint set"""
		return type(self)(
			self.body_idx[:, idx],
			self.anchors[:, idx],
			self.compliance[idx],
		)
	def concatenate(self, other):
		return type(self)(
			# FIXME: concat is numpy/jax specific, no?
			np.concatenate([self.body_idx, other.body_idx], axis=1),
			self.anchors.concatenate(other.anchors, axis=1),
			self.compliance.concatenate(other.compliance, axis=0),
		)
