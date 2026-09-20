"""Rigid body dynamics and Extended Position-Based Dynamics (XPBD) in Geometric Algebra."""

from __future__ import annotations

from typing import Sequence

from numga import Extensor
from examples.mechanics.integrators import RK4
from examples.mechanics.rigid_body.base import BodyBase, ConstraintBase, register_pytree


class Body(BodyBase):
    """Rigid body dynamics evaluator."""

    @classmethod
    def from_point_cloud(cls, points: Extensor) -> Body:
        # Each point contributes a rate-to-momentum map; add before inverting.
        Bivector = points.context.gatype.bivector()
        inertia = (points & points.commutator(Bivector)).sum(axis=-1)
        return cls.from_mass_properties(points.sum(axis=-1), inertia, inertia.inverse())

    def forques(self) -> Extensor:
        """External forque line on each body, in body-local frame."""
        damping = -(self.rate * self.damping).dual()
        gravity_local = self.motor << self.gravity
        gravity = self.first_moment & gravity_local
        return damping + gravity

    def rate_derivative(self) -> Extensor:
        """Generalized Euler rotational equation in body frame:
        d/dt rate = I^-1(forque - I(rate) x rate)
        """
        momentum = self.inertia(self.rate)
        gyro = momentum.commutator(self.rate)
        return self.inertia_inv(self.forques() - gyro)

    def pre_integrate(self, dt: float) -> Body:
        """Verlet pre-integration step: unconstrained inertial state update."""
        rate = RK4(lambda r: self.copy(rate=r).rate_derivative(), self.rate, dt)
        motor = self.motor * (rate * (-dt / 2)).exp()
        return self.copy(motor=motor, rate=rate)

    def post_integrate(self, old: Body, dt: float) -> Body:
        """Verlet post-integration step: update rates from relaxed motors."""
        motor = self.motor.normalized()
        rate = (~old.motor * motor).log() * (-2 / dt)
        return self.copy(motor=motor, rate=rate)

    def integrate(self, dt: float, constraint_sets: Sequence[Constraint] = ()) -> Body:
        """Full Verlet integration step with XPBD constraint relaxation."""
        new = self.pre_integrate(dt)
        for c in constraint_sets:
            new = c.apply(new, dt=dt)
        new = new.post_integrate(self, dt=dt)
        for c in constraint_sets:
            new = c.v_apply(new, dt=dt)
        return new


class Constraint(ConstraintBase):
    """XPBD point-to-point constraint between pairs of rigid bodies."""

    def apply(self, bodies: Body, dt: float) -> Body:
        """Relax position constraint violations."""
        idx = self.body_idx
        motors = self.apply_indexed(bodies.motor[idx], bodies.inertia_inv[idx], dt)
        return bodies.copy(motor=bodies.motor.at[idx].set(motors))

    def apply_indexed(self, motors: Extensor, inertia_inv: Extensor, dt: float) -> Extensor:
        """Compute relaxed motor states minimizing constraint violation."""
        anchors = motors >> self.anchors
        forque = anchors[0] & anchors[1]
        magnitude = forque.norm()
        direction = forque / (magnitude + 1e-26)

        local_dir = motors << direction
        steps = self.distribute_forque(
            local_dir,
            magnitude,
            inertia_inv,
            self.compliance / (dt**2),
        )
        return motors * (steps * -0.5).exp()

    def v_apply(self, bodies: Body, dt: float) -> Body:
        """Relax velocity constraint violations."""
        idx = self.body_idx
        rates = self.v_apply_indexed(bodies.motor[idx], bodies.rate[idx], bodies.inertia_inv[idx], dt)
        return bodies.copy(rate=bodies.rate.at[idx].set(rates))

    def v_apply_indexed(
        self, motors: Extensor, rates: Extensor, inertia_inv: Extensor, dt: float
    ) -> Extensor:
        """Resolve velocity impulses at anchors."""
        velocities = motors >> self.anchors_map(rates)
        forque = -(self.connectivity * velocities).sum(axis=0)
        magnitude = forque.norm()
        direction = forque / (magnitude + 1e-26)

        local_dir = motors << direction
        steps = self.distribute_forque(
            local_dir,
            magnitude,
            inertia_inv,
            self.compliance * 0.0,
        )
        return rates + steps

    def distribute_forque(
        self,
        directions: Extensor,
        magnitude: Extensor,
        inertia_inv: Extensor,
        compliance: Extensor,
    ) -> Extensor:
        """Distribute an impulse forque line to momentum-conserving bivector displacements."""
        steps = inertia_inv(directions)
        inertial_compliances = steps & directions
        total_compliance = compliance + inertial_compliances.sum(axis=0) + 1e-26
        multiplier = magnitude / total_compliance
        return steps * self.connectivity * multiplier


register_pytree(Body, ("motor", "rate", "first_moment", "inertia", "inertia_inv", "damping", "gravity"))
register_pytree(Constraint, ("body_idx", "anchors", "compliance"))
