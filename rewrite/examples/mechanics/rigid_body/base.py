"""Data structures and base classes for rigid body physics in Geometric Algebra."""

from __future__ import annotations

from typing import Any, Tuple
import numpy as np

from numga import Extensor


class BodyBase:
    """Rigid body state represented in Geometric Algebra.

    Attributes
    ----------
    motor : Extensor
        Even-graded rotor describing body orientation and position.
        Transforms body-local geometry to world frame via `motor.sandwich(local)`.
    rate : Extensor
        Bivector rate of change (angular and linear velocities) in body-local frame.
    first_moment : Extensor
        Antivector point encoding the total mass and mass centroid.
    inertia : Extensor
        Unary map (bivector -> bivector) mapping velocity rates to momentum lines.
    inertia_inv : Extensor
        Unary map (bivector -> bivector) inverting the inertia tensor.
    damping : Extensor
        Scalar damping coefficient.
    gravity : Extensor
        Antivector gravity direction/line in world coordinates.
    """

    def __init__(
        self,
        motor: Extensor,
        rate: Extensor,
        first_moment: Extensor,
        inertia: Extensor,
        inertia_inv: Extensor,
        damping: Extensor,
        gravity: Extensor,
    ) -> None:
        self.motor = motor
        self.rate = rate
        self.first_moment = first_moment
        self.inertia = inertia
        self.inertia_inv = inertia_inv
        self.damping = damping
        self.gravity = gravity

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.motor.shape

    def __getitem__(self, index: Any) -> BodyBase:
        return type(self)(
            self.motor[index],
            self.rate[index],
            self.first_moment[index],
            self.inertia[index],
            self.inertia_inv[index],
            self.damping[index],
            self.gravity[index],
        )

    def copy(self, **kwargs: Any) -> BodyBase:
        attrs = {
            "motor": self.motor,
            "rate": self.rate,
            "first_moment": self.first_moment,
            "inertia": self.inertia,
            "inertia_inv": self.inertia_inv,
            "damping": self.damping,
            "gravity": self.gravity,
        }
        attrs.update(kwargs)
        return type(self)(**attrs)

    @classmethod
    def from_point_cloud(cls, points: Extensor) -> BodyBase:
        """Initialize a rigid body from an antivector point cloud.

        Parameters
        ----------
        points : Extensor
            Antivector points of shape `[..., n_points]`.
        """
        context = points.context
        spaces = context.algebra.subspace
        bivector = spaces.bivector()

        # Mass centroid (first moment)
        first_moment = points.sum(axis=-1)
        batch_shape = first_moment.shape

        # Inertia tensor = sum_p p & (p x Rate): maps rates to momenta
        per_point = points.regressive(points.commutator(bivector))
        inertia = per_point.sum(axis=-1)
        inertia_inv = inertia.inverse()

        motor = context.multivector.rotor().broadcast_to(batch_shape)
        rate = context.multivector.bivector().broadcast_to(batch_shape)
        damping = (context.multivector.scalar() * 0.0).broadcast_to(batch_shape)
        gravity = context.multivector.antivector(
            np.zeros(batch_shape + (len(spaces.antivector()),))
        )

        return cls(
            motor=motor,
            rate=rate,
            first_moment=first_moment,
            inertia=inertia,
            inertia_inv=inertia_inv,
            damping=damping,
            gravity=gravity,
        )

    def kinetic_energy(self) -> Extensor:
        """Compute the kinetic energy 0.5 * (Rate & Inertia(Rate))."""
        momentum = self.inertia(self.rate)
        return momentum.regressive(self.rate) * 0.5


class ConstraintBase:
    """Pairwise point-to-point constraint between rigid bodies.

    Attributes
    ----------
    body_idx : np.ndarray or jax.Array
        Shape `[2, n_constraints]`, integer indices of connected bodies.
    anchors : Extensor
        Shape `[2, n_constraints]`, anchor coordinates in body-local frames.
    compliance : Extensor
        Shape `[n_constraints]`, constraint inverse stiffness (compliance).
    """

    def __init__(
        self,
        body_idx: Any,
        anchors: Extensor,
        compliance: Extensor,
    ) -> None:
        context = anchors.context
        spaces = context.algebra.subspace
        bivector = spaces.bivector()

        self.body_idx = body_idx
        self.anchors = anchors
        self.compliance = compliance

        conn = np.array([[[+0.5]], [[-0.5]]], dtype=float)
        self.connectivity = context.multivector.scalar(conn)
        self.anchors_map = anchors.regressive(anchors.commutator(bivector))

    def __getitem__(self, index: Any) -> ConstraintBase:
        return type(self)(
            self.body_idx[:, index],
            self.anchors[:, index],
            self.compliance[index],
        )
