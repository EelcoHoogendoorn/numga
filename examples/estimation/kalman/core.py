"""Pose filtering on the motor manifold tracks a moving body's pose and uncertainty in the plane.

The state is a pose motor and its uncertainty is a covariance over body-frame twist perturbations. A linear readout of a twist is a line, so the covariance is a linear map `Twist <- Line` taking each readout line to the twist correlated with it. Prediction carries the covariance through each motion step as it moves any map, its readout pulled through the step and its twist pushed back, and measurement updates correct the motor and covariance using the logarithm of the relative pose innovation.

In 2D Projective Geometric Algebra (`PGA2D`, signature `x+y+w0`), lines are vectors, points and infinitesimal twists are bivectors, and rigid motions in the plane are even-grade motors.

In the matrix notation of the extended Kalman filter the covariance reads as a three-by-three matrix propagated and updated by the Jacobians of the motion and measurement models.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
# A pose perturbation, the twist that carries the estimate to the truth.
Twist = ga.gatype.bivector()
Scalar = ga.gatype.scalar()
Line = ga.gatype.antibivector()
# A linear readout of a twist is a line; the covariance takes it to the correlated twist.
Covariance = ga.gatype((Twist, Line))         # Twist <- Line


# --- math -----------------------------------------------------------------------------
def predict(estimate: Motor, sigma: Covariance, step: Motor, motion_noise: Covariance) -> tuple[Motor, Covariance]:
    """Carry the estimate and its covariance through one body-frame step, adding the motion noise.

    Pose errors are right perturbations, `estimate * (delta * 0.5).exp()`, with sigma their
    covariance on twists. A perturbation on the right of the estimate is carried through the step by
    step << delta: the adjoint of the step is its sandwich, step << Twist.
    """
    # The covariance takes a readout of the perturbation to the twist correlated with it,
    # so it moves like any map: pull the readout through the step, push the twist back.
    return estimate * step, (step << sigma(step >> Line)) + motion_noise


def update(estimate: Motor, sigma: Covariance, measured: Motor, measurement_noise: Covariance) -> tuple[Motor, Covariance]:
    """Correct the estimate and its covariance by a noisy pose measurement, its error a right perturbation
    with the measurement noise as covariance."""
    # The innovation is the log of the relative motor, the gain is a ratio of covariance maps, and the
    # correction is exponentiated.
    innovation = (estimate.inverse() * measured).log() * 2
    gain = sigma((sigma + measurement_noise).inverse())
    return estimate * (gain(innovation) * 0.5).exp(), sigma - gain(sigma)


def position_ellipse(estimate: Motor, sigma: Covariance, origin: Point):
    """The estimated position, with the principal variances and axes of its uncertainty.

    A readout of position along a line l, `l & shift(delta)`, is a readout of the twist through the
    incidence pairing, so the covariance of those readouts is a form on lines. The readout takes
    only the lines that see a displacement, those through the origin, which the line metric
    measures fully, so against it the principal readouts are the ellipse axes.
    """
    here = estimate >> origin                                  # [readings] Point
    shift = Twist.commutator(here)(estimate >> Twist)          # [readings] Point <- Twist
    readout = shift.adjugate()                                 # [readings] Line <- Line
    position = readout & sigma(readout)                        # [readings] Scalar <- (Line, Line)
    variances, axes = position.eigh()                          # [readings, modes] Scalar, [readings, modes] Line
    return here, variances, axes


# --- plumbing: covariance, sampling and simulation --------------------------------------
def covariance(translation_std: float, rotation_std: float) -> Covariance:
    """Body-frame noise: isotropic translation, and rotation about the body origin."""
    translation = mv.yw * (mv.yw & Line) + mv.wx * (mv.wx & Line)   # [] Twist <- Line
    rotation = mv.xy * (mv.xy & Line)                               # [] Twist <- Line
    return translation * translation_std**2 + rotation * rotation_std**2


def sample(cov: Covariance, normals: np.ndarray) -> Twist:
    """Twists drawn from a covariance map, given independent standard normal draws `[..., 3]`.

    Readouts orthonormal in the covariance's own form carry independent unit normal draws:
    the twists they select, `cov(lines)`, then have covariance
    `(cov(lines) * (cov(lines) & Line)).sum(axis=0)`, which is `cov` itself.
    """
    readouts = Line & cov                                      # [] Scalar <- (Line, Line)
    # Lines orthonormal in the form itself.
    _, lines = readouts.eigh(readouts)                         # [modes] Line
    return (cov(lines) * normals).sum(axis=-1)                 # [...] Twist
