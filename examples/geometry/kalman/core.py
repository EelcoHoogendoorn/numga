"""Pose filtering on the motor manifold in PGA2D: the covariance is a map from readouts to bivectors.

The state is a motor; its uncertainty is a covariance over body-frame bivector perturbations.
A linear readout of a bivector is a line, so the covariance is a unary extensor line -> bivector:
the twist correlated with a readout. Prediction moves it like any map, pulling the readout
through the step and pushing the twist back, and the update solves with the map inverse. The
3x3 Jacobians of the motion and measurement models are never derived by hand.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Bivector = ga.gatype.bivector()
Scalar = ga.gatype.scalar()
Line = ga.gatype.vector()
Covariance = ga.gatype((Bivector, Line))      # twist <- linear readout of a twist (a line)


# --- math -----------------------------------------------------------------------------
def kalman_filter(estimate: Motor, sigma: Covariance, steps: Motor, measurements: Motor,
                  motion_noise: Covariance, measurement_noise: Covariance) -> Iterator[tuple[Motor, Covariance]]:
    """Kalman filter on motors, driven by body-frame steps and noisy full-pose measurements.

    Pose errors are right perturbations, estimate * exp(delta / 2), with sigma
    their covariance on bivectors. motion_noise adds uncertainty at each prediction;
    measurement_noise is the covariance of measurement error in the same perturbation convention.
    Each measurement is paired with the motion steps since the previous reading.
    Predict through those steps, then yield the corrected pose and covariance.
    """
    for prediction_steps, measured in zip(steps, measurements):
        # Predict. A perturbation on the right of the estimate, m exp(δ/2), is carried through
        # the step by step⁻¹ δ step: the adjoint of the step is its sandwich with a hole.
        for step in prediction_steps:
            estimate = estimate * step
            # The covariance takes a readout of the perturbation to the twist correlated with it,
            # so it moves like any map: pull the readout through the step, push the twist back.
            sigma = step << sigma(step >> Line) + motion_noise

        # Update on a noisy pose measurement. The innovation is the log of the relative
        # motor, the gain is a ratio of covariance maps, and the correction is exponentiated.
        innovation = (estimate.inverse() * measured).log() * 2
        gain = sigma((sigma + measurement_noise).inverse())
        estimate = estimate * (gain(innovation) * 0.5).exp()
        sigma = sigma - gain(sigma)

        yield estimate, sigma


def position_ellipse(estimate: Motor, sigma: Covariance, origin: Point):
    """The estimated position, with the principal variances and axes of its uncertainty.

    A readout of position along a line, l & shift(δ), is a readout of the twist through the
    incidence pairing, so the covariance of those readouts is a form on lines. Against the
    line metric, which measures a line's normal and not its offset, its principal readouts
    are the ellipse axes. The offset mode is zero in both forms and has no variance.
    """
    here = estimate >> origin                                  # [n] Point
    shift = Bivector.commutator(here)(estimate >> Bivector)    # [n] Point <- Bivector
    readout = (Line & Bivector).solve(Line & shift)            # [n] Line <- Line
    position = readout & sigma(readout)                        # [n] Scalar <- (Line, Line)
    variances, axes = position.eig()                           # [n, 3] Scalar, [n, 3] Line; the eigenpairs of this symmetric pair of forms are real
    return here, variances.real(), axes.real()


# --- plumbing: covariance, sampling and simulation --------------------------------------
def covariance(translation_std: float, rotation_std: float) -> Covariance:
    """Body-frame noise: isotropic translation, and rotation about the body origin."""
    translation = mv.yw * (mv.yw & Line) + mv.wx * (mv.wx & Line)   # [] Bivector <- Line
    rotation = mv.xy * (mv.xy & Line)                               # [] Bivector <- Line
    return translation * translation_std**2 + rotation * rotation_std**2


def sample(cov: Covariance, rng: np.random.Generator) -> Bivector:
    """One bivector drawn from a covariance map.

    Readouts orthonormal in the covariance's own form carry independent unit normal draws:
    the twists they select, cov(l), then have covariance cov (Line & cov)^-1 cov = cov.
    """
    readouts = Line & cov                                      # [] Scalar <- (Line, Line)
    _, lines = readouts.eigh(readouts)                         # [3] Line: orthonormal in the form itself
    return (cov(lines) * rng.normal(size=3)).sum(axis=0)       # [] Bivector


def simulate_motion(initial: Motor, increments: Bivector, motion_noise: Covariance,
                    measurement_noise: Covariance, rng: np.random.Generator):
    """Simulate noisy motion and a pose measurement after each group of increments."""
    truth = dead = initial
    true_path, dead_path, measurements = [], [], []
    for segment in increments:
        for increment in segment:
            truth = truth * ((increment + sample(motion_noise, rng)) * 0.5).exp()
            dead = dead * (increment * 0.5).exp()
        true_path.append(truth)
        dead_path.append(dead)
        measurements.append(truth * (sample(measurement_noise, rng) * 0.5).exp())
    return Extensor.stack(true_path), Extensor.stack(dead_path), Extensor.stack(measurements)
