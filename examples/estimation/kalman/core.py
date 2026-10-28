"""A Kalman filter tracks a vehicle's pose in space, and how uncertain it is, from dead-reckoned motion commands and sparse noisy pose readings.

The pose is a motor, and an error in it is a small twist on its right. The uncertainty is a covariance of those twists, typed as a map from a line, a linear readout of a twist through their pairing, to the twist correlated with it, `Twist <- Line`. Being a map, it moves through each step of the drive the way any map does, its readout pulled through the step and its twist pushed back, `step << sigma(step >> Line)`, with no Jacobian to derive. The gain at a reading is a ratio of two such maps, and the position ellipsoid is the covariance read through the map from twists to the motion of the vehicle's centre.

In the matrix notation of the extended Kalman filter on $\\mathrm{SE}(2)$ the covariance reads as a $3 \\times 3$ matrix $P$, propagated as $F P F^{\\mathsf T} + Q$ with the adjoint matrix $F$ of each step and updated with the Kalman gain $K = P (P + R)^{-1}$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# A linear readout of a twist is an antibivector line; a covariance maps it to the correlated twist.
Line = ga.gatype.antibivector()                     # Antibivector-
# A pose perturbation, the twist on the right that carries the estimate to the truth.
Twist = ga.gatype.bivector()                        # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Covariance = ga.gatype((Twist, Line))
Quadric = ga.gatype((Plane, Point))


# --- math -----------------------------------------------------------------------------
def update(estimate: Motor, sigma: Covariance, reading: Motor, measurement_noise: Covariance) -> tuple[Motor, Covariance]:
    """A noisy pose reading, its error a right perturbation with covariance `measurement_noise`, corrects the estimate and shrinks its covariance."""
    # The innovation is the log of the relative motor, the gain is a ratio of covariance maps, and the
    # correction is exponentiated.
    innovation = (estimate.inverse() * reading).log() * 2                       # [] Twist+
    gain = sigma((sigma + measurement_noise).inverse())                         # [] Twist+ <- Twist+
    correction = (gain(innovation) * 0.5).exp()                                 # [] Motor+
    return estimate * correction, sigma - gain(sigma)


def position_ellipsoid(estimate: Motor, sigma: Covariance, positions: Point, sigmas: float) -> Quadric:
    """The quadric of each estimated position's uncertainty vanishes on its confidence ellipsoid, `sigmas` standard deviations out."""
    # A twist moves the position by its commutator with it: a map from twists to displacements.
    shift = Twist.commutator(positions)(estimate >> Twist)                      # [states] Point- <- Twist+
    # Its adjugate pulls each plane back to the line that reads the same displacement from the twist.
    readout = shift.adjugate()                                                  # [states] Line- <- Plane+
    # The covariance pushed forward through the motion adds to the position's own dyad: its second moment.
    moment = positions * (Plane & positions) + shift(sigma(readout))            # [states] Point- <- Plane+
    return moment.inverse() - (1 + sigmas**2) * mv.w * (mv.w & Point)           # [states] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def covariance(translation_std: float, rotation_std: float) -> Covariance:
    """A body-frame noise covariance is isotropic in translation and in rotation about the body's origin."""
    translation = mv.yw * (mv.yw & Line) + mv.wx * (mv.wx & Line)               # [] Twist+ <- Line-
    rotation = mv.xy * (mv.xy & Line)                                           # [] Twist+ <- Line-
    return translation * translation_std**2 + rotation * rotation_std**2


def sample(noise: Covariance, normals: np.ndarray) -> Twist:
    """The sampled twists have covariance `noise`: readouts orthonormal in its own form are independent under it,
    so the twists they select, `noise(lines)`, weighted by the independent standard normal draws `[..., 3]`,
    reproduce it, `(noise(lines) * (noise(lines) & Line)).sum(axis=0) == noise`.
    """
    readouts = Line & noise                                                     # [] Scalar+ <- (Line-, Line-)
    # Lines orthonormal in the form itself.
    _, lines = readouts.eigh(readouts)                                          # [modes] Line-
    return (noise(lines) * normals).sum(axis=-1)                                # [...] Twist+
