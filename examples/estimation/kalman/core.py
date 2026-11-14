"""A Kalman filter tracks a vehicle's pose in planar space and its uncertainty from dead-reckoned motion commands and sparse noisy pose readings.

The pose is modelled as a motor, and an error in it as a small twist on its right. A linear readout of a twist is an antibivector line paired with it through `line & twist`, where the term *duality map* here means a linear map between a space and its dual readout space; the pose uncertainty is a covariance duality map `Twist <- Line` sending each readout line to its expected twist, so `line & sigma(line)` gives the variance of that readout. Propagating through a commanded step pulls the readout line forward and pushes the twist back via `step << sigma(step >> Line)` with no coordinate Jacobian to derive, the Kalman gain at a reading is a ratio of two covariance maps `sigma((sigma + measurement_noise).solve(Twist))`, and the confidence ellipse of the vehicle's centre pushes the twist covariance forward through `Twist.commutator(positions)`.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
# A linear readout of a twist is an antibivector line; a covariance duality map sends it to a twist.
Line = ga.gatype.antibivector()                     # Antibivector-
# A pose perturbation on the right carrying the estimated motor toward the true motor.
Twist = ga.gatype.bivector()                        # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Covariance = ga.gatype((Twist, Line))               # Twist+ <- Line-
Quadric = ga.gatype((Plane, Point))                 # Plane+ <- Point-

# Unit weight point at the origin.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def filter_update(
    estimate: Motor, sigma: Covariance, reading: Motor, measurement_noise: Covariance,
) -> tuple[Motor, Covariance]:
    """A noisy pose `reading` (`[] Motor+`) with right-perturbation covariance `measurement_noise`
    (`[] Twist+ <- Line-`) updates `estimate` and shrinks `sigma` (`[] Motor+`, `[] Twist+ <- Line-`)."""
    # Relative twist from the current estimate to the reading, weighted by the ratio of covariance maps.
    innovation = (estimate.inverse() * reading).log() * 2                       # [] Twist+
    gain = sigma((sigma + measurement_noise).solve(Twist))                      # [] Twist+ <- Twist+
    correction = (gain(innovation) * 0.5).exp()                                 # [] Motor+
    return estimate * correction, sigma - gain(sigma)


def position_ellipsoid(estimate: Motor, sigma: Covariance, positions: Point, sigmas: float) -> Quadric:
    """Pushing `sigma` (`[...] Twist+ <- Line-`) through the body-twist motion of `positions` (`[...] Point-`)
    gives the implicit confidence quadrics (`[...] Plane+ <- Point-`) at `sigmas` standard deviations."""
    # Body-frame twists carried to world space move the vehicle centre by their commutator with its position.
    shift = Twist.commutator(positions)(estimate >> Twist)                      # [...] Point- <- Twist+
    # Pulling each displacement-measuring plane back through the motion gives the corresponding twist readout line.
    readout = shift.adjugate()                                                  # [...] Line- <- Plane+
    # Adding the pushed-forward position covariance to the position's own dyad forms its second-moment dual quadric.
    moment = positions * (Plane & positions) + shift(sigma(readout))            # [...] Point- <- Plane+
    return moment.solve(Point) - (1 + sigmas**2) * mv.w * (mv.w & Point)        # [...] Plane+ <- Point-


# --- plumbing -------------------------------------------------------------------------
def body_covariance(translation_std: float, rotation_std: float) -> Covariance:
    """Translation variance `translation_std**2` along both body axes and rotation variance `rotation_std**2`
    about the body origin form an isotropic body-frame covariance (`[] Twist+ <- Line-`)."""
    translation = mv.yw * (mv.yw & Line) + mv.wx * (mv.wx & Line)               # [] Twist+ <- Line-
    rotation = mv.xy * (mv.xy & Line)                                           # [] Twist+ <- Line-
    return translation * translation_std**2 + rotation * rotation_std**2


def twist_samples(noise: Covariance, normals: np.ndarray) -> Twist:
    """Standard normal draws `normals` (`[..., 3]`) along readout lines orthonormal in `noise` (`[] Twist+ <- Line-`)
    produce twist perturbations (`[...] Twist+`) with covariance `noise`."""
    readouts = Line & noise                                                     # [] Scalar+ <- (Line-, Line-)
    # Generalized eigenvectors of the readout form with itself are readout lines orthonormal under that form.
    _, lines = readouts.eigh(readouts)                                          # [3] Line-
    return (noise(lines) * normals).sum(axis=-1)                                # [...] Twist+

