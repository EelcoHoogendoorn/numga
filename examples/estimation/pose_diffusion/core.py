"""Station keeping in gusts computes how far a dynamically positioned vessel wanders from its set point under random wind and wave disturbances.

A vessel holds a fixed position and heading with its thrusters while wind and wave gusts push it off course and the controller pushes it back in proportion to the error. Because gusts hit the side of the hull harder than the bow, the vessel wanders further sideways than forward, and the covariance of its excursions determines both the clearance required around the set point and the controller gain needed to hold a given tolerance.

The pose error is modelled as a twist bivector from the set point to the vessel's pose, paired with antibivector readout lines through `line & error`. Where the term *duality map* here means a linear map between a space and its dual readout space, the pose spread is a covariance duality map `Twist <- Line` whose measurement variance along a line is `line & sigma(line)`. Proportional control together with any steady turning rate defines a linear drift map `Twist.commutator(rate) - gain * Twist` acting on the covariance's twist output and, via `dynamics.adjugate()`, on its line input, reducing the steady-state balance between gust injection and controller damping to one linear solve batched across controller gains.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
# An antibivector line reads a twist through the regressive product.
Line = ga.gatype.antibivector()                     # Antibivector-
# A plane vector reads the displacement of a point through the regressive product.
Plane = ga.gatype.vector()                          # Vector+
Twist = ga.gatype.bivector()                        # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Covariance = ga.gatype((Twist, Line))               # Twist+ <- Line-
Dynamics = ga.gatype((Twist, Twist))                # Twist+ <- Twist-
Spread = ga.gatype((Scalar, Plane, Plane))          # Scalar+ <- (Plane+, Plane+)
Gusts = ga.gatype((Twist, Line))                    # Twist+ <- Line-
Quadric = ga.gatype((Plane, Point))                 # Plane+ <- Point-

# Commanded set point at the origin.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def gust_covariance(gusts: Gusts) -> Covariance:
    """Unit-variance white noise across an orthonormal line frame maps through `gusts` (`[] Twist+ <- Line-`)
    to the covariance rate (`[] Twist+ <- Line-`) added per second."""
    # Orthonormal readout lines carry independent unit-variance white noise, so each adds its gust dyad.
    frame = mv.bivector(np.eye(3)).dual()                                           # [3] Line-
    return (gusts(frame) * (gusts(frame) & Line)).sum(axis=0)                       # [] Twist+ <- Line-


def steady_covariance(dynamics: Dynamics, noise: Covariance) -> Covariance:
    """Balancing drift `dynamics` (`[...] Twist+ <- Twist+`) on both sides against `noise` (`[] Twist+ <- Line-`)
    gives the steady-state covariance (`[...] Twist+ <- Line-`)."""
    # Expanding the covariance on dyads turns its drift rate into a linear map on both twist factors and the line.
    lyapunov = dynamics(Twist) * (Twist & Line) + Twist * (dynamics(Twist) & Line)  # [...] Twist+ <- (Twist+, Twist+, Line-)
    # Matching the line input against -noise solves for both twist factors; Line.dual() feeds a line in as a twist.
    return lyapunov.lstsq(-noise)(Line.dual())                                      # [...] Twist+ <- Line-


def position_spread(sigma: Covariance) -> Spread:
    """Pulling displacement-measuring planes back through the set-point motion maps `sigma`
    (`[...] Twist+ <- Line-`) to the symmetric position variance form (`[...] Scalar+ <- (Plane+, Plane+)`)."""
    # A twist displaces the set point by their commutator; its adjugate pulls displacement planes back to twist lines.
    shift = Twist.commutator(origin)                                                # [] Point- <- Twist+
    readout = shift.adjugate()                                                      # [] Line- <- Plane+
    return readout & sigma(readout)                                                 # [...] Scalar+ <- (Plane+, Plane+)


def position_ellipsoid(sigma: Covariance, sigmas: float) -> Quadric:
    """Pushing `sigma` (`[...] Twist+ <- Line-`) to the set point's second moment gives the implicit confidence
    quadrics (`[...] Plane+ <- Point-`) at `sigmas` standard deviations."""
    shift = Twist.commutator(origin)                                                # [] Point- <- Twist+
    readout = shift.adjugate()                                                      # [] Line- <- Plane+
    moment = origin * (Plane & origin) + shift(sigma(readout))                      # [...] Point- <- Plane+
    return moment.solve(Point) - (1 + sigmas**2) * mv.w * (mv.w & Point)            # [...] Plane+ <- Point-


def fleet_diffusion(
    dynamics: Dynamics,
    gusts: Gusts,
    noise: Covariance,
    limit: Covariance,
    dt: float,
    n_steps: int,
    n_stride: int,
    n_vessels: int,
    seed: int,
) -> Iterator[tuple[Motor, Covariance, Covariance]]:
    """Yields `(poses, predicted, limit)` (`[n_vessels] Motor+`, `[] Twist+ <- Line-`, `[] Twist+ <- Line-`) every
    `n_stride` steps across `n_steps` Euler-Maruyama steps of a fleet starting at the set point."""
    rng = np.random.default_rng(seed)
    errors = mv.bivector(np.zeros((n_vessels, 3)))                                  # [n_vessels] Twist+
    predicted = noise * 0.0                                                         # [] Twist+ <- Line-
    for step_index in range(n_steps):
        # Integrated white noise scales as sqrt(dt), while drift acts on both twist output and line input of predicted.
        white = mv.bivector(rng.normal(size=(n_vessels, 3))).dual() * np.sqrt(dt)   # [n_vessels] Line-
        errors = errors + dynamics(errors) * dt + gusts(white)                      # [n_vessels] Twist+
        growth = dynamics(predicted) + predicted(dynamics.adjugate()) + noise       # [] Twist+ <- Line-
        predicted = predicted + growth * dt                                         # [] Twist+ <- Line-
        if (step_index + 1) % n_stride == 0:
            yield (errors * 0.5).exp(), predicted, limit

