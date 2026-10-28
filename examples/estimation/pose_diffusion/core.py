"""Station keeping in gusts computes how far a vessel on dynamic positioning wanders from its set point under random wind and wave disturbances. A vessel holds a fixed position and heading with its thrusters while wind and wave gusts push it off course and the controller pushes it back in proportion to the error. Because gusts hit the side of the hull harder than the bow, the vessel wanders further sideways than forward, and the covariance of its excursions determines how much clearance the vessel needs and how stiff the controller must be.

The vessel's pose error is a twist, the small motion from the set point to where the vessel is, and a linear measurement of that error is a line, paired with it by `line & error`. Their spread is a covariance map taking each measurement line to the twist correlated with it, so `line & covariance(line)` is the variance of the measurement. The controller and any steady turn define a linear drift map on twists, which acts on the covariance from both sides—on the twists it returns and, through `dynamics.adjugate()`, on the lines it takes—so the steady spread where the controller removes uncertainty as fast as the gusts add it is found in a single linear solve.

In matrix notation the error reads as a coordinate vector $(v_x, v_y, \\omega)$, the covariance reads as a symmetric $3 \\times 3$ matrix $P$ evolving as $\\dot P = F P + P F^{\\mathsf T} + Q$, and the steady-state covariance solves the continuous Lyapunov equation $F P + P F^{\\mathsf T} + Q = 0$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
# A line measures a twist through the regressive product, so it is an antibivector.
Line = ga.gatype.antibivector()                     # Antibivector-
# A plane measures a displacement of a point the same way, so it is a vector.
Plane = ga.gatype.vector()                          # Vector+
Twist = ga.gatype.bivector()                        # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Covariance = ga.gatype((Twist, Line))
Dynamics = ga.gatype((Twist, Twist))
Spread = ga.gatype((Scalar, Plane, Plane))
Gusts = ga.gatype((Twist, Line))
# The commanded set point at the origin.
origin = mv.w.dual()                                # [] Point-


# --- math -----------------------------------------------------------------------------
def covariance(gusts: Gusts) -> Covariance:
    """The noise covariance added per second by white-noise gusts."""
    # White noise has unit variance along each axis of an orthonormal line frame, so each axis adds its gust dyad.
    frame = mv.antibivector(np.eye(3))                                              # [axes] Line-
    return (gusts(frame) * (gusts(frame) & Line)).sum(axis=0)                       # [] Twist+ <- Line-


def settled(dynamics: Dynamics, noise: Covariance) -> Covariance:
    """The steady-state covariance at which the controller removes uncertainty as fast as the gusts add it."""
    # Writing the covariance as a dyad t * (s & l) turns its rate of change into a linear map of both twists and the line.
    lyapunov = dynamics(Twist) * (Twist & Line) + Twist * (dynamics(Twist) & Line)  # [...] Twist+ <- (Twist+, Twist+, Line-)
    # Match the line input against -noise and solve for both twist inputs; Line.dual() feeds a line in as a twist.
    return lyapunov.lstsq(-noise)(Line.dual())                                      # [...] Twist+ <- Line-


def position_spread(covariance: Covariance) -> Spread:
    """The symmetric covariance form on planes measuring displacements of the set point, pulled back from the twist covariance."""
    # An error moves the set point by their commutator; its adjugate pulls a displacement-measuring plane back to a twist measurement.
    shift = Twist.commutator(origin)                                                # [] Point- <- Twist+
    readout = shift.adjugate()                                                      # [] Line- <- Plane+
    return readout & covariance(readout)                                            # [...] Scalar+ <- (Plane+, Plane+)
