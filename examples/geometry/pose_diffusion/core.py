"""How far a station-keeping vessel wanders from its set point, in PGA2D.

A vessel on dynamic positioning holds a fixed position and heading with its thrusters. Wind and
wave gusts push it off; the controller pushes it back in proportion to the error. The vessel
never sits exactly on its set point. It moves about it at random, further sideways than forward,
because gusts hit the side of the hull harder than the bow. How far it wanders, and in which
directions, is a covariance. Knowing it tells you how much clearance the vessel needs and how
stiff the controller has to be.

The position error is a twist, the small motion from the set point to the actual pose. Its
covariance is a map from lines to twists. For a line l and a twist t, l & t is one measurement of
the error, for example its component across the line, and l & P(l) is the variance of that
measurement. The covariance grows from zero as gusts accumulate and levels off where the
controller balances them. This module computes that level both by integrating the growth over
time and by solving directly for the covariance at which growth stops.

For comparison, in matrix notation the growth is F P + P F^T + Q and the steady state solves the
continuous Lyapunov equation.
"""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
# A line measures a twist, so it is an antibivector.
Line = ga.gatype.antibivector()
Covariance = ga.gatype((Twist, Line))             # Twist <- Line: a measurement line to the correlated twist
Dynamics = ga.gatype((Twist, Twist))              # Twist <- Twist: rate of change of a position error
Readouts = ga.gatype((Line, Line))                # Line <- Line: the same rate of change, acting on lines
Spread = ga.gatype((Scalar, Line, Line))          # Scalar <- (Line, Line): covariance of two position measurements
Kicks = ga.gatype((Twist, Line))                  # Twist <- Line: white noise to a gust's push on the vessel
ORIGIN = mv.xy                                    # the set point; in PGA2D a point is a bivector


# --- math -----------------------------------------------------------------------------
def drift(rate: Twist, relaxation: float) -> Dynamics:
    """Rate of change of a position error.

    If the vessel turns on the spot, the error turns with it, which is the commutator with the
    turning rate. The controller reduces the error in proportion to its size; the relaxation rate
    is that proportion, per second.
    """
    return Twist.commutator(rate) - relaxation * Twist


def on_readouts(dynamics: Dynamics) -> Readouts:
    """The rate of change of an error, expressed as a map on measurement lines.

    Measuring the changed error dynamics(t) with a line l gives the same number as measuring t
    itself with the line on_readouts(l). The map is found by solving the incidence form
    Line & Twist against Line & dynamics. In matrix terms it is the transpose of the dynamics.
    """
    return (Line & Twist).solve(Line & dynamics)


def growth(dynamics: Dynamics, covariance: Covariance, noise: Covariance) -> Covariance:
    """Rate of change of the covariance.

    The first term applies the dynamics to the twists the covariance returns, the second applies
    them to the lines it takes as input, and the third is the covariance the gusts add per second.
    """
    return dynamics(covariance) + covariance(on_readouts(dynamics)) + noise


def covariance(kicks: Kicks) -> Covariance:
    """Covariance per second of the gusts.

    A gust is white noise mapped to a twist by the kick map. White noise has unit variance on each
    basis line and no correlation between them, so the covariance is the sum over the basis lines
    of the dyad formed by each line's image under the kick map.
    """
    basis = mv.basis()                                                        # [3] Line
    return (kicks(basis) * (kicks(basis) & Line)).sum(axis=0)


def step(dynamics: Dynamics, noise: Covariance, errors: Twist, covariance: Covariance,
         kicks: Twist, dt: float) -> tuple[Twist, Covariance]:
    """One explicit Euler step of length dt, for the errors and for their predicted covariance.

    The gusts passed in must already be scaled by the square root of dt: the variance of white
    noise over an interval grows with the length of the interval.
    """
    return errors + dynamics(errors) * dt + kicks, covariance + growth(dynamics, covariance, noise) * dt


def settled(dynamics: Dynamics, noise: Covariance) -> Covariance:
    """The covariance at which growth stops.

    Growth is linear in the covariance, so setting it to zero gives a linear equation whose
    unknown is a map. Write the covariance as a sum of dyads t * (s & l), with twists t and s and
    a line l, and leave all three open. Growth then becomes a map with three inputs, and lstsq
    matches its line input against the noise and solves for the coefficients on the two twist
    inputs. The solution is a map from twists to twists; it takes a line through the line's dual,
    the twist whose coefficients are the line's measurements of the basis twists.
    """
    # growth with the dyad open: the dynamics applied to t, and applied to s
    lyapunov = dynamics(Twist) * (Twist & Line) + Twist * (dynamics(Twist) & Line)   # Twist <- (Twist, Twist, Line)
    return lyapunov.lstsq(-noise)(Line.dual())


def position_spread(covariance: Covariance) -> Spread:
    """Covariance of measurements of the vessel's position.

    An error moves the set point, and the commutator gives that displacement per twist. Measuring
    the displacement with a line is the same as measuring the twist with a different line, found
    by solving the incidence form against the displacement. Pairing two such measurements through
    the covariance gives a symmetric form on lines. Its eigenpairs are the principal axes and
    variances of the position's uncertainty ellipse.
    """
    shift = Twist.commutator(ORIGIN)                                          # Point <- Twist: displacement of the set point
    readout = (Line & Twist).solve(Line & shift)                              # Line <- Line: position measurements as twist measurements
    return readout & covariance(readout)
