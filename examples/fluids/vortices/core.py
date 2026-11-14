"""Vortices in a plane of fluid, the derivative of their flow, and their motion.

Each vortex turns the fluid around its centre at a rate that falls off as the inverse squared
distance beyond a small core, modelled as the vortex blob of Rosenhead and Moore so the flow stays
finite at the centre. Circulation is a bivector whose magnitude gives the vortex strength and whose
sign gives the sense of turning. A fluid parcel moves with the commutator of its separation and the
rate set by the circulation, so mirror-reflected vortices with reversed circulations move as the
mirror image of the flow. Periodic patterns sum over an array of offset copies, while an isolated
group uses the zero offset alone. Both the velocity and the velocity gradient `Point <- Point` sum
over the vortices and their copies in closed form.

Multiplying the gradient by `Point` and contracting forms the geometric derivative of the flow,
`(Point * gradient(Point)).contract(1, 2)`, whose scalar part is the divergence (zero here) and
whose bivector part is the vorticity. Half the negated trace of the squared gradient,
`-0.5 * gradient(gradient).trace()`, requires no metric and weighs swirl against strain, staying
positive where turning outpaces stretching. Each vortex drifts with the induced velocity at its own
centre, where its own blob contributes zero.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D

ga = VGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                 # Scalar+
# Points, displacements, and velocities are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                                              # Antivector-
# Circulation and fluid turning rates are bivectors with extrinsic orientation.
Circulation = ga.gatype.bivector()                                          # Bivector+
Rate = ga.gatype.bivector()                                                 # Bivector+
Even = ga.gatype.even()                                                     # Even+
Gradient = ga.gatype((Point, Point))                                        # Point- <- Point-


# --- math -----------------------------------------------------------------------------
def velocity(centres: Point, circulations: Circulation, core_radius: float, copies: Point, points: Point) -> Point:
    """The fluid velocity induced at each point by the vortices and their periodic copies."""
    offsets = separations(centres, copies, points)                          # [...] Point-
    rates, _ = blob_profile(circulations, core_radius, offsets)             # [...] Rate+
    # Each vortex carries the fluid along the separation turned a quarter turn by its local rate.
    return offsets.commutator(rates).sum(axis=-1).sum(axis=-1)              # [...] Point-


def gradient(centres: Point, circulations: Circulation, core_radius: float, copies: Point, points: Point) -> Gradient:
    """The spatial velocity gradient mapping a small displacement to the resulting velocity change."""
    offsets = separations(centres, copies, points)                          # [...] Point-
    rates, rate_slopes = blob_profile(circulations, core_radius, offsets)   # [...] Rate+, Rate+
    # A displacement turns with the local rate and shifts the rate through its radial component.
    turning = Point.commutator(rates)                                       # [...] Point- <- Point-
    radial_shear = 2 * offsets.commutator(rate_slopes) * (offsets | Point)  # [...] Point- <- Point-
    return (turning + radial_shear).sum(axis=-1).sum(axis=-1)               # [...] Point- <- Point-


def derivative(gradients: Gradient) -> Even:
    """The geometric derivative of the flow, combining divergence and vorticity."""
    return (Point * gradients(Point)).contract(1, 2)                        # [...] Even+


def swirl(gradients: Gradient) -> Scalar:
    """The excess of turning squared over strain squared, positive inside vortex cores."""
    return -0.5 * gradients(gradients).trace()                              # [...] Scalar+


def drift(centres: Point, circulations: Circulation, core_radius: float, copies: Point) -> Point:
    """The induced velocity of each vortex at its own centre."""
    return velocity(centres, circulations, core_radius, copies, centres)    # [...] Point-


def step(centres: Point, circulations: Circulation, core_radius: float, copies: Point, dt: float) -> Point:
    """The vortex centres advanced over one time step `dt` by fourth-order Runge-Kutta integration."""
    first = drift(centres, circulations, core_radius, copies)               # [...] Point-
    second = drift(centres + first * (0.5 * dt), circulations, core_radius, copies)   # [...] Point-
    third = drift(centres + second * (0.5 * dt), circulations, core_radius, copies)   # [...] Point-
    fourth = drift(centres + third * dt, circulations, core_radius, copies)           # [...] Point-
    return centres + (first + 2 * second + 2 * third + fourth) * (dt / 6)   # [...] Point-


def trajectory(
    centres: Point,
    circulations: Circulation,
    core_radius: float,
    copies: Point,
    n_frames: int,
    n_substeps: int,
    dt: float,
) -> Iterator[Point]:
    """Yield the vortex centres across `n_frames` frames separated by `n_substeps` steps."""
    for _ in range(n_frames):
        yield centres
        for _ in range(n_substeps):
            centres = step(centres, circulations, core_radius, copies, dt)


def separations(centres: Point, copies: Point, points: Point) -> Point:
    """The displacements from every vortex and periodic copy to every evaluation point."""
    return points[..., None, None] - (centres[..., None] + copies)          # [...] Point-


def blob_profile(circulations: Circulation, core_radius: float, offsets: Point) -> tuple[Rate, Rate]:
    """The Rosenhead-Moore turning rate and its derivative with respect to squared distance."""
    blob = (offsets | offsets) + core_radius**2                             # [...] Scalar+
    rates = circulations[:, None] / (2 * np.pi) / blob                      # [...] Rate+
    return rates, -rates / blob
