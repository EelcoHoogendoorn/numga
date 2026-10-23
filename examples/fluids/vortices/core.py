"""Vortices in a plane of fluid, the derivative of their flow, and their motion.

Each vortex turns the fluid around its centre, at a rate that falls off as one over the squared
distance beyond a small core: the vortex blob of Rosenhead and Moore, finite at its own centre. Its
spin is a bivector, the turn it gives the fluid, the circulation its size and its sign the sense of the
turn; a point moves with its commutator with the spin, so the sense comes from the vortices alone. A pattern of vortices may repeat at a set of offsets, its periodic copies; a pattern that
does not repeat has the zero offset alone. The velocity is a sum over the vortices and their copies,
and so is the velocity gradient at a point: a map on displacements, `Point <- Point`, the change in
velocity over a small step, written in closed form.

The gradient multiplied by `Point`, the space of all points, and contracted is the derivative of the
flow, `(Point * gradient(Point)).contract(1, 2)`: its scalar part is the divergence, zero for this flow, and
its bivector part the vorticity, the rate at which the fluid turns. The trace of the gradient applied
twice needs no metric and tells swirl from strain: `-0.5 * gradient(gradient).trace()` is positive
where the fluid turns faster than it stretches. Each vortex moves with the flow at its centre, to
which its own blob adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D

ga = VGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points, and displacements and velocities of points, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
# A vortex's spin: the turn it gives the fluid, its size the circulation.
Spin = ga.gatype.bivector()
Even = ga.gatype.even()
Gradient = ga.gatype((Point, Point))                         # Point <- Point


@dataclass(frozen=True)
class Vortices:
    centres: Point                                           # [..., vortices] Point
    spins: Spin                                              # [vortices] Spin
    core_radius: float
    copies: Point                                            # [copies] Point, offsets the pattern repeats at


# --- math -----------------------------------------------------------------------------
def velocity(vortices: Vortices, points: Point) -> Point:
    """The velocity the vortices and their copies induce at each point."""
    separations = _separations(vortices, points)                            # [..., vortices, copies] Point
    rate, _ = _profile(vortices, separations)                               # [..., vortices, copies] Spin
    # Each vortex moves the fluid along the separation turned a quarter turn by its spin.
    return separations.commutator(rate).sum(axis=-1).sum(axis=-1)           # [...] Point


def gradient(vortices: Vortices, points: Point) -> Gradient:
    """The velocity gradient at each point: the change in velocity for a small displacement."""
    separations = _separations(vortices, points)                            # [..., vortices, copies] Point
    rate, rate_change = _profile(vortices, separations)                     # [..., vortices, copies] Spin each
    # A displacement turns with the rate at the point, and changes the rate by its component along
    # the separation, which acts on the turned separation.
    turning = Point.commutator(rate)                                        # [..., vortices, copies] Point <- Point
    spreading = 2 * separations.commutator(rate_change) * (separations | Point)   # [..., vortices, copies] Point <- Point
    return (turning + spreading).sum(axis=-1).sum(axis=-1)                  # [...] Point <- Point


def derivative(gradients: Gradient) -> Even:
    """The derivative of the flow: the divergence plus the vorticity bivector."""
    return (Point * gradients(Point)).contract(1, 2)                        # [...] Even


def swirl(gradients: Gradient) -> Scalar:
    """How much faster the fluid turns than it stretches; positive inside a vortex core."""
    return -0.5 * gradients(gradients).trace()                              # [...] Scalar


def step(vortices: Vortices, dt: float) -> Vortices:
    """The vortices moved for a time `dt` by the classical fourth-order Runge–Kutta rule."""
    def moved(rate: Point, fraction: float) -> Vortices:
        return replace(vortices, centres=vortices.centres + rate * (fraction * dt))

    first = drift(vortices)                                                 # [..., vortices] Point
    second = drift(moved(first, 0.5))                                       # [..., vortices] Point
    third = drift(moved(second, 0.5))                                       # [..., vortices] Point
    fourth = drift(moved(third, 1.0))                                       # [..., vortices] Point
    return moved((first + 2 * second + 2 * third + fourth) / 6, 1.0)


def drift(vortices: Vortices) -> Point:
    """The velocity of each vortex: the flow at its centre."""
    return velocity(vortices, vortices.centres)                             # [..., vortices] Point


def _separations(vortices: Vortices, points: Point) -> Point:
    """From every vortex and copy to every point."""
    return points[..., None, None] - (vortices.centres[..., None] + vortices.copies)   # [..., vortices, copies] Point


def _profile(vortices: Vortices, separations: Point) -> tuple[Spin, Spin]:
    """The turn the fluid makes about each centre, and its change with the squared distance."""
    blob = (separations | separations) + vortices.core_radius**2              # [..., vortices, copies] Scalar
    rate = vortices.spins[:, None] / (2 * np.pi) / blob                      # [..., vortices, copies] Spin
    return rate, -rate / blob


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Point:
    """Displacements on a grid centred on the origin, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x * x[None, :] + mv.y * y[:, None]).dual()                   # [rows, columns] Point
