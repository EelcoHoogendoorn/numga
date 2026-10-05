"""Vortices in a plane of fluid, the derivative of their flow, and their motion.

Each vortex turns the fluid around its centre in the plane `xy`, at a rate that falls off as one over
the squared distance beyond a small core: the vortex blob of Rosenhead and Moore, finite at its own
centre. A pattern of vortices may repeat at a set of offsets, its periodic copies; a pattern that
does not repeat has the zero offset alone. The velocity is a sum over the vortices and their copies,
and so is the velocity gradient at a point: a map on displacements, `Vector <- Vector`, the change in
velocity over a small step, written in closed form.

Contracting the gradient against an open vector gives the derivative of the flow,
`(Vector * gradient(Vector)).contract()`: its scalar part is the divergence, zero for this flow, and
its bivector part the vorticity, the rate at which the fluid turns. The trace of the gradient applied
twice needs no metric and tells swirl from strain: `-0.5 * gradient(gradient).trace()` is positive
where the fluid turns faster than it stretches. Each vortex moves with the flow at its centre, to
which its own blob adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Even = ga.gatype.even()
Gradient = ga.gatype((Vector, Vector))                       # Vector <- Vector
PLANE = mv.xy


@dataclass(frozen=True)
class Vortices:
    centres: Vector                                          # [..., vortices] Vector
    circulations: np.ndarray                                 # [vortices] circulation of each
    core_radius: float
    copies: Vector                                           # [copies] Vector, offsets the pattern repeats at


# --- math -----------------------------------------------------------------------------
def velocity(vortices: Vortices, points: Vector) -> Vector:
    """The velocity the vortices and their copies induce at each point."""
    separations = _separations(vortices, points)                            # [..., vortices, copies] Vector
    rate, _ = _profile(vortices, separations)                               # [..., vortices, copies] Scalar
    # Each vortex moves the fluid along the separation turned a quarter turn in the plane.
    return (rate * (separations | PLANE)).sum(axis=-1).sum(axis=-1)         # [...] Vector


def gradient(vortices: Vortices, points: Vector) -> Gradient:
    """The velocity gradient at each point: the change in velocity for a small displacement."""
    separations = _separations(vortices, points)                            # [..., vortices, copies] Vector
    rate, rate_change = _profile(vortices, separations)                     # [..., vortices, copies] Scalar each
    # A displacement turns with the rate at the point, and changes the rate by its component along
    # the separation, which acts on the turned separation.
    turning = rate * (Vector | PLANE)                                        # [..., vortices, copies] Vector <- Vector
    spreading = 2 * rate_change * (separations | PLANE) * (separations | Vector)    # [..., vortices, copies] Vector <- Vector
    return (turning + spreading).sum(axis=-1).sum(axis=-1)                  # [...] Vector <- Vector


def derivative(gradients: Gradient) -> Even:
    """The derivative of the flow: the divergence plus the vorticity bivector."""
    return (Vector * gradients(Vector)).contract()                          # [...] Even


def swirl(gradients: Gradient) -> Scalar:
    """How much faster the fluid turns than it stretches; positive inside a vortex core."""
    return -0.5 * gradients(gradients).trace()                              # [...] Scalar


def step(vortices: Vortices, dt: float) -> Vortices:
    """The vortices moved for a time `dt` by the classical fourth-order Runge–Kutta rule."""
    def moved(rate: Vector, fraction: float) -> Vortices:
        return replace(vortices, centres=vortices.centres + rate * (fraction * dt))

    first = drift(vortices)                                                 # [..., vortices] Vector
    second = drift(moved(first, 0.5))                                       # [..., vortices] Vector
    third = drift(moved(second, 0.5))                                       # [..., vortices] Vector
    fourth = drift(moved(third, 1.0))                                       # [..., vortices] Vector
    return moved((first + 2 * second + 2 * third + fourth) / 6, 1.0)


def drift(vortices: Vortices) -> Vector:
    """The velocity of each vortex: the flow at its centre."""
    return velocity(vortices, vortices.centres)                             # [..., vortices] Vector


def _separations(vortices: Vortices, points: Vector) -> Vector:
    """From every vortex and copy to every point."""
    return points[..., None, None] - (vortices.centres[..., None] + vortices.copies)   # [..., vortices, copies] Vector


def _profile(vortices: Vortices, separations: Vector) -> tuple[Scalar, Scalar]:
    """The rate at which the fluid turns about each centre, and its change with the squared
    distance."""
    blob = (separations | separations) + vortices.core_radius**2              # [..., vortices, copies] Scalar
    rate = vortices.circulations[:, None] / (2 * np.pi) / blob               # [..., vortices, copies] Scalar
    return rate, -rate / blob


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Vector:
    """Displacements on a grid centred on the origin, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return mv.x * x[None, :] + mv.y * y[:, None]                            # [rows, columns] Vector
