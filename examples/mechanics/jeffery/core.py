"""Jeffery orbits: rigid ellipsoids tumbling in a shear flow, in the geometric algebra of
three-dimensional space.

A slow, sticky flow, such as honey sheared between two plates, moves each point at a velocity that
changes linearly across the flow: a map on offsets, `Gradient = Point <- Point`. A drop of the
fluid itself would both turn and stretch with it; a rigid particle can only turn, and the fluid
presses on it until it turns at the one rate that leaves no torque on it, found by Jeffery in 1922.

The particle is an ellipsoid, a stretch of the unit sphere along its own axes, and its shape is the
stretch applied twice: a map `Shape = Point <- Point` whose surface is the offsets `x` with
`~x | shape.solve(x) == 1`. Jeffery's spin, a bivector, makes the shape's product with the particle's
turning map, from both sides, equal to the gradient's product with the shape less the shape's with
the gradient's adjoint:

    shape(turning) + turning(shape) == gradient(shape) - shape(gradient.adjoint())

with `turning = Point.commutator(spin)`, the velocity each offset has under the spin. Both sides are
antisymmetric maps, so their curls, `Point.reverse().commutator(map(Point)).contract(1, 2)`, carry them
whole: a
linear map from the spin's bivector to a bivector, and one solve. A sphere spins at the flow's own
rate, half the curl of the gradient. A rod lingers along the flow, then flips end over end, and its
axis comes back to where it started after a fixed time, on a closed path on the sphere of
directions: a Jeffery orbit. An ellipsoid with three different axes never quite repeats.

In the index notation of suspension mechanics, in the particle's principal frame, the relative
spin reads as $\\omega_{ij} - \\Omega_{ij} = \\frac{a_i^2 - a_j^2}{a_i^2 + a_j^2}\\,E_{ij}$, with
$\\Omega$ the vorticity and $E$ the strain rate of the flow.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Offsets, and their velocities, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Stretch = ga.gatype((Point, Point))                        # Point <- Point: the unit sphere onto the body
Shape = ga.gatype((Point, Point))                          # Point <- Point: the body's quadric
Gradient = ga.gatype((Point, Point))                       # Point <- Point: velocity per offset

# The planes across the flow, the gradient and the vorticity.
axes = mv.basis()                                          # [3] Plane


# --- math -----------------------------------------------------------------------------
def stretch(semi_axes: np.ndarray) -> Stretch:
    """Ellipsoids in their own frames: the unit sphere stretched by each semi-axis `[..., 3]` along
    its axis."""
    # Each point carried along each axis by its height above the axis's plane, times the semi-axis.
    return (axes.dual() * semi_axes * (axes & Point)).sum(axis=-1)            # [...] Stretch


def curl(gradient: Gradient) -> Bivector:
    """The curl of a flow's gradient: the bivector whose turning map is the gradient less its
    adjoint, twice the spin of the fluid. The reciprocal of a unit antivector is its reverse, and the
    commutator of two directions is the plane they span."""
    return Point.reverse().commutator(gradient(Point)).contract(1, 2)        # [...] Bivector


def spin(shape: Shape, gradient: Gradient) -> Bivector:
    """The spin of a rigid ellipsoid of the given shape in a flow of the given gradient, free of
    torque: Jeffery's."""
    turning = Point.commutator(Bivector)                                      # Point <- (Point, Bivector)
    left = shape(turning) + turning(shape)                                    # [...] Point <- (Point, Bivector)
    right = gradient(shape) - shape(gradient.adjoint())                       # [...] Point <- Point
    # Both sides are antisymmetric, so their curls carry them whole: a map on spins, and a bivector.
    return curl(left).solve(curl(right))                                      # [...] Bivector


def placed(stretches: Stretch, rotors: Rotor) -> Shape:
    """The shape of each body turned by its rotor."""
    shape = stretches(stretches)                                              # [...] Shape: the stretch, twice
    return rotors >> shape(rotors << Point)                                   # [...] Shape


def step(rotors: Rotor, stretches: Stretch, gradient: Gradient, dt: float) -> Rotor:
    """The bodies turned for a time `dt` by their spin, taken at the midpoint of the step."""
    halfway = (spin(placed(stretches, rotors), gradient) * (-dt / 4)).exp() * rotors
    return (spin(placed(stretches, halfway), gradient) * (-dt / 2)).exp() * rotors


def tumble(rotors: Rotor, stretches: Stretch, gradient: Gradient, dt: float, steps: int) -> Iterator[Rotor]:
    """The bodies' orientations, from the start and after each of the steps."""
    yield rotors
    for _ in range(steps):
        rotors = step(rotors, stretches, gradient, dt)
        yield rotors

