"""Jeffery orbits: rigid ellipsoids tumbling in a shear flow, in the geometric algebra of
three-dimensional space.

A slow, sticky flow, such as honey sheared between two plates, moves each point at a velocity that
changes linearly across the flow: a map on offsets, `Gradient = Point <- Point`. A drop of the
fluid itself would both turn and stretch with it; a rigid particle can only turn, and the fluid
presses on it until it turns at the one rate that leaves no torque on it, found by Jeffery in 1922.

The particle is an ellipsoid, a stretch of the unit sphere along its own axes. Its shape is its dual
quadric, a map from planes to offsets, `Shape = Point <- Plane`: the unit sphere's, `Plane.dual()`, carried
by the stretch, the offsets forward and the planes back by its adjugate,
`stretch(Plane.dual()(stretch.adjugate()))`. Roundness is where the metric enters, and the surface is the
offsets with `shape.solve(offset) & offset == 1`.
Jeffery's spin, a bivector, satisfies one equation between maps:

    Point.commutator(spin)(shape) + shape(Plane.commutator(spin)) == gradient(shape) - shape(gradient.adjugate())

On the left, the shape's rate of change as the body turns at the spin: its offsets turn, and so do the
planes that read them. On the right, the gradient acting on the shape's offsets, less its adjugate acting on the
planes. Both sides are antisymmetric, so their joins with a free offset, `(Point & map(Plane)).trace(1, 2)`,
carry them whole, with no metric: a linear map from the spin to a join, and one solve. A sphere spins at
the flow's own rate, half the curl of the gradient. A rod lingers along the flow, then flips end over end,
and its axis comes back to where it started after a fixed time, on a closed path on the sphere of
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
Scalar = ga.gatype.scalar()                                # Scalar+
# Offsets, and their velocities, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                             # Antivector-
Bivector = ga.gatype.bivector()                            # Bivector+
Rotor = ga.gatype.rotor()                                  # Rotor+
# The planes that read offsets are vectors, with extrinsic orientation.
Plane = ga.gatype.vector()                                 # Vector+
# A stretch, carrying the unit sphere onto the body.
Stretch = ga.gatype((Point, Point))                        # Point- <- Point-
# The body's dual quadric.
Shape = ga.gatype((Point, Plane))                          # Point- <- Plane+
# A flow's gradient: the velocity per offset.
Gradient = ga.gatype((Point, Point))                       # Point- <- Point-

# The planes across the flow and across the gradient, and the vortex plane, the plane the fluid turns in.
axes = mv.basis()                                          # [3] Plane+


# --- math -----------------------------------------------------------------------------
def stretch(semi_axes: np.ndarray) -> Stretch:
    """Ellipsoids in their own frames: the unit sphere stretched by each semi-axis `[..., 3]` along
    its axis."""
    # Each point carried along each axis by its height above the axis's plane, times the semi-axis.
    return (axes.dual() * semi_axes * (axes & Point)).sum(axis=-1)   # [...] Point- <- Point-


def curl(gradient: Gradient) -> Bivector:
    """The curl of a flow's gradient: the bivector whose turning map is the gradient less its
    adjoint, twice the spin of the fluid. The reciprocal of a unit antivector is its reverse, and the
    commutator of two directions is the plane they span."""
    return Point.reverse().commutator(gradient(Point)).contract(1, 2)   # [...] Bivector+


def spin(shape: Shape, gradient: Gradient) -> Bivector:
    """The spin of a rigid ellipsoid of the given shape in a flow of the given gradient, free of
    torque: Jeffery's."""
    # The shape's rate of change as the body turns at a spin: its offsets turn, and the planes that read them.
    turning_rate = Point.commutator(Bivector)(shape) + shape(Plane.commutator(Bivector))   # [...] Point- <- (Plane+, Bivector+)
    # The gradient on the shape's offsets, less its adjugate on the planes.
    flow_rate = gradient(shape) - shape(gradient.adjugate())   # [...] Point- <- Plane+
    # Both sides are antisymmetric: their joins with a free offset carry them whole, a map on spins and a join.
    return (Point & turning_rate(Plane)).trace(1, 2).solve((Point & flow_rate(Plane)).trace(1, 2))   # [...] Bivector+


def placed(stretches: Stretch, rotors: Rotor) -> Shape:
    """The shape of each body turned by its rotor: the unit sphere's dual quadric, carried by the turned
    stretch."""
    stretch = rotors >> stretches(rotors << Point)         # [...] Point- <- Point-
    # The unit sphere is where the metric enters: each plane to the direction across it.
    sphere = Plane.dual()                                  # [] Point- <- Plane+
    return stretch(sphere(stretch.adjugate()))             # [...] Point- <- Plane+


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

