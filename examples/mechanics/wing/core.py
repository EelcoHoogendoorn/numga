"""Potential flow past a wing: a velocity whose geometric derivative vanishes.

The velocity is a field of displacement rates, antivectors like the points they move. Its geometric
derivative, the velocity's gradient map multiplied by `Point`, the space of all points, and contracted, has a scalar part, the divergence, and a bivector part, the vorticity. Air
past a wing, taken as ideal, has neither: its velocity's derivative vanishes everywhere.

Past a cylinder in a uniform stream the velocity is a formula: the stream, the stream's image in the
cylinder, `radius**2 * r.inverse() * stream * r.inverse()` subtracted, and the swirl of a
circulation around it, itself a bivector. How the inverse of the position changes with a step gives
its gradient map. A step changes the velocity potential by `velocity | step`, how far the step goes
along the flow, and the stream function, a bivector, by `velocity ^ step`, how much flow crosses it:
the two parts of `velocity * step`. The circulation is the one that lets the flow leave the wing's
sharp trailing edge smoothly, the Kutta condition, `4 * pi * (stream ^ edge)`.

The Joukowski map, `p + critical * p.inverse() * critical`, adds to each point its inverse sandwiched
by the critical point: reflected in the line along it, and scaled by its squared distance. Its Jacobian turns and scales every step, keeping
angles, except at the two critical points, `critical` and its negative, where it vanishes and doubles angles instead. Circles
through both critical points and circles around each land on circles of the same kinds around their
images. The cylinder through `critical`, the trailing edge, becomes the wing, its smooth edge folded
there into a cusp. The potential and the stream function keep their values at a point's image, so
the potential's gradient map, composed with the inverse of the Jacobian, is the wing's, and its
derivative is the velocity there.

In the notation of complex analysis the flow is the complex potential of the flow past a cylinder,
carried by Joukowski's map z = ζ + c²/ζ: analytic by the equations of Cauchy and Riemann, its
derivative the conjugate of the velocity, and its lift the Kutta–Joukowski theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Points, and displacements and velocities of points, are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Even = ga.gatype.even()
PotentialGradient = ga.gatype((Scalar, Point))               # Scalar <- Point
VelocityGradient = ga.gatype((Point, Point))                 # Point <- Point


@dataclass(frozen=True)
class Wing:
    centre: Point                                           # [] Point, the cylinder's centre before the map
    scale: float                                             # the cylinder passes through scale * chord, the trailing edge
    chord: Point                                            # [] Point, unit, from the leading to the trailing edge


# --- math -----------------------------------------------------------------------------
def circulation(wing: Wing, stream: Point) -> Bivector:
    """The circulation that makes the trailing edge a stagnation point of the cylinder's flow."""
    edge = wing.scale * wing.chord - wing.centre                             # [] Point
    return 4 * np.pi * (stream ^ edge)                                       # [] Bivector


def cylinder(wing: Wing, stream: Point, plane: Point) -> tuple[Point, VelocityGradient, Bivector]:
    """The flow past the cylinder: its velocity, the velocity's gradient map, and the stream function."""
    offsets = plane - wing.centre                                            # [...] Point
    edge = wing.scale * wing.chord - wing.centre                             # [] Point
    radius_squared = edge | edge                                             # [] Scalar
    swirl = circulation(wing, stream)                                        # [] Bivector
    inverse = offsets.inverse()                                              # [...] Point
    # The stream, less its image in the cylinder, and the swirl around it.
    velocity = stream - radius_squared * (inverse * stream * inverse) - swirl * inverse / (2 * np.pi)   # [...] Point
    # A step changes the inverse by minus the step between two inverses, and the velocity with it.
    change = -(inverse * Point * inverse)                                    # [...] Point <- Point
    gradient = -radius_squared * (change * stream * inverse + inverse * stream * change) - swirl * change / (2 * np.pi)
    # The stream function: the flow crossing from the centre's line along the stream, the image's,
    # and the swirl's, which grows with the logarithm of the distance.
    flux = (stream ^ offsets) + radius_squared * (inverse ^ stream) - swirl * (offsets | offsets).log() / (4 * np.pi)
    return velocity, gradient, flux


def joukowski(critical: Point, plane: Point) -> Point:
    """Where the map with the given critical point sends each point."""
    return plane + critical * plane.inverse() * critical                     # [...] Point


def flow(wing: Wing, stream: Point, plane: Point, critical: Point
         ) -> tuple[Point, Point, PotentialGradient, Bivector]:
    """The flow past the image of the cylinder under the Joukowski map with the given critical
    point, at the images of points around the cylinder: the images, the velocity there, the
    potential's gradient map, and the stream function, constant along streamlines. The wing's
    critical point is its trailing edge, `wing.scale * wing.chord`."""
    velocity, _, flux = cylinder(wing, stream, plane)
    # How the map moves a small step: less the step turned and scaled by the even `critical * p.inverse()`.
    jacobian = Point - ((critical * plane.inverse()) >> Point)               # [...] Point <- Point
    # The potential keeps its value at a point's image: a step there is first undone by the Jacobian.
    potential_gradient = (velocity | Point)(jacobian.inverse())              # [...] Scalar <- Point
    at_wing = (Point * potential_gradient(Point)).contract(1, 2)             # [...] Point
    return joukowski(critical, plane), at_wing, potential_gradient, flux


def derivative(gradient: VelocityGradient) -> Even:
    """The velocity's derivative: its divergence, the scalar part, and its vorticity, the bivector part."""
    return (Point * gradient(Point)).contract(1, 2)                         # [...] Even


def lift(wing: Wing, stream: Point, density: float) -> Point:
    """The Kutta–Joukowski lift: the density times the circulation contracted with the stream."""
    return density * (circulation(wing, stream) | stream)                  # [] Point


# --- plumbing -------------------------------------------------------------------------
def rings(wing: Wing, rings_count: int, angles: int, reach: float) -> Point:
    """Points on circles around the cylinder's centre, counterclockwise, from its surface out to
    `reach` times its radius, half an angle step off the trailing edge."""
    edge = wing.scale * wing.chord - wing.centre                             # [] Point
    growth = np.exp(np.linspace(0.0, np.log(reach), rings_count))
    turns = (mv.xy * (-(np.arange(angles) + 0.5) / angles * np.pi)).exp()    # [angles] Rotor
    return wing.centre + (turns >> edge)[None, :] * growth[:, None]         # [rings, angles] Point
