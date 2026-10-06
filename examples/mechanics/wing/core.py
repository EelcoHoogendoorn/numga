"""Potential flow past a wing: a velocity whose geometric derivative vanishes.

The velocity is a vector field. Its geometric derivative, the open vector times the velocity's
gradient map contracted, has a scalar part, the divergence, and a bivector part, the vorticity. Air
past a wing, taken as ideal, has neither: its velocity's derivative vanishes everywhere.

Past a cylinder in a uniform stream the velocity is a formula: the stream, the stream's image in the
cylinder, `radius**2 * r.inverse() * stream * r.inverse()` subtracted, and the swirl of a
circulation around it, itself a bivector. How the inverse of the position changes with a step gives
its gradient map. A step changes the velocity potential by `velocity | step`, how far the step goes
along the flow, and the stream function, a bivector, by `velocity ^ step`, how much flow crosses it:
the two parts of `velocity * step`. The circulation is the one that lets the flow leave the wing's
sharp trailing edge smoothly, the Kutta condition, `4 * pi * (stream ^ edge)`.

The Joukowski map, `p + (critical >> p.inverse())`, adds to each point its inverse reflected in the
critical direction. Its Jacobian turns and scales every step, keeping angles, except at the two
critical points, `critical` and its negative, where it vanishes and doubles angles instead. Circles
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
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Even = ga.gatype.even()
PotentialGradient = ga.gatype((Scalar, Vector))              # Scalar <- Vector
VelocityGradient = ga.gatype((Vector, Vector))               # Vector <- Vector


@dataclass(frozen=True)
class Wing:
    centre: Vector                                           # [] Vector, the cylinder's centre before the map
    scale: float                                             # the cylinder passes through scale * chord, the trailing edge
    chord: Vector                                            # [] Vector, unit, from the leading to the trailing edge


@dataclass(frozen=True)
class Flow:
    points: Vector                                           # [...] Vector, around the wing
    velocity: Vector                                         # [...] Vector
    potential_gradient: PotentialGradient                    # [...] Scalar <- Vector
    stream: Bivector                                         # [...] Bivector, constant along streamlines


# --- math -----------------------------------------------------------------------------
def pitched(wing: Wing, attack: float) -> Wing:
    """The wing turned nose up by the angle of attack."""
    turn = (mv.xy * (attack / 2)).exp()                                      # [] Rotor
    return Wing(turn >> wing.centre, wing.scale, turn >> wing.chord)


def circulation(wing: Wing, stream: Vector) -> Bivector:
    """The circulation that makes the trailing edge a stagnation point of the cylinder's flow."""
    edge = wing.scale * wing.chord - wing.centre                             # [] Vector
    return 4 * np.pi * (stream ^ edge)                                       # [] Bivector


def cylinder(wing: Wing, stream: Vector, plane: Vector) -> tuple[Vector, VelocityGradient, Bivector]:
    """The flow past the cylinder: its velocity, the velocity's gradient map, and the stream function."""
    offsets = plane - wing.centre                                            # [...] Vector
    edge = wing.scale * wing.chord - wing.centre                             # [] Vector
    radius_squared = edge | edge                                             # [] Scalar
    swirl = circulation(wing, stream)                                        # [] Bivector
    inverse = offsets.inverse()                                              # [...] Vector
    # The stream, less its image in the cylinder, and the swirl around it.
    velocity = stream - radius_squared * (inverse * stream * inverse) - swirl * inverse / (2 * np.pi)   # [...] Vector
    # A step changes the inverse by minus the step between two inverses, and the velocity with it.
    change = -(inverse * Vector * inverse)                                   # [...] Vector <- Vector
    gradient = -radius_squared * (change * stream * inverse + inverse * stream * change) - swirl * change / (2 * np.pi)
    # The stream function: the flow crossing from the centre's line along the stream, the image's,
    # and the swirl's, which grows with the logarithm of the distance.
    flux = (stream ^ offsets) + radius_squared * (inverse ^ stream) - swirl * (offsets | offsets).log() / (4 * np.pi)
    return velocity, gradient, flux


def joukowski(critical: Vector, plane: Vector) -> Vector:
    """Where the map with the given critical point sends each point."""
    return plane + (critical >> plane.inverse())                             # [...] Vector


def flow(wing: Wing, stream: Vector, plane: Vector, critical: Vector) -> Flow:
    """The flow past the image of the cylinder under the Joukowski map with the given critical
    point, at the images of points around the cylinder; the wing's critical point is its trailing
    edge, `wing.scale * wing.chord`."""
    velocity, _, flux = cylinder(wing, stream, plane)
    # How the map moves a small step: less the step turned and scaled by the even `critical * p.inverse()`.
    jacobian = Vector - ((critical * plane.inverse()) >> Vector)             # [...] Vector <- Vector
    # The potential keeps its value at a point's image: a step there is first undone by the Jacobian.
    potential_gradient = (velocity | Vector)(jacobian.solve(1 * Vector))     # [...] Scalar <- Vector
    at_wing = (Vector * potential_gradient(Vector)).contract(1, 2)           # [...] Vector
    return Flow(joukowski(critical, plane), at_wing, potential_gradient, flux)


def derivative(gradient: VelocityGradient) -> Even:
    """The velocity's derivative: its divergence, the scalar part, and its vorticity, the bivector part."""
    return (Vector * gradient(Vector)).contract(1, 2)                       # [...] Even


def lift(wing: Wing, stream: Vector, density: float) -> Vector:
    """The Kutta–Joukowski lift: the density times the circulation contracted with the stream."""
    return density * (circulation(wing, stream) | stream)                  # [] Vector


# --- plumbing -------------------------------------------------------------------------
def rings(wing: Wing, rings_count: int, angles: int, reach: float) -> Vector:
    """Points on circles around the cylinder's centre, counterclockwise, from its surface out to
    `reach` times its radius, half an angle step off the trailing edge."""
    edge = wing.scale * wing.chord - wing.centre                             # [] Vector
    growth = np.exp(np.linspace(0.0, np.log(reach), rings_count))
    turns = (mv.xy * (-(np.arange(angles) + 0.5) / angles * np.pi)).exp()    # [angles] Rotor
    return wing.centre + (turns >> edge)[None, :] * growth[:, None]         # [rings, angles] Vector
