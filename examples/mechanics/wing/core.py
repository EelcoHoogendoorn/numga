"""Potential flow past a wing: a velocity field that is the derivative of a potential.

The velocity is a vector field, and it is the derivative of a potential with two parts: a scalar,
the velocity potential, which rises along the flow, and a bivector, the stream function, which rises
across it and measures the fluid passing between two streamlines. Together they make one
multivector of even grade, `W`. Past a cylinder in a uniform stream `U`, `W` is the geometric product
of the stream and the position, `U * r`, plus the stream's image in the cylinder,
`radius**2 * r.inverse() * U`, plus the swirl of a circulation around it, itself a bivector. The
potential's gradient at a point is a map from a small step to the change in `W`, `Even <- Vector`.
Contracting it against an open vector gives its derivative, which vanishes: the flow has neither
divergence nor vorticity. The same contraction of its reverse is twice the velocity. The circulation
is the one that lets the flow leave the wing's sharp trailing edge smoothly, the Kutta condition,
`4 * pi * (stream ^ edge)`.

The Joukowski map, `p + (critical >> p.inverse())`, doubles how each point sees the two critical
points, `critical` and its negative: the angle at which it sees the segment between them and the
ratio of its distances to them, while moving the points twice as far out. Circles through both and
circles around each land on circles of the same kinds. The cylinder through `critical`, the trailing
edge, becomes the wing, the doubled angle there making the edge a cusp. Composing the potential's gradient with the inverse of the map's Jacobian carries it to
the wing, where its derivative is again zero and the velocity again half that of its reverse.

In the notation of complex analysis, the even multivectors of the plane read as complex numbers,
with `xy` as the imaginary unit and the chord as the real axis: a point `p` as `chord * p`, the
potential as the complex potential φ + iψ, the Joukowski map as z = ζ + c²/ζ, the vanishing
derivative as the Cauchy–Riemann equations, and the velocity as the conjugate of dW/dz.
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
PotentialGradient = ga.gatype((Even, Vector))                # Even <- Vector


@dataclass(frozen=True)
class Wing:
    centre: Vector                                           # [] Vector, the cylinder's centre before the map
    scale: float                                             # the cylinder passes through scale * chord, the trailing edge
    chord: Vector                                            # [] Vector, unit, from the leading to the trailing edge


@dataclass(frozen=True)
class Flow:
    points: Vector                                           # [...] Vector, around the wing
    velocity: Vector                                         # [...] Vector
    potential_gradient: PotentialGradient                    # [...] Even <- Vector
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


def flow(wing: Wing, stream: Vector, plane: Vector, critical: Vector) -> Flow:
    """The flow past the image of the cylinder under the Joukowski map with the given critical
    point, at the images of points around the cylinder; the wing's critical point is its trailing
    edge, `wing.scale * wing.chord`."""
    offsets = plane - wing.centre                                            # [...] Vector
    edge = wing.scale * wing.chord - wing.centre                             # [] Vector
    radius_squared = edge | edge                                             # [] Scalar
    swirl = circulation(wing, stream)                                        # [] Bivector
    # Each point inverted in the cylinder: the same direction, at the radius squared over its distance.
    inverted = radius_squared * offsets.inverse()                           # [...] Vector
    # The potential's gradient past the cylinder: the stream's, `stream * r`; its image's, the stream's
    # potential reversed at the inverted point, `inverted * stream`, which a step moves by the step
    # reflected in it; and the swirl's.
    uniform = stream * Vector                                                # [] Even <- Vector
    image = -(inverted >> Vector) * stream / radius_squared                  # [...] Even <- Vector
    turning = -swirl * offsets.inverse() * Vector / (2 * np.pi)              # [...] Even <- Vector
    # The stream function: the bivector parts of the stream's and the image's potentials, and the
    # swirl's, which grows with the logarithm of the distance.
    flux = (stream ^ offsets) + (inverted ^ stream) - swirl * (offsets | offsets).log() / (4 * np.pi)
    # The map, and how it moves a small step: less the step turned and scaled by the even
    # `critical * p.inverse()`. Undone, the Jacobian carries the gradient to the wing.
    points = joukowski(critical, plane)                                      # [...] Vector
    jacobian = Vector - ((critical * plane.inverse()) >> Vector)             # [...] Vector <- Vector
    gradient = (uniform + image + turning)(jacobian.solve(1 * Vector))       # [...] Even <- Vector
    return Flow(points, velocity(gradient), gradient, flux)


def joukowski(critical: Vector, plane: Vector) -> Vector:
    """Where the map with the given critical point sends each point."""
    return plane + (critical >> plane.inverse())                             # [...] Vector


def velocity(gradient: PotentialGradient) -> Vector:
    """Half the derivative of the potential's reverse."""
    return 0.5 * (Vector * gradient(Vector).reverse()).contract()           # [...] Vector


def derivative(gradient: PotentialGradient) -> Even:
    """The derivative of the potential, zero where the flow is that of a potential."""
    return (Vector * gradient(Vector)).contract()                           # [...] Even


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
