r"""Potential flow past a wing: a velocity field whose geometric derivative vanishes.

Points, and the displacement rates that move them, are modelled as antivectors. Multiplying the velocity's
gradient map by `Point` and contracting gives the velocity's geometric derivative: its scalar part is the
divergence and its bivector part is the vorticity. For an incompressible, inviscid flow past a wing, that
derivative vanishes at every point in the fluid.

Past a cylinder in a uniform stream, the velocity adds three parts: the uniform stream, minus its reflection
in the cylinder `radius_squared * (inverse * stream * inverse)`, and the swirl of a bivector circulation
around the cylinder. Along a small step, the velocity potential changes by `velocity | step` and the bivector
stream function changes by `velocity ^ step`, the two parts of `velocity * step`. Imposing stagnation at the
trailing edge sets the Kutta circulation `4 * pi * (stream ^ edge)`, and joining that circulation with the
stream gives the Kutta–Joukowski lift `-density * (swirl & stream)`.

The Joukowski map `positions + critical * positions.inverse() * critical` adds to each point its inverse
sandwiched by a critical point. Its Jacobian `Point - ((critical * positions.inverse()) >> Point)` turns and
scales steps everywhere except at `critical` and `-critical`, where it vanishes and doubles angles instead,
folding the cylinder through `critical` into a sharp trailing edge. Carrying the velocity potential and stream
function along the map pulls the cylinder's potential gradient back through `jacobian.inverse()` and contracts
it with `Point` to yield the flow past the wing.

In complex-variable potential-flow notation, the plane is identified with $z = x + iy$ in a chosen coordinate
frame, the cylinder potential $W(\zeta) = U e^{-i\alpha}(\zeta - \zeta_0) + U e^{i\alpha} R^2/(\zeta - \zeta_0)
+ \frac{i\Gamma}{2\pi}\ln(\zeta - \zeta_0)$ with clockwise circulation $\Gamma = 4\pi U R \sin(\alpha + \beta)$
is mapped by $z = \zeta + c^2/\zeta$, the velocity components $(u, v)$ are recovered from the conjugate
derivative $u - iv = (dW/d\zeta)/(1 - c^2/\zeta^2)$, and the Kutta–Joukowski force is written as
$F_x + i F_y = i\rho U e^{i\alpha}\Gamma$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA2D

ga = VGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
# Points, displacements, and velocities of points are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()                                               # Antivector-
# A force on the wing pairs with a displacement of the wing via `force & displacement` to give work.
Force = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Even = ga.gatype.even()                                                      # Even+
Rotor = ga.gatype.rotor()                                                    # Rotor+
PotentialGradient = ga.gatype((Scalar, Point))                               # Scalar+ <- Point-
VelocityGradient = ga.gatype((Point, Point))                                 # Point- <- Point-


# --- math -----------------------------------------------------------------------------
def kutta_circulation(centre: Point, trailing_edge: Point, stream: Point) -> Bivector:
    """The circulation that makes the trailing edge a stagnation point of the cylinder's flow."""
    edge = trailing_edge - centre                                            # [] Point-
    return 4 * np.pi * (stream ^ edge)                                       # [] Bivector+


def cylinder_flow(centre: Point, trailing_edge: Point, swirl: Bivector, stream: Point,
                  positions: Point) -> tuple[Point, VelocityGradient, Bivector]:
    """The velocity, velocity gradient map, and stream function past the cylinder through `trailing_edge`."""
    offsets = positions - centre                                             # [...] Point-
    edge = trailing_edge - centre                                            # [] Point-
    radius_squared = edge.squared()                                          # [] Scalar+
    inverse = offsets.inverse()                                              # [...] Point-
    # The uniform stream, minus its reflection in the cylinder, plus the circulation swirl around it.
    velocity = stream - radius_squared * (inverse * stream * inverse) - swirl * inverse / (2 * np.pi)   # [...] Point-
    # A small step changes the position inverse by minus the step sandwiched between two inverses.
    inverse_gradient = -(inverse * Point * inverse)                          # [...] Point- <- Point-
    gradient = (                                                             # [...] Point- <- Point-
        -radius_squared * (inverse_gradient * stream * inverse + inverse * stream * inverse_gradient)
        - swirl * inverse_gradient / (2 * np.pi)
    )
    # Stream function: net flow crossing from the centre's streamline, its image, and the logarithmic swirl.
    flux = (stream ^ offsets) + radius_squared * (inverse ^ stream) - swirl * offsets.squared().log() / (4 * np.pi)   # [...] Bivector+
    return velocity, gradient, flux


def joukowski_image(critical: Point, positions: Point) -> Point:
    """Image points under the Joukowski map with critical points at `critical` and `-critical`."""
    return positions + critical * positions.inverse() * critical             # [...] Point-


def wing_flow(centre: Point, trailing_edge: Point, swirl: Bivector, stream: Point,
              positions: Point, critical: Point) -> tuple[Point, Point, PotentialGradient, Bivector]:
    """Image points, wing velocity, potential gradient map, and stream function under the Joukowski map."""
    velocity, _, flux = cylinder_flow(centre, trailing_edge, swirl, stream, positions)
    # How the map moves a small step: the step minus the step turned and scaled by `critical * positions.inverse()`.
    jacobian = Point - ((critical * positions.inverse()) >> Point)           # [...] Point- <- Point-
    # Because the potential keeps its value at each image point, a step there pulls back through `jacobian.inverse()`.
    potential_gradient = (velocity | Point)(jacobian.inverse())              # [...] Scalar+ <- Point-
    wing_velocity = (Point * potential_gradient(Point)).contract(1, 2)       # [...] Point-
    return joukowski_image(critical, positions), wing_velocity, potential_gradient, flux


def geometric_derivative(gradient: VelocityGradient) -> Even:
    """The velocity's geometric derivative: scalar divergence plus bivector vorticity."""
    return (Point * gradient(Point)).contract(1, 2)                          # [...] Even+


def kutta_lift(swirl: Bivector, stream: Point, density: float) -> Force:
    """The Kutta–Joukowski lift force from the bound circulation in the uniform stream."""
    # Directed across the stream toward the faster, lower-pressure side where circulation runs with the stream.
    return -density * (swirl & stream)                                       # [] Force+


# --- plumbing -------------------------------------------------------------------------
def cylinder_rings(centre: Point, trailing_edge: Point, n_rings: int, n_angles: int, outer_distance: float) -> Point:
    """Concentric sample rings around `centre` out to `outer_distance` radii, offset half a step from `trailing_edge`."""
    edge = trailing_edge - centre                                            # [] Point-
    growth = np.exp(np.linspace(0, np.log(outer_distance), n_rings))
    turns = (mv.xy * (-(np.arange(n_angles) + 0.5) / n_angles * np.pi)).exp()   # [n_angles] Rotor+
    return centre + (turns >> edge)[None, :] * growth[:, None]               # [n_rings, n_angles] Point-
