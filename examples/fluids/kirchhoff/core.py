"""Kirchhoff's equations: a solid ellipsoid moving through still, ideal fluid, in the projective
geometric algebra of three-dimensional space.

A body moving through water has to push the water aside, and the water it sets moving carries
kinetic energy of its own. In an ideal fluid, one without viscosity that cannot be compressed, that
energy depends only on the body's motion, quadratically, so the fluid acts as extra inertia carried
along with the body: its added mass. Unlike the body's own mass, the added mass depends on the
direction of motion: an elongated body moving end-on pushes little water aside, moving broadside a
lot. Its momentum then no longer points along its motion, and a body launched nearly end-on turns
broadside, the Munk moment that makes airships and submarines unstable without fins.

The body's shape is a map on directions, `Shape = Direction <- Direction`, with its squared
semi-axes along its own axes; the ellipsoids confocal with it are `shape + widening * Direction`,
every squared semi-axis grown by the same amount. The flow around an ellipsoid is built from that
confocal family: averaged over it, weighted by the body's volume over each confocal ellipsoid's,
the inverse of the family gives the fluid's response to shifts, a map on directions, and its
outermorphism the response to turns, a map on couples. Added to the body's own inertia, the result
is one map from rates to momenta, `Forque <- Twist`, whose quadratic form `rate & inertia(rate)` is
twice the kinetic energy of body and fluid together. With no force on either, the impulse, the
momentum of body and fluid, stays fixed in space, and the body moves at each moment at the rate
whose momentum is that impulse as the body sees it.

In the vector notation of fluid dynamics, with the body's velocity $\\mathbf U$, angular velocity
$\\boldsymbol\\omega$, impulse $\\mathbf P$ and angular impulse $\\boldsymbol\\Pi$ in its own frame,
Kirchhoff's equations read as $\\dot{\\mathbf P} = \\mathbf P \\times \\boldsymbol\\omega$ and
$\\dot{\\boldsymbol\\Pi} = \\boldsymbol\\Pi \\times \\boldsymbol\\omega + \\mathbf P \\times \\mathbf U$,
and Lamb's coefficients of the ellipsoid as
$\\alpha_0 = abc \\int_0^\\infty \\frac{d\\lambda}{(a^2 + \\lambda)\\Delta}$,
$\\Delta = \\sqrt{(a^2+\\lambda)(b^2+\\lambda)(c^2+\\lambda)}$, with the added mass
$\\frac{\\alpha_0}{2 - \\alpha_0}$ times the displaced fluid's along the first axis.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
Plane = ga.gatype.vector()                                 # Vector+
Point = ga.gatype.antivector()                             # Antivector-
# Offsets, and the velocities of points, are points at infinity: antivectors with intrinsic orientation.
Direction = Point.derive.containing("w")                   # Antivector-
Twist = ga.gatype.bivector()                               # Bivector+
Forque = ga.gatype.antibivector()                          # Antibivector-
# Couples are forques along a line at infinity: the momentum of a turn.
Couple = Forque.derive.containing("w")                     # Antibivector-
Motor = ga.gatype.rotor()                                  # Rotor+
# A stretch, carrying the unit sphere onto the body.
Stretch = ga.gatype((Direction, Direction))                # Direction- <- Direction-
# A shape: the squared semi-axes along the body's axes.
Shape = ga.gatype((Direction, Direction))                  # Direction- <- Direction-
# An inertia: the momentum per rate.
Inertia = ga.gatype((Forque, Twist))                       # Forque- <- Twist+

# The body's centre, the planes through it square to its axes, and the directions along those axes.
centre = mv.point([0, 0, 0])                               # [] Point-
axes = stack([mv.x, mv.y, mv.z])                           # [3] Plane+
directions = axes.dual()                                   # [3] Direction-
# The velocity of the centre under a rate, and the couple of a turn: half the sum, over the axes, of
# each direction joined with its own velocity. A shift moves no direction, so it has no couple.
centre_velocity = centre.commutator(Twist)                 # Direction- <- Twist+
turn_couple = (directions & directions.commutator(Twist)).sum(axis=-1) / 2   # Couple <- Twist+


# --- math -----------------------------------------------------------------------------
def stretch(semi_axes: np.ndarray) -> Stretch:
    """Ellipsoids in their own frames: the unit sphere stretched by each semi-axis `[..., 3]` along its
    axis. The shape is the stretch applied twice."""
    # Each direction carried along each axis by its part along it, times the semi-axis.
    return (directions * semi_axes * (axes & Direction)).sum(axis=-1)   # [...] Direction- <- Direction-


def inertia(shape: Shape, density: np.ndarray, fluid_density: np.ndarray, n_widenings: int) -> Inertia:
    """The inertia of a solid ellipsoid of each shape and the given density, together with the ideal
    fluid of the given density it moves through."""
    widenings, weights = quadrature(n_widenings)
    # The mean squared semi-axis, the trace over the axes, sets the scale of the widenings, and the body's
    # volume is a square root of the shape's determinant.
    scale = shape.trace() / len(axes)                      # [...] Scalar+
    volume = 4 / 3 * np.pi * shape.det().square_root()     # [...] Scalar+
    # The confocal ellipsoids, every squared semi-axis grown by the same widening, each weighed by the
    # body's volume over its own.
    confocal = shape[..., None] + (scale[..., None] * widenings) * Direction   # [..., widenings] Direction- <- Direction-
    weight = (shape.det()[..., None] / confocal.det()).square_root() * scale[..., None] * weights   # [..., widenings] Scalar+
    # Averaged over the confocal family, the inverse shape gives the response to a shift along each axis,
    # and its outermorphism the response to a turn in each plane. The first has trace two.
    shifting = (confocal.inverse() * weight).sum(axis=-1)  # [...] Direction- <- Direction-
    turning = (confocal.inverse().outermorphism(Couple) * weight).sum(axis=-1)   # [...] Couple <- Couple
    # On each couple, the outermorphism multiplies the squared semi-axes in its plane and `spread` adds them:
    # the mixed term of the outermorphism of the shape plus the identity, the shape acting on one direction of
    # the couple at a time. Their discriminant is the square of their difference, which vanishes for a turn
    # about an axis of symmetry: such a turn moves no fluid.
    areas = shape.outermorphism(Couple)                    # [...] Couple <- Couple
    spread = (Direction + shape).outermorphism(Couple) - Couple - areas   # [...] Couple <- Couple
    discriminant = spread(spread) - 4 * areas              # [...] Couple <- Couple
    # The fluid's mass under a shift and its inertia under a turn, per unit of the displaced fluid's mass.
    shift_mass = (2 * Direction - shifting).solve(shifting)   # [...] Direction- <- Direction-
    turn_mass = (2 * Couple - spread(turning)).solve(discriminant(turning)) / 5   # [...] Couple <- Couple
    # The solid's own: its mass whichever way it moves, and a fifth of it times the spread under a turn.
    masses = (volume * density) * Direction + (volume * fluid_density) * shift_mass   # [...] Direction- <- Direction-
    turns = (volume * density / 5) * spread + (volume * fluid_density) * turn_mass   # [...] Couple <- Couple
    # A shift's momentum is a line through the centre, a turn's a couple.
    return (centre & masses(centre_velocity)) + turns(turn_couple)   # [...] Forque- <- Twist+


def glide(motor: Motor, impulse: Forque, inertia: Inertia, dt: float, n_steps: int) -> Iterator[Motor]:
    """The body's placement from the start and after each step, with no force on body or fluid: the
    impulse stays fixed in space, and the body moves at the rate whose momentum is the impulse as the
    body sees it, taken at the midpoint of each step."""
    mobility = inertia.inverse()                           # [...] Twist+ <- Forque-
    yield motor
    for _ in range(n_steps):
        halfway = motor * (mobility(motor << impulse) * (-dt / 4)).exp()
        motor = motor * (mobility(halfway << impulse) * (-dt / 2)).exp()
        yield motor


# --- plumbing -------------------------------------------------------------------------
def quadrature(n_widenings: int) -> tuple[np.ndarray, np.ndarray]:
    """Widenings from zero to infinity in units of the mean squared semi-axis, and their weights `[widenings]`:
    Gauss–Legendre nodes in `t` on the unit interval, at widenings `1 / t**2 - 1`, so that the integrands
    fall off as powers of `t`."""
    nodes, weights = np.polynomial.legendre.leggauss(n_widenings)
    t = (nodes + 1) / 2
    return 1 / t**2 - 1, weights / t**3
