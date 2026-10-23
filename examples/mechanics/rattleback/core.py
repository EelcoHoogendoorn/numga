"""A rattleback: a body with an ellipsoidal underside rolling without slipping on a table, which turns
rocking into spin and spin of one sense into rocking and back, in PGA3D.

Its points and directions are antivectors, and its underside a map from planes to directions, the support
map: the table's plane through the shell's centre, read in the body, goes to the direction from that
centre to the point that touches the table, `-shape(normal) / height`, with the height the pairing
`normal & shape(normal)` under its square root. The body turns about its centre of mass, the origin, and
the shell's centre sits off it by `centre`, so the touching point lies at `centre + support` from the
centre of mass. Rolling holds that point still: a spin of the body about its centre of mass becomes the
rate that turns it about the contact, the spin carried by the translation to the contact, the product
`contact * origin.inverse()`. The adjugate of that rolling map pulls forques back onto spins and does so
without the contact's reaction, which does no work under any rolling motion, so one solve gives the
body's angular acceleration with the reaction eliminated.

As the body rocks, the contact moves across the shell. Differentiating the rolling map contributes to
the acceleration even at a fixed spin; this contribution and the rotation of the body's momentum remain
in the balance being solved.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

from numga import Algebra, concatenate
from numga.backend.context import Context

ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()
# Planes through the origin; points and directions are antivectors, intrinsic.
Plane = ga.gatype.vector().derive.nondegenerate
Point = ga.gatype.antivector()
Direction = Point.derive.degenerate
Spin = ga.gatype.bivector().derive.nondegenerate
Rate = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Forque = ga.gatype.antibivector()
AngularMomentum = Forque.derive.degenerate
Shape = ga.gatype((Direction, Plane))                    # Direction <- Plane: the support map
Stretch = ga.gatype((Direction, Direction))              # Direction <- Direction: the unit sphere onto the shell
Rolling = ga.gatype((Rate, Spin))
Inertia = ga.gatype((Forque, Rate))
# Planes anywhere, extrinsic; a quadric pairs each point with its plane, negative inside and zero on the surface,
# where the plane is the tangent plane.
Cut = ga.gatype.vector()
Quadric = ga.gatype((Cut, Point))                       # Cut <- Point

infinity = mv.w
origin = infinity.dual()


# --- math

def contact_geometry(motor: Motor, shape: Shape, centre: Direction,
                     up: Direction) -> tuple[Plane, Direction, Scalar, Rolling]:
    """The table's plane in the body, the direction from the shell's centre to the lowest point, the shell
    centre's height, and the motions that keep the lowest point instantaneously at rest."""
    normal = (motor << up).dual_inverse()                         # Plane
    height = (normal & shape(normal)).square_root()               # Scalar
    support = -shape(normal) / height                             # Direction

    # A spin about the centre of mass carried to the contact by the translation there, the product of the
    # two points, turns the body about the contact.
    rolling = ((centre + support) * origin.inverse() / 2).exp() >> Spin   # Rate <- Spin
    return normal, support, height, rolling


def dynamics(motor: Motor, spin: Spin, shape: Shape, centre: Direction, inertia: Inertia,
             mass: float, gravity: Direction, up: Direction) -> tuple[Rate, Spin]:
    """Rolling velocity and angular acceleration, with the contact force eliminated."""
    normal, support, height, rolling = contact_geometry(motor, shape, centre, up)
    rate = rolling(spin)                                          # Rate

    # The touching point changes as the body rocks; differentiate its support geometry. The shell's centre
    # is fixed in the body, so only the support moves.
    normal_rate = spin.commutator(normal)                         # Plane
    contact_rate = -shape(normal_rate) / height - support * (normal & shape(normal_rate)) / height.squared()  # Direction
    rolling_change = (contact_rate * origin.inverse()).commutator(spin)   # Rate
    force = (origin & (motor << gravity)) * mass                  # Forque

    # The adjugate pairs forces with allowed rolling motions, so the resting contact does no work.
    pullback = rolling.adjugate()                                 # AngularMomentum <- Forque
    effective = pullback(inertia(rolling))                        # AngularMomentum <- Spin
    acceleration = effective.solve(pullback(
        force - inertia(rate).commutator(rate) - inertia(rolling_change)
    ))                                                           # Spin
    return rate, acceleration


def energy(motor: Motor, spin: Spin, shape: Shape, centre: Direction, inertia: Inertia,
           mass: float, gravity: Direction, up: Direction) -> Scalar:
    """Kinetic and gravitational energy of a rolling body."""
    _, _, _, rolling = contact_geometry(motor, shape, centre, up)
    rate = rolling(spin)                                          # Rate
    potential = -mass * (gravity.dual_inverse() & (motor >> origin))  # Scalar
    return (rate & inertia(rate)) / 2 + potential


# --- plumbing

class Body(NamedTuple):
    """The rattleback as built, in the frame of its centre of mass, by intersections of quadrics: a wooden solid
    inside its underside and its cuts and outside its balls, and the steel balls, each filling the pocket its
    complement cuts into the wood."""
    underside: Quadric                                        # [] the ellipsoid
    deck: Quadric                                             # [] the flat top
    chamfers: Quadric                                         # [sides] cuts across opposite corners of the top
    balls: Quadric                                            # [sides]

    def wood(self) -> Quadric:
        """The wooden solid's quadrics `[quadrics]`, intersected: inside the underside and the cuts, outside the
        balls."""
        return concatenate([self.underside[None], self.deck[None], self.chamfers, -self.balls])


def step(motor: Motor, spin: Spin, dt: float,
         velocity_and_acceleration: Callable[[Motor, Spin], tuple[Rate, Spin]]) -> tuple[Motor, Spin]:
    """A fourth-order Runge–Kutta–Munthe-Kaas step in the body's motion algebra."""
    def derivative(increment: Rate, angular: Spin) -> tuple[Rate, Spin]:
        rate, acceleration = velocity_and_acceleration(motor * increment.exp(), angular)
        ad = increment.commutator(Rate) * 2                       # Rate <- Rate
        differential = Rate + ad / 2 + ad(ad) / 12                # Rate <- Rate
        return differential(-rate / 2), acceleration

    first_motion, first_spin = derivative(spin * 0, spin)
    second_motion, second_spin = derivative(first_motion * (dt / 2), spin + first_spin * (dt / 2))
    third_motion, third_spin = derivative(second_motion * (dt / 2), spin + second_spin * (dt / 2))
    fourth_motion, fourth_spin = derivative(third_motion * dt, spin + third_spin * dt)
    increment = (first_motion + 2 * second_motion + 2 * third_motion + fourth_motion) * (dt / 6)  # Rate
    spin = spin + (first_spin + 2 * second_spin + 2 * third_spin + fourth_spin) * (dt / 6)  # Spin
    return motor * increment.exp(), spin


def evolve(motor: Motor, spin: Spin, dt: float,
           velocity_and_acceleration: Callable[[Motor, Spin], tuple[Rate, Spin]]) -> Iterator[tuple[Motor, Spin]]:
    """Placements and body-frame spins, from the start, without end."""
    while True:
        yield motor, spin
        motor, spin = step(motor, spin, dt, velocity_and_acceleration)
