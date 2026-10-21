"""Thomas precession: two boosts in different directions do not make a boost, they leave a turn
behind, and a spin carried around any closed loop of velocities comes back turned by the area the
loop encloses in the space of velocities. With a magnetic moment, the same frame-turning bivector
gives the Bargmann–Michel–Telegdi equation that the muon g−2 storage ring measures.

A velocity is a direction in time, an antivector like every direction: the lab's time direction under
every boost, `boost >> time`, a hyperboloid whose geometry is the hyperbolic plane. The pure boost from
one velocity to another is the square root of their product, `(head * tail).square_root()`, and
carrying a spin along a path of velocities is the product of those boosts. A spin is the plane the
gyroscope turns in, a rate like every turning, and an observer at rest reads it after the pure boost
back to rest, `(time * velocity).square_root() >> spin`.

In a field bivector `field`, with charge-to-mass ratio one, the force is the field read along the
velocity, `velocity & field`, and the bivector `(1 + anomaly) * field - anomaly * (force ^ velocity.dual())`
turns the particle's whole frame: its velocity as the Lorentz force does, whatever the anomaly, and its
spin by the Bargmann–Michel–Telegdi equation. The metric enters through the dual, which turns the velocity,
a direction, into the slice of space across it that the force's plane is built from.

In the four-tensor index notation of relativistic electrodynamics, the Bargmann–Michel–Telegdi
equation reads as $\\frac{ds^\\mu}{d\\tau} = \\frac{q}{m}\\left(\\frac{g}{2} F^{\\mu}{}_{\\nu} s^\\nu
+ \\left(\\frac{g}{2} - 1\\right) u^\\mu F_{\\nu\\lambda} u^\\nu s^\\lambda\\right)$.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Rotor = ga.gatype.rotor()

# One unit along the lab's time: the direction across the lab's slice of space `mv.t`.
time = -mv.t.dual()                                          # [] Antivector


# --- math -----------------------------------------------------------------------------
def boost(direction: Antivector, rapidity: np.ndarray) -> Rotor:
    """The pure boost by `rapidity` along a unit spatial direction: a turn in the plane of the direction
    and time, their join, which the dual turns into a generator."""
    return ((direction & time).dual() * (rapidity / 2)).exp()                   # [...] Rotor


def rest_boost(velocity: Antivector) -> Rotor:
    """The pure boost carrying the lab's time direction to `velocity`."""
    return (velocity * time).square_root()                                      # [...] Rotor


def lab_view(velocity: Antivector, spin: Bivector) -> Bivector:
    """A spin of the frame moving at `velocity`, as the observer at rest reads it: after the pure
    boost back to rest."""
    return (time * velocity).square_root() >> spin                              # [...] Bivector


def transport(tail: Antivector, head: Antivector) -> Rotor:
    """The pure boost from velocity `tail` to velocity `head`, in their plane."""
    return (head * tail).square_root()                                          # [...] Rotor


def geodesic(tail: Antivector, head: Antivector, fractions: np.ndarray) -> Antivector:
    """The velocities along the pure boost from `tail` to `head`, at each fraction of the way."""
    return (transport(tail, head).log() * fractions).exp() >> tail             # [fractions, ...] Antivector


def on_hyperboloid(spatial: Antivector) -> Antivector:
    """The velocity whose spatial part is `spatial`: the unit direction in time above it."""
    return spatial + time * (1 - spatial.squared()).square_root()               # [...] Antivector


def carried(velocities: Antivector, spin: Bivector) -> Iterator[Bivector]:
    """A spin carried along a path of velocities `[steps + 1, ...]` by the pure boost between each
    velocity and the next, torque-free: the spin at each velocity."""
    yield spin
    for tail, head in zip(velocities[:-1], velocities[1:]):
        spin = transport(tail, head) >> spin                                    # [...] Bivector
        yield spin


def bmt_generator(field: Bivector, velocity: Antivector, anomaly: float) -> Bivector:
    """The bivector turning the frame of a particle of magnetic anomaly `anomaly`, `g / 2 - 1`: its
    velocity by the Lorentz force and its spin by the Bargmann–Michel–Telegdi equation."""
    force = velocity & field                                                    # [...] Vector
    # On the velocity the force's plane with the velocity acts as the field does, so the anomaly
    # cancels there.
    return (1 + anomaly) * field - anomaly * (force ^ velocity.dual())         # [...] Bivector


def ring_field(event: Antivector, magnetic: float, electric: np.ndarray) -> Bivector:
    """A storage ring's field at an event: a uniform magnetic field in the orbit's plane, and an
    electric field of strength `electric` pointing away from the ring's axis."""
    position = event - (mv.t & event) * time                                    # [...] Antivector
    # Directions of space square to minus their length squared.
    outward = position / (-position.squared()).square_root()                    # [...] Antivector
    return magnetic * mv.xy + electric * (outward & time).dual()               # [...] Bivector


def bmt_step(event: Antivector, frame: Rotor, magnetic: float, electric: np.ndarray, anomaly: float,
             proper_step: np.ndarray) -> tuple[Antivector, Rotor]:
    """One midpoint step in proper time of a particle's event and frame in the ring's field."""
    rate = bmt_generator(ring_field(event, magnetic, electric), frame >> time, anomaly)   # [...] Bivector
    halfway_frame = (rate * (proper_step / 4)).exp() * frame                   # [...] Rotor
    halfway_event = event + (frame >> time) * (proper_step / 2)                 # [...] Antivector
    halfway_rate = bmt_generator(ring_field(halfway_event, magnetic, electric), halfway_frame >> time, anomaly)
    return event + (halfway_frame >> time) * proper_step, (halfway_rate * (proper_step / 2)).exp() * frame


def evolve(event: Antivector, frame: Rotor, magnetic: float, electric: np.ndarray, anomaly: float,
           proper_step: np.ndarray) -> Iterator[tuple[Antivector, Rotor]]:
    """The particle's event and frame at every step, without end."""
    while True:
        yield event, frame
        event, frame = bmt_step(event, frame, magnetic, electric, anomaly, proper_step)


def ring_radius(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> np.ndarray:
    """The radius of the circular orbit at Lorentz factor `gamma`, where the magnetic pull inwards less
    the electric push outwards turns the momentum."""
    speed = np.sqrt(1 - 1 / gamma**2)
    return gamma * speed**2 / (speed * magnetic - electric)


def anomalous_angle(velocity: Antivector, spin: Bivector) -> Scalar:
    """The angle in the orbit's plane from the plane a spin along the momentum turns in to the spin, as
    the observer at rest reads both."""
    motion = velocity - (mv.t & velocity) * time                                # [...] Antivector
    direction = motion / (-motion.squared()).square_root()                      # [...] Antivector
    # A spin along the momentum turns in the plane of the direction turned a quarter turn in the orbit's
    # plane and the orbit's normal.
    along = mv.z ^ direction.commutator(mv.xy).dual()                          # [...] Bivector
    # The product of two such planes is the rotor through their angle.
    return (lab_view(velocity, spin) * along.inverse()).log().scalar_product(mv.yx)   # [...] Scalar
