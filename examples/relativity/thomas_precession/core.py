"""Thomas precession: two boosts in different directions do not make a boost, they leave a turn
behind, and a spin carried around any closed loop of velocities comes back turned by the area the
loop encloses in the space of velocities. With a magnetic moment, the same frame-turning bivector
gives the Bargmann–Michel–Telegdi equation that the muon g−2 storage ring measures.

The velocities a particle can have are the unit timelike vectors, the time axis under every boost,
`boost >> mv.t`: a hyperboloid whose geometry is the hyperbolic plane. The pure boost from one
velocity to another is the square root of their product, `(head * tail).square_root()`, and carrying
a spin along a path of velocities is the product of those boosts. An observer at rest reads a spin
after the pure boost back to rest, `(mv.t * velocity).square_root() >> spin`.

In a field bivector `field`, with charge-to-mass ratio one, the acceleration is `field | velocity`,
and the bivector `(1 + anomaly) * field - anomaly * (acceleration ^ velocity)` turns the particle's
whole frame: its velocity as the Lorentz force does, whatever the anomaly, and its spin by the
Bargmann–Michel–Telegdi equation.

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
Rotor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def boost(direction: Vector, rapidity: np.ndarray) -> Rotor:
    """The pure boost by `rapidity` along a unit spatial direction."""
    return ((direction ^ mv.t) * (rapidity / 2)).exp()                          # [...] Rotor


def rest_boost(velocity: Vector) -> Rotor:
    """The pure boost carrying the observer's time axis to `velocity`."""
    return (velocity * mv.t).square_root()                                      # [...] Rotor


def lab_view(velocity: Vector, vector: Vector) -> Vector:
    """A vector of the frame moving at `velocity`, as the observer at rest reads it: after the pure
    boost back to rest."""
    return (mv.t * velocity).square_root() >> vector                            # [...] Vector


def transport(tail: Vector, head: Vector) -> Rotor:
    """The pure boost from velocity `tail` to velocity `head`, in their plane."""
    return (head * tail).square_root()                                          # [...] Rotor


def geodesic(tail: Vector, head: Vector, fractions: np.ndarray) -> Vector:
    """The velocities along the pure boost from `tail` to `head`, at each fraction of the way."""
    return (transport(tail, head).log() * fractions).exp() >> tail             # [fractions, ...] Vector


def on_hyperboloid(spatial: Vector) -> Vector:
    """The velocity whose spatial part is `spatial`: the unit timelike vector above it."""
    return spatial + mv.t * (1 - spatial.squared()).square_root()               # [...] Vector


def carried(velocities: Vector, spin: Vector) -> Iterator[Vector]:
    """A spin carried along a path of velocities `[steps + 1, ...]` by the pure boost between each
    velocity and the next, torque-free: the spin at each velocity."""
    yield spin
    for tail, head in zip(velocities[:-1], velocities[1:]):
        spin = transport(tail, head) >> spin                                    # [...] Vector
        yield spin


def bmt_generator(field: Bivector, velocity: Vector, anomaly: float) -> Bivector:
    """The bivector turning the frame of a particle of magnetic anomaly `anomaly`, `g / 2 - 1`: its
    velocity by the Lorentz force and its spin by the Bargmann–Michel–Telegdi equation."""
    acceleration = field | velocity                                             # [...] Vector
    # On the velocity the wedge gives back the acceleration, so the anomaly cancels there.
    return (1 + anomaly) * field - anomaly * (acceleration ^ velocity)          # [...] Bivector


def ring_field(event: Vector, magnetic: float, electric: np.ndarray) -> Bivector:
    """A storage ring's field at an event: a uniform magnetic field in the orbit's plane, and an
    electric field of strength `electric` pointing away from the ring's axis."""
    position = mv.t | (mv.t ^ event)                                            # [...] Vector
    # Vectors of space square to minus their length squared.
    outward = position / (-position.squared()).square_root()                    # [...] Vector
    return magnetic * mv.xy + electric * (outward ^ mv.t)                       # [...] Bivector


def bmt_step(event: Vector, frame: Rotor, magnetic: float, electric: np.ndarray, anomaly: float,
             proper_step: np.ndarray) -> tuple[Vector, Rotor]:
    """One midpoint step in proper time of a particle's event and frame in the ring's field."""
    rate = bmt_generator(ring_field(event, magnetic, electric), frame >> mv.t, anomaly)   # [...] Bivector
    halfway_frame = (rate * (proper_step / 4)).exp() * frame                   # [...] Rotor
    halfway_event = event + (frame >> mv.t) * (proper_step / 2)                 # [...] Vector
    halfway_rate = bmt_generator(ring_field(halfway_event, magnetic, electric), halfway_frame >> mv.t, anomaly)
    return event + (halfway_frame >> mv.t) * proper_step, (halfway_rate * (proper_step / 2)).exp() * frame


def evolve(event: Vector, frame: Rotor, magnetic: float, electric: np.ndarray, anomaly: float,
           proper_step: np.ndarray) -> Iterator[tuple[Vector, Rotor]]:
    """The particle's event and frame at every step, without end."""
    while True:
        yield event, frame
        event, frame = bmt_step(event, frame, magnetic, electric, anomaly, proper_step)


def ring_radius(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> np.ndarray:
    """The radius of the circular orbit at Lorentz factor `gamma`, where the magnetic pull inwards less
    the electric push outwards turns the momentum."""
    speed = np.sqrt(1 - 1 / gamma**2)
    return gamma * speed**2 / (speed * magnetic - electric)


def anomalous_angle(velocity: Vector, spin: Vector) -> Scalar:
    """The angle in the orbit's plane from the momentum's direction to the spin, as the observer at
    rest reads both."""
    motion = mv.t | (mv.t ^ velocity)                                           # [...] Vector
    direction = motion / (-motion.squared()).square_root()                      # [...] Vector
    # Minus the product of two unit vectors of space is the rotor through twice their angle.
    return (-(lab_view(velocity, spin) * direction)).log().scalar_product(mv.yx)   # [...] Scalar
