"""Spinor regularization: close Kepler passages as smooth motion of a scaled spinor."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Positions and velocities are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
# The plane an orbit sweeps, the join of two points: a vector with intrinsic orientation.
Patch = ga.gatype.vector().derive.flipped_orientation
Spinor = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
# The point at unit distance along x, carried by a spinor in both its places.
placement = Spinor >> mv.x.dual()                                        # Point <- Spinor, Spinor


@dataclass
class State:
    """Physical time, position and velocity, with matching batch axes."""

    time: Scalar
    position: Point
    velocity: Point

    def energy(self, gravity: float) -> Scalar:
        return self.velocity.scalar_norm_squared() / 2 - gravity / self.position.norm()

    def momentum(self) -> Patch:
        """The angular momentum per mass: the plane of the orbit, weighted by twice the areal speed."""
        return self.position & self.velocity

    @classmethod
    def from_spinor(cls, spinor: Spinor, rate: Spinor, time: Scalar) -> State:
        """Read a position spinor and its regularized derivative in physical time."""
        # The spinor times its reverse is the radius, the distance from the attracting centre at the origin.
        radius = spinor.scalar_norm_squared()                            # [...] Scalar
        # The sandwich differentiated in each of its two arguments; their scalar parts, the unused gauge rate, cancel.
        velocity = (placement(rate, spinor) + placement(spinor, rate)) / radius  # [...] Point
        return cls(time, spinor >> mv.x.dual(), velocity)

    @classmethod
    def collect(cls, states: Iterable[State]) -> State:
        states = list(states)
        return cls(stack([state.time for state in states]),
                   stack([state.position for state in states]),
                   stack([state.velocity for state in states]))


@dataclass
class BoundOrbit:
    """A negative-energy Kepler orbit, lifted into a position spinor and its derivative."""

    spinor: Spinor
    rate: Spinor
    energy: Scalar

    @classmethod
    def lift(cls, start: State, spinor: Spinor, gravity: float) -> BoundOrbit:
        """Lift a physical state through a spinor that places the point along x, `mv.x.dual()`, at its
        position."""
        # This derivative gives the prescribed velocity without spinning in the plane across the radius.
        rate = start.velocity * spinor * ~mv.x.dual() / 2                # [...] Spinor
        return cls(spinor, rate, start.energy(gravity))

    @property
    def frequency(self) -> Scalar:
        """The oscillator's angular frequency in the regularized clock."""
        return (-self.energy / 2).square_root()

    def spinor_at(self, parameter: Scalar) -> Spinor:
        """Exact oscillator evolution of the position spinor."""
        phase = self.frequency * parameter                               # [...] Scalar
        return self.spinor * phase.cos() + self.rate * (phase.sin() / self.frequency)

    def sample(self, parameter: Scalar) -> State:
        """Exact oscillator evolution and its integrated physical clock, for bound motion."""
        frequency = self.frequency                                       # [...] Scalar
        phase = frequency * parameter                                    # [...] Scalar
        cosine, sine = phase.cos(), phase.sin()                          # [...] Scalar each
        spinor = self.spinor_at(parameter)                               # [...] Spinor
        rate = self.rate * cosine - self.spinor * (frequency * sine)     # [...] Spinor
        # Physical time advances by the radius: the clock slows smoothly near the centre.
        initial_radius = self.spinor.scalar_norm_squared()               # [...] Scalar
        rate_radius = self.rate.scalar_norm_squared() / frequency.squared()  # [...] Scalar
        overlap = self.spinor.dot(self.rate) / frequency                      # [...] Scalar
        time = ((initial_radius + rate_radius) * parameter / 2
                + (initial_radius - rate_radius) * (2 * phase).sin() / (4 * frequency)
                + overlap * sine.squared() / frequency)                  # [...] Scalar
        return State.from_spinor(spinor, rate, time)

    def at_time(self, time: Scalar, n_iterations: int) -> State:
        """Invert the monotone physical clock by batched bisection."""
        frequency = self.frequency                                       # [...] Scalar
        mean_radius = (self.spinor.scalar_norm_squared()
                       + self.rate.scalar_norm_squared() / frequency.squared()) / 2  # [...] Scalar
        middle = time / mean_radius                                      # [...] Scalar
        half_period = np.pi / frequency                                  # [...] Scalar
        lower, upper = middle - half_period, middle + half_period        # [...] Scalar each
        for _ in range(n_iterations):
            middle = (lower + upper) / 2                                 # [...] Scalar
            before = self.sample(middle).time < time                     # [...] bool
            lower = before * middle + (1 - before) * lower               # [...] Scalar
            upper = before * upper + (1 - before) * middle               # [...] Scalar
        return self.sample((lower + upper) / 2)

    def position_error(self, trajectory: State, n_iterations: int) -> Scalar:
        """Position RMS error weighted by elapsed physical time, against the exact orbit."""
        exact = self.at_time(trajectory.time, n_iterations)              # [samples, ...] State
        squared = (trajectory.position - exact.position).scalar_norm_squared()  # [samples, ...] Scalar
        intervals = trajectory.time[1:] - trajectory.time[:-1]           # [samples - 1, ...] Scalar
        integral = ((squared[1:] + squared[:-1]) * intervals / 2).sum(axis=0)  # [...] Scalar
        return (integral / (trajectory.time[-1] - trajectory.time[0])).square_root()


def physical_verlet(initial: State, gravity: float, step: float, n_steps: int) -> Iterator[State]:
    """Kick-drift-kick integration with equal steps in physical time."""
    position, velocity, time = initial.position, initial.velocity, initial.time  # [...] Point, Point, Scalar
    acceleration = -gravity * position / position.norm() ** 3            # [...] Point
    yield initial
    for _ in range(n_steps):
        half_velocity = velocity + acceleration * (step / 2)             # [...] Point
        position = position + half_velocity * step                       # [...] Point
        acceleration = -gravity * position / position.norm() ** 3        # [...] Point
        velocity = half_velocity + acceleration * (step / 2)             # [...] Point
        time = time + step                                               # [...] Scalar
        yield State(time, position, velocity)


def regularized_verlet(orbit: BoundOrbit, time: Scalar, step: Scalar, n_steps: int) -> Iterator[State]:
    """Kick-drift-kick integration of the spinor oscillator from a start time, with its physical clock."""
    spinor, rate = orbit.spinor, orbit.rate                              # [...] Spinor each
    yield State.from_spinor(spinor, rate, time)
    for _ in range(n_steps):
        half_rate = rate + orbit.energy * spinor * (step / 4)            # [...] Spinor
        following = spinor + half_rate * step                            # [...] Spinor
        # Integrate the radius, spinor.scalar_norm_squared(), by Simpson's rule over the drift.
        time = time + step / 6 * (spinor.scalar_norm_squared()
                                  + (spinor + following).scalar_norm_squared()
                                  + following.scalar_norm_squared())     # [...] Scalar
        rate = half_rate + orbit.energy * following * (step / 4)         # [...] Spinor
        spinor = following
        yield State.from_spinor(spinor, rate, time)
