"""Close Kepler passages as smooth motion of a scaled spinor."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Spinor = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
@dataclass
class State:
    """Physical time, position and velocity, with matching batch axes."""

    time: Scalar
    position: Vector
    velocity: Vector

    def energy(self, gravity: float) -> Scalar:
        return self.velocity.scalar_norm_squared() / 2 - gravity / self.position.norm()

    def momentum(self) -> Bivector:
        return self.position ^ self.velocity

    @classmethod
    def collect(cls, states: Iterable[State]) -> State:
        states = list(states)
        return cls(stack([state.time for state in states]),
                   stack([state.position for state in states]),
                   stack([state.velocity for state in states]))


def physical_state(spinor: Spinor, rate: Spinor, time: Scalar) -> State:
    """Read a position spinor and its regularized derivative in physical time."""
    radius = spinor.scalar_norm_squared()
    position = spinor >> mv.x
    # The vector part differentiates the sandwich. The trivector is the unused gauge rate.
    velocity = 2 * (rate * mv.x * spinor.reverse()).select[1] / radius
    return State(time, position, velocity)


@dataclass
class BoundOrbit:
    """A negative-energy Kepler orbit, lifted into a position spinor and its derivative."""

    spinor: Spinor
    rate: Spinor
    energy: Scalar

    @classmethod
    def lift(cls, spinor: Spinor, velocity: Vector, gravity: float) -> BoundOrbit:
        position = spinor >> mv.x
        energy = velocity.scalar_norm_squared() / 2 - gravity / position.norm()
        # This derivative gives the prescribed velocity without spinning about the radial axis.
        rate = velocity * spinor * mv.x / 2
        return cls(spinor, rate, energy)

    def sample(self, parameter: Scalar) -> State:
        """Exact oscillator evolution and its integrated physical clock, for bound motion."""
        frequency = (-self.energy / 2).square_root()
        phase = frequency * parameter
        cosine, sine = phase.cos(), phase.sin()
        spinor = self.spinor * cosine + self.rate * (sine / frequency)
        rate = self.rate * cosine - self.spinor * (frequency * sine)
        # Physical time advances by the radius: the clock slows smoothly near the focus.
        initial_radius = self.spinor.scalar_norm_squared()
        rate_radius = self.rate.scalar_norm_squared() / frequency.squared()
        overlap = self.spinor.scalar_product(self.rate.reverse()) / frequency
        time = ((initial_radius + rate_radius) * parameter / 2
                + (initial_radius - rate_radius) * (2 * phase).sin() / (4 * frequency)
                + overlap * sine.squared() / frequency)
        return physical_state(spinor, rate, time)

    def at_time(self, time: Scalar, iterations: int) -> State:
        """Invert the monotone physical clock by batched bisection."""
        frequency = (-self.energy / 2).square_root()
        mean_radius = (self.spinor.scalar_norm_squared()
                       + self.rate.scalar_norm_squared() / frequency.squared()) / 2
        middle = time / mean_radius
        half_period = np.pi / frequency
        lower, upper = middle - half_period, middle + half_period
        for _ in range(iterations):
            middle = (lower + upper) / 2
            before = self.sample(middle).time < time
            lower = before * middle + (1 - before) * lower
            upper = before * upper + (1 - before) * middle
        return self.sample((lower + upper) / 2)

    def position_error(self, trajectory: State, iterations: int) -> Scalar:
        """Position RMS error weighted by elapsed physical time, against the exact orbit."""
        exact = self.at_time(trajectory.time, iterations)
        squared = (trajectory.position - exact.position).scalar_norm_squared()
        intervals = trajectory.time[1:] - trajectory.time[:-1]
        integral = ((squared[1:] + squared[:-1]) * intervals / 2).sum(axis=0)
        return (integral / (trajectory.time[-1] - trajectory.time[0])).square_root()


def physical_verlet(initial: State, gravity: float, step: float, steps: int) -> Iterator[State]:
    """Kick-drift-kick integration with equal steps in physical time."""
    position, velocity, time = initial.position, initial.velocity, initial.time
    acceleration = -gravity * position / position.norm() ** 3
    yield initial
    for _ in range(steps):
        half_velocity = velocity + acceleration * (step / 2)
        position = position + half_velocity * step
        acceleration = -gravity * position / position.norm() ** 3
        velocity = half_velocity + acceleration * (step / 2)
        time = time + step
        yield State(time, position, velocity)


def regularized_verlet(orbit: BoundOrbit, step: Scalar, steps: int) -> Iterator[State]:
    """Kick-drift-kick integration of the spinor oscillator, with its physical clock."""
    spinor, rate = orbit.spinor, orbit.rate
    time = orbit.energy * 0
    yield physical_state(spinor, rate, time)
    for _ in range(steps):
        half_rate = rate + orbit.energy * spinor * (step / 4)
        following = spinor + half_rate * step
        # Integrate the squared spinor along the linear drift, including the cross term.
        time = time + step / 3 * (spinor.scalar_norm_squared()
                                  + spinor.scalar_product(following.reverse())
                                  + following.scalar_norm_squared())
        rate = half_rate + orbit.energy * following * (step / 4)
        spinor = following
        yield physical_state(spinor, rate, time)
