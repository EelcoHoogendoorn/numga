"""Spinor regularization maps eccentric Kepler orbits onto smooth harmonic-oscillator motion in a distance-scaled clock."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Positions and velocities are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Spinor = ga.gatype.even()


# --- math -----------------------------------------------------------------------------
# Unit point along x carried by a spinor in both factors of the sandwich product.
placement = Spinor >> mv.x.dual()                                        # [] Point- <- (Spinor+, Spinor+)
# Gauge-free velocity lift with zero spin in the plane perpendicular to the reference point.
velocity_lift = Point * Spinor * ~mv.x.dual() / 2                        # [] Spinor+ <- (Point-, Spinor+)


def oscillator_spinor(spinor: Spinor, rate: Spinor, frequency: Scalar, parameter: Scalar) -> Spinor:
    """Harmonic-oscillator evolution advances `spinor` (`[...] Spinor+`) at angular `frequency` in the regularized clock `parameter`."""
    phase = frequency * parameter                                        # [...] Scalar+
    return spinor * phase.cos() + rate * (phase.sin() / frequency)


def physical_state(spinor: Spinor, rate: Spinor) -> tuple[Point, Point]:
    """The sandwich product and its derivative map `spinor` and `rate` to physical position and velocity (`[...] Point-`, `[...] Point-`)."""
    # The spinor times its reverse equals the distance from the attracting centre at the origin.
    radius = spinor.scalar_norm_squared()                                # [...] Scalar+
    # Differentiating both factors of the sandwich cancels the scalar gauge rate.
    velocity = (placement(rate, spinor) + placement(spinor, rate)) / radius  # [...] Point-
    return spinor >> mv.x.dual(), velocity


def exact_orbit(spinor: Spinor, rate: Spinor, frequency: Scalar,
                parameter: Scalar) -> tuple[Scalar, Point, Point]:
    """Closed-form oscillator motion and integrated radius give physical `(time, position, velocity)` at regularized clock `parameter`."""
    phase = frequency * parameter                                        # [...] Scalar+
    sine = phase.sin()                                                   # [...] Scalar+
    current_spinor = oscillator_spinor(spinor, rate, frequency, parameter)  # [...] Spinor+
    current_rate = rate * phase.cos() - spinor * (frequency * sine)      # [...] Spinor+
    # Physical time advances by the radius, slowing the regularized clock smoothly near the centre.
    initial_radius = spinor.scalar_norm_squared()                        # [...] Scalar+
    rate_radius = rate.scalar_norm_squared() / frequency.squared()       # [...] Scalar+
    overlap = spinor.scalar_product(~rate) / frequency                   # [...] Scalar+
    time = ((initial_radius + rate_radius) * parameter / 2
            + (initial_radius - rate_radius) * (2 * phase).sin() / (4 * frequency)
            + overlap * sine.squared() / frequency)                      # [...] Scalar+
    position, velocity = physical_state(current_spinor, current_rate)    # [...] Point-, Point-
    return time, position, velocity


def physical_verlet(position: Point, velocity: Point, time: Scalar,
                    gravity: float, step: float, n_steps: int) -> Iterator[tuple[Scalar, Point, Point]]:
    """Yields `(time, position, velocity)` states across `n_steps` equal kick-drift-kick steps in physical time."""
    acceleration = -gravity * position / position.norm() ** 3            # [...] Point-
    yield time, position, velocity
    for _ in range(n_steps):
        half_velocity = velocity + acceleration * (step / 2)             # [...] Point-
        position = position + half_velocity * step                       # [...] Point-
        acceleration = -gravity * position / position.norm() ** 3        # [...] Point-
        velocity = half_velocity + acceleration * (step / 2)             # [...] Point-
        time = time + step                                               # [...] Scalar+
        yield time, position, velocity


def regularized_verlet(spinor: Spinor, rate: Spinor, energy: Scalar, time: Scalar,
                       step: Scalar, n_steps: int) -> Iterator[tuple[Scalar, Point, Point]]:
    """Yields `(time, position, velocity)` states across `n_steps` kick-drift-kick oscillator steps in the regularized clock."""
    initial_position, initial_velocity = physical_state(spinor, rate)    # [...] Point-, Point-
    yield time, initial_position, initial_velocity
    for _ in range(n_steps):
        half_rate = rate + energy * spinor * (step / 4)                  # [...] Spinor+
        next_spinor = spinor + half_rate * step                          # [...] Spinor+
        # Simpson's rule integrates the quadratic drift radius over the regularized step.
        time = time + step / 6 * (spinor.scalar_norm_squared()
                                  + (spinor + next_spinor).scalar_norm_squared()
                                  + next_spinor.scalar_norm_squared())   # [...] Scalar+
        rate = half_rate + energy * next_spinor * (step / 4)             # [...] Spinor+
        spinor = next_spinor                                             # [...] Spinor+
        next_position, next_velocity = physical_state(spinor, rate)      # [...] Point-, Point-
        yield time, next_position, next_velocity


def state_at_time(spinor: Spinor, rate: Spinor, frequency: Scalar,
                  time: Scalar, n_iterations: int) -> tuple[Scalar, Point, Point]:
    """Batched bisection over `n_iterations` steps inverts the monotone physical clock to evaluate `(time, position, velocity)` at `time`."""
    mean_radius = (spinor.scalar_norm_squared()
                   + rate.scalar_norm_squared() / frequency.squared()) / 2  # [...] Scalar+
    midpoint = time / mean_radius                                        # [...] Scalar+
    half_period = np.pi / frequency                                      # [...] Scalar+
    lower_bound, upper_bound = midpoint - half_period, midpoint + half_period  # [...] Scalar+, Scalar+
    for _ in range(n_iterations):
        midpoint = (lower_bound + upper_bound) / 2                       # [...] Scalar+
        earlier_mask = exact_orbit(spinor, rate, frequency, midpoint)[0] < time
        lower_bound = earlier_mask * midpoint + (1 - earlier_mask) * lower_bound  # [...] Scalar+
        upper_bound = earlier_mask * upper_bound + (1 - earlier_mask) * midpoint  # [...] Scalar+
    return exact_orbit(spinor, rate, frequency, (lower_bound + upper_bound) / 2)


def position_error(spinor: Spinor, rate: Spinor, frequency: Scalar,
                   states: Iterable[tuple[Scalar, Point, Point]], n_iterations: int) -> Scalar:
    """Trapezoidal quadrature over elapsed physical time computes the root-mean-square position error (`[...] Scalar+`) against the exact orbit."""
    sample_list = list(states)
    n_samples = len(sample_list)
    times = stack([sample[0] for sample in sample_list])                 # [n_samples, ...] Scalar+
    positions = stack([sample[1] for sample in sample_list])             # [n_samples, ...] Point-
    exact_positions = state_at_time(spinor, rate, frequency, times, n_iterations)[1]  # [n_samples, ...] Point-
    squared_errors = (positions - exact_positions).scalar_norm_squared()  # [n_samples, ...] Scalar+
    intervals = times[1:] - times[:-1]                                   # [n_samples - 1, ...] Scalar+
    # Weight by elapsed physical time so clustered periapsis samples do not dominate the norm.
    integral = ((squared_errors[1:] + squared_errors[:-1]) * intervals / 2).sum(axis=0)  # [...] Scalar+
    return (integral / (times[-1] - times[0])).square_root()


def lift_states(spinor: Spinor, rate: Spinor, frequency: Scalar,
                n_revolutions: int, n_frames: int) -> Iterator[tuple[Point, Spinor]]:
    """Yields `(position, spinor)` pairs (`[...] Point-`, `[...] Spinor+`) at `n_frames + 1` uniform regularized-clock samples over `n_revolutions` orbits."""
    for frame_index in range(n_frames + 1):
        parameter = frame_index * n_revolutions * np.pi / (n_frames * frequency)  # [...] Scalar+
        current_spinor = oscillator_spinor(spinor, rate, frequency, parameter)  # [...] Spinor+
        yield current_spinor >> mv.x.dual(), current_spinor
