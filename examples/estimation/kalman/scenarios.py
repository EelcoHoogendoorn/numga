"""Tracking scene for the planar Kalman filter: a meandering drive with dead-reckoned motion noise and sparse pose readings."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial

import numpy as np

from examples.estimation.kalman.core import (
    Covariance,
    Line,
    Motor,
    Twist,
    body_covariance,
    filter_update,
    mv,
    origin,
    position_ellipsoid,
    twist_samples,
)
from numga import stack

# Timestep in seconds, reading count, commanded steps between readings, and random seed.
dt = 0.1
n_readings = 12
n_steps_per_reading = 25
seed = 2
# Sinusoidal commanded turn-rate amplitude (rad/s) and angular frequency (rad/s).
turn_amplitude = 0.9
turn_frequency = 0.2
# Translation and rotation standard deviations: per square-root second for motion noise, and per pose reading.
motion_std = (0.05, 0.12)
measurement_std = (0.3, 0.1)
# Animation frame duration in milliseconds and confidence-ellipse standard-deviation level.
frame_duration_ms = 40
sigmas = 2.0


# --- math -----------------------------------------------------------------------------
def commanded_increments(n_readings: int, n_steps_per_reading: int) -> Twist:
    """Unit forward speed along `+x` with a sinusoidal turn rate forms the commanded body twists
    (`[n_readings, n_steps_per_reading] Twist+`)."""
    times = np.arange(n_readings * n_steps_per_reading) * dt
    # The generator -mv.wx translates along +x while mv.xy turns about the body origin.
    turn = turn_amplitude * np.sin(turn_frequency * times)
    return ((mv.xy * turn - mv.wx) * dt).reshape(n_readings, n_steps_per_reading)   # [n_readings, n_steps_per_reading] Twist+


def drive_states(
    initial: Motor,
    increments: Twist,
    motion_noise: Covariance,
    measurement_noise: Covariance,
    rng: np.random.Generator,
) -> Iterator[tuple[Motor, Motor, Motor, Covariance, Motor]]:
    """Yields `(truth, dead, estimate, sigma, reading)` after each commanded step in `increments`
    (`[n_readings, n_steps_per_reading] Twist+`) and after each pose reading."""
    truth = dead = estimate = reading = initial
    sigma = body_covariance(0.0, 0.0)                                               # [] Twist+ <- Line-
    for segment in increments:
        for increment in segment:
            step = (increment * 0.5).exp()                                          # [] Motor+
            noisy_step = ((increment + twist_samples(motion_noise, rng.normal(size=3))) * 0.5).exp()   # [] Motor+
            truth = truth * noisy_step
            dead = dead * step
            # Pull the readout line through the step and push the expected twist back.
            estimate, sigma = estimate * step, (step << sigma(step >> Line)) + motion_noise
            yield truth, dead, estimate, sigma, reading
        measurement_error = (twist_samples(measurement_noise, rng.normal(size=3)) * 0.5).exp()   # [] Motor+
        reading = truth * measurement_error
        estimate, sigma = filter_update(estimate, sigma, reading, measurement_noise)
        yield truth, dead, estimate, sigma, reading


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate: Callable, *arguments, name: str, duration_ms: int) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    """Streams the meandering Kalman filter drive once to `plots/kalman.gif` and checks tracking accuracy and symmetry."""
    from examples.estimation.kalman import render

    increments = commanded_increments(n_readings, n_steps_per_reading)              # [n_readings, n_steps_per_reading] Twist+
    motion_noise = body_covariance(*(np.array(motion_std) * np.sqrt(dt)))           # [] Twist+ <- Line-
    measurement_noise = body_covariance(*measurement_std)                           # [] Twist+ <- Line-
    ellipsoid_of = partial(position_ellipsoid, sigmas=sigmas)
    recorded = save_states(
        drive_states(mv.rotor(), increments, motion_noise, measurement_noise, np.random.default_rng(seed)),
        render.tracking_animation,
        increments,
        ellipsoid_of,
        name="kalman",
        duration_ms=frame_duration_ms,
    )
    truth, dead, estimates, covariances, _ = (stack(part) for part in zip(*recorded))   # [...] Motor+, Motor+, Motor+, Twist+ <- Line-, Motor+

    # checks
    # Sparse noisy pose readings keep the filtered trajectory far closer to the truth than dead reckoning,
    # while every covariance map stays symmetric under `.adjugate()` across all three twist-line pairings.
    true_positions = truth >> origin                                                # [...] Point-
    positions = estimates >> origin                                                 # [...] Point-
    dead_error = (true_positions & (dead >> origin)).norm()                         # [...] Scalar+
    filtered_error = (true_positions & positions).norm()                            # [...] Scalar+
    assert bool(filtered_error.mean(axis=0) < 0.25 * dead_error.mean(axis=0))
    frame = mv.bivector(np.eye(3)).dual()                                           # [3] Line-
    skew = covariances.adjugate() - covariances                                     # [...] Twist+ <- Line-
    symmetry_residual = (frame[:, None] & skew[..., None, None](frame[None, :])).abs().max()   # [] Scalar+
    assert bool(symmetry_residual < 1e-12)


if __name__ == "__main__":
    main()

