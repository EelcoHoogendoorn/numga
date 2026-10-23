"""The tracking scene for the Kalman filter simulates a meandering drive in the plane with sparse pose readings."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Extensor
from examples.estimation.kalman.core import Covariance, Motor, Twist, covariance, mv, position_ellipse, predict, sample, update

# The time step, the pose readings and the commanded steps between them, and the seed of the noise.
DT = 0.1
N_READINGS = 12
N_STEPS_PER_READING = 25
SEED = 2
# The commanded turn rate wanders as a sine of this amplitude and angular frequency.
TURN_AMPLITUDE = 0.9
TURN_FREQUENCY = 0.2
# Standard deviations of translation and rotation: of the motion noise per square root of time, and of a
# pose reading.
MOTION_NOISE = (0.05, 0.12)
MEASUREMENT_NOISE = (0.3, 0.1)
# The duration of an animation frame in milliseconds.
FRAME_DURATION = 40


# --- math -----------------------------------------------------------------------------
def drive(initial: Motor, increments: Twist, motion_noise: Covariance, measurement_noise: Covariance,
          rng: np.random.Generator) -> Iterator[tuple[Motor, Motor, Motor, Covariance, Motor]]:
    """The drive, one state per commanded step and one more after each reading: the true pose, moved by each
    increment and a draw of the motion noise; the pose dead-reckoned from the commands alone; the filter's
    estimate and covariance, predicted through each step and corrected at each reading; and the latest
    reading, a noisy measurement of the true pose."""
    truth = dead = estimate = reading = initial
    sigma = covariance(0.0, 0.0)
    for segment in increments:
        for increment in segment:
            truth = truth * ((increment + sample(motion_noise, rng.normal(size=3))) * 0.5).exp()
            dead = dead * (increment * 0.5).exp()
            estimate, sigma = predict(estimate, sigma, (increment * 0.5).exp(), motion_noise)
            yield truth, dead, estimate, sigma, reading
        reading = truth * (sample(measurement_noise, rng.normal(size=3)) * 0.5).exp()
        estimate, sigma = update(estimate, sigma, reading, measurement_noise)
        yield truth, dead, estimate, sigma, reading


def tracking():
    """Simulate the drive and filter it. Returns, over every state of the drive, the true, dead-reckoned,
    estimated and latest measured poses, and the filter's covariance."""
    times = np.arange(N_READINGS * N_STEPS_PER_READING) * DT
    # Body-frame command: drive 1 m/s along +x (the generator -mv.wx moves along +x) while the turn rate
    # wanders, so the path meanders instead of retracing itself.
    turn = TURN_AMPLITUDE * np.sin(TURN_FREQUENCY * times)
    increments = ((mv.xy * turn - mv.wx) * DT).reshape(N_READINGS, N_STEPS_PER_READING)   # [readings, steps] Twist
    motion_noise = covariance(*(np.array(MOTION_NOISE) * np.sqrt(DT)))
    measurement_noise = covariance(*MEASUREMENT_NOISE)
    states = drive(mv.rotor(), increments, motion_noise, measurement_noise, np.random.default_rng(SEED))
    # [states] Motor, Motor, Motor, Twist <- Line, Motor
    return tuple(Extensor.stack(path) for path in zip(*states))


def errors(truth: Motor, dead: Motor, estimates: Motor):
    """The position errors of dead reckoning and of the filter: the distance between two points of unit
    weight is the norm of the line joining them."""
    origin = mv.w.dual()
    true_here = truth >> origin
    return (true_here & (dead >> origin)).norm(), (true_here & (estimates >> origin)).norm()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.estimation.kalman import render

    truth, dead, estimates, covariances, readings = tracking()
    here, variances, axes = position_ellipse(estimates, covariances, mv.w.dual())
    save_animation(render.animate_tracking(truth, dead, estimates, here, variances, axes, readings), "kalman", FRAME_DURATION)

    # checks
    # Sparse noisy pose readings keep the filtered path far closer to the truth than the dead-reckoned one,
    # and the covariance stays its own adjugate through the filter.
    dead_error, filtered_error = errors(truth, dead, estimates)
    assert filtered_error.mean(axis=0).to_array() < 0.25 * dead_error.mean(axis=0).to_array()
    np.testing.assert_allclose(covariances.adjugate().kernel, covariances.kernel, atol=1e-12)


if __name__ == "__main__":
    main()
