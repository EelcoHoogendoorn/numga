"""The tracking scene for the Kalman filter simulates a meandering drive in the plane with sparse pose readings."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.estimation.kalman.core import Covariance, Line, Motor, Twist, covariance, mv, position_ellipse, sample, update

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
# The duration of an animation frame in milliseconds, and the standard-deviation level of the ellipse.
FRAME_DURATION = 40
SIGMAS = 2.0


# --- math -----------------------------------------------------------------------------
def drive(initial: Motor, increments: Twist, motion_noise: Covariance, measurement_noise: Covariance,
          rng: np.random.Generator) -> Iterator[tuple[Motor, Motor, Motor, Covariance, Motor]]:
    """The drive yields, at every commanded step and after every reading, the true pose, the dead-reckoned pose,
    the filter's estimate and covariance, and the latest reading."""
    truth = dead = estimate = reading = initial
    sigma = covariance(0.0, 0.0)
    for segment in increments:
        for increment in segment:
            step = (increment * 0.5).exp()                                          # [] Motor+
            noisy_step = ((increment + sample(motion_noise, rng.normal(size=3))) * 0.5).exp()  # [] Motor+
            truth = truth * noisy_step
            dead = dead * step
            # Carry the readout through the step and push the correlated twist back.
            estimate, sigma = estimate * step, (step << sigma(step >> Line)) + motion_noise
            yield truth, dead, estimate, sigma, reading
        measurement_error = (sample(measurement_noise, rng.normal(size=3)) * 0.5).exp()  # [] Motor+
        reading = truth * measurement_error
        estimate, sigma = update(estimate, sigma, reading, measurement_noise)
        yield truth, dead, estimate, sigma, reading


def tracking() -> tuple[Motor, Motor, Motor, Covariance, Motor]:
    """The meandering drive's states are the true, dead-reckoned, estimated and read poses, with the filter's covariance."""
    times = np.arange(N_READINGS * N_STEPS_PER_READING) * DT
    # Body-frame command: drive 1 m/s along +x (the generator -mv.wx moves along +x) while the turn rate
    # wanders, so the path meanders instead of retracing itself.
    turn = TURN_AMPLITUDE * np.sin(TURN_FREQUENCY * times)
    increments = ((mv.xy * turn - mv.wx) * DT).reshape(N_READINGS, N_STEPS_PER_READING)  # [readings, steps] Twist+
    motion_noise = covariance(*(np.array(MOTION_NOISE) * np.sqrt(DT)))          # [] Twist+ <- Line-
    measurement_noise = covariance(*MEASUREMENT_NOISE)                          # [] Twist+ <- Line-
    states = drive(mv.rotor(), increments, motion_noise, measurement_noise, np.random.default_rng(SEED))
    # [states] Motor+, Motor+, Motor+, Twist+ <- Line-, Motor+
    return tuple(stack(path) for path in zip(*states))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.estimation.kalman import render

    origin = mv.w.dual()                                                        # [] Point-
    truth, dead, estimates, covariances, readings = tracking()
    positions = estimates >> origin                                             # [states] Point-
    ellipses = position_ellipse(estimates, covariances, positions, SIGMAS)      # [states] Plane+ <- Point-
    save_animation(render.animate_tracking(truth, dead, estimates, positions, ellipses, readings), "kalman", FRAME_DURATION)

    # checks
    # Sparse noisy pose readings keep the filtered path far closer to the truth than the dead-reckoned one,
    # and the covariance stays its own adjugate through the filter.
    true_positions = truth >> origin                                            # [states] Point-
    dead_error = (true_positions & (dead >> origin)).norm()                     # [states] Scalar+
    filtered_error = (true_positions & positions).norm()                        # [states] Scalar+
    assert filtered_error.mean(axis=0) < 0.25 * dead_error.mean(axis=0)
    np.testing.assert_allclose(covariances.adjugate().kernel, covariances.kernel, atol=1e-12)


if __name__ == "__main__":
    main()
