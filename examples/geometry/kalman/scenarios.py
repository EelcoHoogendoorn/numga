"""The tracking scene for the Kalman example: a meandering drive with sparse pose readings."""

from __future__ import annotations

import numpy as np

from numga import Extensor
from examples.geometry.kalman.core import covariance, kalman_filter, mv, position_ellipse, simulate_motion


def tracking():
    """Simulate the drive, filter it, and return the paths, the ellipses and the position errors."""
    rng = np.random.default_rng(2)
    dt, readings, steps_per_reading = 0.1, 12, 25
    # Body-frame command: drive 1 m/s along +x (exp(-wx) moves +x) while the turn rate
    # wanders, so the path meanders instead of retracing itself.
    turn = 0.9 * np.sin(0.2 * np.arange(readings * steps_per_reading) * dt)
    increments = ((mv.xy * turn - mv.wx) * dt).reshape(readings, steps_per_reading)
    times = np.arange(1, readings + 1) * steps_per_reading * dt
    motion_noise = covariance(0.05 * np.sqrt(dt), 0.12 * np.sqrt(dt))
    measurement_noise = covariance(0.3, 0.1)

    origin = mv.xy
    initial = mv.rotor()
    sigma = covariance(0.0, 0.0)
    truth, dead, measurements = simulate_motion(initial, increments, motion_noise, measurement_noise, rng)
    estimates, covariances = zip(*kalman_filter(
        initial, sigma, (increments * 0.5).exp(), measurements, motion_noise, measurement_noise))
    here, variances, axes = position_ellipse(Extensor.stack(estimates), Extensor.stack(covariances), origin)

    # The distance between two points of unit weight is the norm of the line joining them.
    true_here = truth >> origin
    dead_error = (true_here & (dead >> origin)).norm()
    filtered_error = (true_here & here).norm()
    return true_here, dead >> origin, measurements >> origin, here, variances, axes, dead_error, filtered_error, times


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.kalman import render

    track = tracking()
    save_figure(render.draw_tracking(*track), "kalman")
    *_, dead_error, filtered_error, _ = track
    print(f"mean position error, dead reckoning: {dead_error.mean(axis=0).to_array():.3f}")
    print(f"mean position error, filtered:       {filtered_error.mean(axis=0).to_array():.3f}")
