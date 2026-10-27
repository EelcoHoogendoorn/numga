"""Satellite navigation near Earth's pole, with changing sky coverage and noisy clocks."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.relativity.gps import core
from examples.relativity.gps.core import across, emitted_events, locate, mv, origin, time

# Earth's radius, the orbits' altitude, light's speed and Earth's gravitational parameter, in metres and seconds;
# lengths below are in Earth radii and times in Earth-radius light-travel times, so light speed is one.
EARTH_RADIUS_METRES = 6_371_000
ORBIT_ALTITUDE_METRES = 20_200_000
LIGHT_SPEED = 299_792_458
GRAVITATIONAL_PARAMETER = 3.986004418e14
# The satellites, the seed of their timestamps' noise, the error of a range in metres, set far above a real
# receiver's so that the uncertainty shows at the scale of the orbits, and of the receiver's clock in seconds, and
# each satellite's phase along its orbit, in degrees, offset from the scene's.
N_SATELLITES, NOISE_SEED = 6, 7
RANGE_ERROR_METRES, CLOCK_ERROR_SECONDS = 1_500_000, 30e-6
PHASE_OFFSETS = np.deg2rad([-12, -5, 8, 15, -9, 4])
# The fit: the steps finding the retarded transmissions and the steps of the fit, and the start's error off the
# receiver, along y and z, in Earth radii.
N_EMISSION_STEPS, N_POSITION_STEPS = 3, 5
INITIAL_Y, INITIAL_Z = 0.02, 0.01
# The scene's phase, and the phases the satellites move between; the frames and the duration of a frame.
PHASE = np.deg2rad(20)
FIRST_PHASE, LAST_PHASE = np.deg2rad([5, 50])
N_FRAMES, DURATION_MS = 60, 65

# Six circular orbits sharing the polar axis, their distinct phases giving different elevations, which separates
# radial position error from clock error; their radius and angular speed; the receiver on Earth's surface; a
# timestamp's error; the receiver clock's reading off; and the noise on each satellite's timestamp.
PLANES = (mv.yz * (-np.arange(N_SATELLITES) * (2 * np.pi / N_SATELLITES) / 2)).exp() >> mv.xy   # [satellites] Orbit+
ORBIT_RADIUS = (EARTH_RADIUS_METRES + ORBIT_ALTITUDE_METRES) / EARTH_RADIUS_METRES
ANGULAR_SPEED = (np.sqrt(GRAVITATIONAL_PARAMETER / (EARTH_RADIUS_METRES + ORBIT_ALTITUDE_METRES) ** 3)
                 * EARTH_RADIUS_METRES / LIGHT_SPEED)
RECEIVER = origin + across                                                   # [] Event-
SIGMA = RANGE_ERROR_METRES / EARTH_RADIUS_METRES
CLOCK_READING = mv.scalar([CLOCK_ERROR_SECONDS * LIGHT_SPEED / EARTH_RADIUS_METRES])   # [] Scalar+
NOISE = mv.scalar(np.random.default_rng(NOISE_SEED).normal(size=(N_SATELLITES, 1)) * SIGMA)   # [satellites] Scalar+


# --- math -----------------------------------------------------------------------------
def positioning(phase: float) -> tuple[core.Event, core.Event, core.Covariance]:
    """The transmissions at the scene's phase, and the reception event recovered from their six noisy timestamps,
    with its covariance."""
    phases = mv.scalar((PHASE_OFFSETS + phase)[..., None])                   # [satellites] Scalar+
    emitted = emitted_events(RECEIVER, PLANES, phases, ORBIT_RADIUS, ANGULAR_SPEED, N_EMISSION_STEPS)
    observed = emitted + time * NOISE
    # The starting event uses the receiver's clock and an approximate location.
    initial = RECEIVER + mv.y.dual() * INITIAL_Y + mv.z.dual() * INITIAL_Z + time * CLOCK_READING
    estimate, _, covariance = locate(observed, initial, SIGMA, N_POSITION_STEPS)
    return emitted, estimate, covariance


def uncertainty(n_frames: int) -> Iterator[tuple[core.Event, core.Event, core.Event, core.Covariance, core.Event]]:
    """The satellites moving from a narrow sky patch to a wider spread of directions, per frame, with the position's
    spread."""
    for phase in np.linspace(FIRST_PHASE, LAST_PHASE, n_frames):
        emitted, estimate, covariance = positioning(phase)
        yield emitted, RECEIVER, estimate, covariance, core.position_spread(covariance)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.gps import render
    from examples.util.animation import save_animation, save_figure

    emitted, estimate, covariance = positioning(PHASE)
    save_figure(render.draw_constellation(emitted, RECEIVER, ORBIT_RADIUS), "gps_constellation")
    save_figure(render.draw_solution(emitted, RECEIVER, estimate, covariance, core.position_spread(covariance),
                                     ORBIT_RADIUS), "gps_positioning")
    save_animation(render.animate_geometry(uncertainty(N_FRAMES), ORBIT_RADIUS),
                   "gps_uncertainty", DURATION_MS, forward_then_back=True)


if __name__ == "__main__":
    main()
