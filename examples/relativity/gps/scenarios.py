"""Satellite navigation near Earth's pole, with changing sky coverage and noisy clocks."""

from __future__ import annotations

import numpy as np

from examples.relativity.gps.core import (
    across, emitted_events, locate, mv, origin, time,
)

EARTH_RADIUS_METRES = 6_371_000


def geometry_inputs(phase: float):
    """Six circular satellite orbits and one receiver, in Earth-radius and light-travel units."""
    earth_radius, orbit_altitude = EARTH_RADIUS_METRES, 20_200_000
    light_speed, gravitational_parameter = 299_792_458, 3.986004418e14
    n_satellites, noise_seed = 6, 7
    range_error, clock_error = 3, 30e-6
    azimuths = np.arange(n_satellites) * (2 * np.pi / n_satellites)
    phase_offsets = np.deg2rad([-12, -5, 8, 15, -9, 4])

    # These circular planes share the polar axis. Their distinct phases give
    # different elevations, separating radial position error from clock error.
    planes = (mv.yz * (-azimuths / 2)).exp() >> mv.xy
    phases = mv.scalar((phase_offsets + phase)[..., None])
    radius = (earth_radius + orbit_altitude) / earth_radius
    angular_speed = np.sqrt(gravitational_parameter / (earth_radius + orbit_altitude) ** 3)
    angular_speed *= earth_radius / light_speed
    receiver = origin + across
    sigma = range_error / earth_radius
    clock_reading = mv.scalar([clock_error * light_speed / earth_radius])
    noise = mv.scalar(np.random.default_rng(noise_seed).normal(size=(n_satellites, 1)) * sigma)
    return planes, phases, radius, angular_speed, receiver, sigma, clock_reading, noise


def positioning(phase: float):
    """Recover one reception event from six noisy satellite timestamps."""
    planes, phases, radius, angular_speed, receiver, sigma, clock_reading, noise = geometry_inputs(phase)
    n_emission_steps, n_position_steps = 3, 5
    initial_y, initial_z = 0.02, 0.01
    emitted = emitted_events(receiver, planes, phases, radius, angular_speed, n_emission_steps)
    observed = emitted + time * noise
    # The starting event uses the receiver's clock and an approximate location.
    initial = receiver + mv.y.dual() * initial_y + mv.z.dual() * initial_z + time * clock_reading
    estimate, information, covariance = locate(observed, initial, sigma, n_position_steps)
    return emitted, receiver, estimate, information, covariance, clock_reading, sigma


def uncertainty(n_frames: int):
    """Move the satellites from a narrow sky patch to a wider spread of directions."""
    first_phase, last_phase = np.deg2rad([5, 50])
    for phase in np.linspace(first_phase, last_phase, n_frames):
        emitted, receiver, estimate, _, covariance, _, _ = positioning(phase)
        yield emitted, receiver, estimate, covariance


if __name__ == "__main__":
    from examples.relativity.gps import render
    from examples.util.animation import save_animation, save_figure

    phase = np.deg2rad(20)
    n_frames, duration_ms = 60, 65
    positioning_extent_metres = 40
    uncertainty_extent_metres = 120
    _, _, radius, _, _, _, _, _ = geometry_inputs(phase)
    emitted, receiver, estimate, _, covariance, _, _ = positioning(phase)
    save_figure(render.draw_constellation(emitted, receiver, radius), "gps_constellation")
    save_figure(render.draw_solution(emitted, receiver, estimate, covariance, radius,
                                    EARTH_RADIUS_METRES, positioning_extent_metres), "gps_positioning")
    save_animation(render.animate_geometry(uncertainty(n_frames), radius, EARTH_RADIUS_METRES,
                                           uncertainty_extent_metres),
                   "gps_uncertainty", duration_ms, forward_then_back=True)
