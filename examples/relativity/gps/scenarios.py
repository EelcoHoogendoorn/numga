"""Polar GPS satellite sky-coverage scenarios with exaggerated clock and range noise."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.gps import core
from examples.relativity.gps.core import across, emitted_events, fitted_reception, mv, origin, time

# Earth radius, GPS orbital altitude, speed of light, and geocentric gravitational parameter in SI units.
earth_radius_metres = 6_371_000
orbit_altitude_metres = 20_200_000
light_speed = 299_792_458
gravitational_parameter = 3.986004418e14
# Constellation count, RNG seed, exaggerated range error (metres) for orbital-scale visibility, and clock offset (seconds).
n_satellites, noise_seed = 6, 7
range_error_metres, clock_error_seconds = 1_500_000, 30e-6
phase_offsets = np.deg2rad([-12, -5, 8, 15, -9, 4])
# Retarded-time and Gauss-Newton iteration counts, plus initial transverse position offset in Earth radii.
n_emission_steps, n_position_steps = 3, 5
initial_y, initial_z = 0.02, 0.01
# Reference snapshot phase, animation phase bounds, half-cycle frame count, and frame duration in milliseconds.
snapshot_phase = np.deg2rad(20)
first_phase, last_phase = np.deg2rad([5, 50])
n_frames, frame_duration_ms = 98, 40

# Six polar orbital planes spaced around the polar `x` axis, dimensionless orbit radius and angular speed, and noise draw.
orbit_rotors = (mv.yz * (-np.arange(n_satellites) * (2 * np.pi / n_satellites) / 2)).exp()  # [n_satellites] Rotor+
orbital_planes = orbit_rotors >> mv.xy                                            # [n_satellites] Bivector+
orbit_radius = (earth_radius_metres + orbit_altitude_metres) / earth_radius_metres
angular_speed = (np.sqrt(gravitational_parameter / (earth_radius_metres + orbit_altitude_metres) ** 3)
                 * earth_radius_metres / light_speed)
receiver = origin + across                                                        # [] Event-
sigma = range_error_metres / earth_radius_metres
clock_offset_units = clock_error_seconds * light_speed / earth_radius_metres
timestamp_noise = np.random.default_rng(noise_seed).normal(size=n_satellites) * sigma


# --- math -----------------------------------------------------------------------------
def navigation_snapshot(phase: float) -> tuple[core.Event, core.Event, core.Covariance]:
    """Retarded transmissions and Gauss-Newton light-cone fitting at orbital `phase` give `(emitted,
    estimate, covariance)` (`[n_satellites] Event-`, `[] Event-`, `[] Step- <- Readout+`)."""
    phases = phase_offsets + phase
    emitted = emitted_events(receiver, orbital_planes, phases, orbit_radius, angular_speed, n_emission_steps)
    observed = emitted + time * timestamp_noise                                   # [n_satellites] Event-
    initial = receiver + mv.y.dual() * initial_y + mv.z.dual() * initial_z + time * clock_offset_units  # [] Event-
    estimate, _, covariance = fitted_reception(observed, initial, sigma, n_position_steps)
    return emitted, estimate, covariance


def sky_spread_states(n_steps: int) -> Iterator[tuple[core.Event, core.Event, core.Event, core.Event]]:
    """Yields `(emitted, receiver, estimate, semi_axes)` across `n_steps` orbital phases as the
    satellites fan out from a narrow polar cluster across the sky."""
    for phase in np.linspace(first_phase, last_phase, n_steps):
        emitted, estimate, covariance = navigation_snapshot(phase)
        yield emitted, receiver, estimate, core.position_semi_axes(covariance)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.gps import render
    from examples.util.animation import save_animation, save_figure

    emitted, estimate, covariance = navigation_snapshot(snapshot_phase)
    save_figure(render.constellation_figure(emitted, receiver, orbit_radius), "gps_constellation")
    save_figure(render.positioning_figure(emitted, receiver, estimate, core.position_semi_axes(covariance),
                                          orbit_radius), "gps_positioning")

    recorded: list[tuple[core.Event, core.Event, core.Event, core.Event]] = []

    def recording() -> Iterator[tuple[core.Event, core.Event, core.Event, core.Event]]:
        for state in sky_spread_states(n_frames):
            recorded.append(state)
            yield state

    save_animation(render.geometry_animation(recording()), "gps_uncertainty", frame_duration_ms)

    # checks
    # Four noiseless cones recover the exact reception event, and spreading the satellites shrinks ellipsoid volume.
    noiseless_cones = core.light_cones(emitted[:4])                               # [4] Slice+ <- Event-
    located = core.causal_reception(core.cone_intersections(noiseless_cones), emitted[:4])  # [] Event-
    assert ((mv.basis() & (located - receiver)[..., None]).abs() < 1e-11).all()
    narrow_radii = core.spatial_distance(origin + recorded[0][3], origin)         # [3] Scalar+
    wide_radii = core.spatial_distance(origin + recorded[-1][3], origin)          # [3] Scalar+
    assert (wide_radii < narrow_radii).all()
    assert bool(wide_radii[0] * wide_radii[1] * wide_radii[2] < 0.4 * narrow_radii[0] * narrow_radii[1] * narrow_radii[2])


def pinned() -> None:
    """Regenerates the gallery GIF `plots/gps.gif` of the shrinking position uncertainty ellipsoid."""
    from examples.relativity.gps import render
    from examples.util.animation import pin_animation

    pin_animation(render.geometry_animation(sky_spread_states(n_frames)), "gps", frame_duration_ms)


if __name__ == "__main__":
    main()
