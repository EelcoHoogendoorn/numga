"""Scenes for gravitational-wave antenna patterns place the Hanford and Livingston LIGO detectors on Earth, evaluate their all-sky responses, and sweep Hanford's response over a half-turn polarization cycle."""

from __future__ import annotations

from collections.abc import Iterator
import numpy as np

from numga import stack
from examples.relativity.ligo import core

# Detector names, latitudes and longitudes in degrees, and arm azimuths in degrees clockwise from north.
detector_names = ["Hanford", "Livingston"]
site_latitudes = [46.4553, 30.5631]
site_longitudes = [-119.4078, -90.7744]
arm_azimuths = [[324, 234], [252, 162]]
n_detectors = len(detector_names)

# Sky grid counts: latitudes from south pole to north pole and longitudes once around.
n_sky_samples = 61

# Polarization sweep counts, free-mass ring resolution, exaggerated ring strain, and frame duration.
n_frames = 96
n_masses = 24
strain_scale = 0.3
frame_duration_ms = 40

# Animation pixel sizes and gallery palette size.
polarization_size = (600, 200)
gallery_size = (720, 240)
n_gallery_colours = 48


def site_geometries() -> tuple[core.Rotor, core.Point]:
    """Site frames (`[2] Rotor+`) and unit arm displacements (`[2, 2] Point-`) place the Hanford and Livingston interferometers."""
    sites = core.earth_frame(np.radians(site_latitudes), np.radians(site_longitudes))   # [2] Rotor+
    arms = core.detector_arms(sites, np.radians(arm_azimuths))                          # [2, 2] Point-
    return sites, arms


def sky_grid(n_samples: int) -> core.Rotor:
    """Arrival frames (`[n_samples, 2 * n_samples - 1] Rotor+`) cover the celestial sphere from pole to pole."""
    latitudes = np.radians(np.linspace(-90, 90, n_samples))
    longitudes = np.radians(np.linspace(-180, 180, 2 * n_samples - 1))
    return core.earth_frame(latitudes[:, None], longitudes)                             # [n_samples, 2 * n_samples - 1] Rotor+


def sky_patterns(arms: core.Point, sky_frames: core.Rotor) -> tuple[core.Scalar, core.Scalar]:
    """Plus and cross antenna responses (`[n_detectors, n_latitudes, n_longitudes] Scalar+`) evaluate every detector over `sky_frames`."""
    plus, cross = core.wave_strains(sky_frames)                                         # [..., ...] Plane+ <- Point-, Plane+ <- Point-
    return core.arm_response(arms[:, None, None], plus), core.arm_response(arms[:, None, None], cross)


def horizontal_ring(site: core.Rotor, n_ring_masses: int) -> core.Point:
    """Equally spaced unit displacements (`[n_ring_masses] Point-`) form a horizontal test-mass ring at `site`."""
    east, north = site >> core.mv.y, site >> core.mv.z                                  # [] Plane+, Plane+
    turns = np.linspace(0, 2 * np.pi, n_ring_masses, endpoint=False)
    ring_rotors = ((east ^ north) * (-turns / 2)).exp()                                 # [n_ring_masses] Rotor+
    return ring_rotors >> east.dual()                                                   # [n_ring_masses] Point-


def main() -> None:
    from examples.relativity.ligo import render
    from examples.util.animation import save_animation, save_figure

    sites, arms = site_geometries()                                                     # [2] Rotor+, [2, 2] Point-
    sky_frames = sky_grid(n_sky_samples)                                                # [n_sky_samples, 2 * n_sky_samples - 1] Rotor+
    first_axis = core.mv.x.dual()                                                       # Antivector-
    directions = sky_frames >> first_axis                                               # [n_sky_samples, 2 * n_sky_samples - 1] Point-
    zeniths = sites >> first_axis                                                       # [2] Point-

    plus_response, cross_response = sky_patterns(arms, sky_frames)                      # [2, n_sky_samples, 2 * n_sky_samples - 1] Scalar+, Scalar+
    sensitivities = core.directional_sensitivity(plus_response, cross_response)         # [2, n_sky_samples, 2 * n_sky_samples - 1] Scalar+
    network_sensitivity = sensitivities.squared().sum(axis=0).square_root()             # [n_sky_samples, 2 * n_sky_samples - 1] Scalar+
    # Cosine alignment between the two detectors' polarization response pairs: -1 where their signs oppose.
    alignment = (plus_response[0] * plus_response[1] + cross_response[0] * cross_response[1]) / (
        sensitivities[0] * sensitivities[1]
    )                                                                                   # [n_sky_samples, 2 * n_sky_samples - 1] Scalar+

    save_figure(
        render.patterns_figure(directions, plus_response, cross_response, sensitivities, zeniths, detector_names),
        "ligo_patterns",
    )
    save_figure(
        render.network_figure(directions, network_sensitivity, alignment, zeniths, detector_names),
        "ligo_network",
    )

    angles = np.linspace(0, np.pi, n_frames, endpoint=False)
    rest_ring = horizontal_ring(sites[0], n_masses)                                     # [n_masses] Point-
    kept_responses: list[core.Scalar] = []

    def recorded_states() -> Iterator[tuple[core.Scalar, core.Point]]:
        for state in core.polarization_states(sites[0], arms[0], sky_frames, angles, rest_ring, strain_scale):
            kept_responses.append(state[0])
            yield state

    save_animation(
        render.polarization_animation(
            directions,
            sites[0],
            arms[0],
            detector_names[0],
            polarization_size,
            recorded_states(),
        ),
        "ligo_polarization",
        frame_duration_ms,
    )
    swept_responses = stack(kept_responses)                                             # [n_frames, n_sky_samples, 2 * n_sky_samples - 1] Scalar+

    # checks
    # Sensitivity peaks at 1 straight overhead each site and never exceeds 1 anywhere on the sky.
    overhead_plus, overhead_cross = core.wave_strains(sites)                            # [2] Plane+ <- Point-, Plane+ <- Point-
    overhead_sensitivity = core.directional_sensitivity(
        core.arm_response(arms, overhead_plus),
        core.arm_response(arms, overhead_cross),
    )                                                                                   # [2] Scalar+
    assert ((overhead_sensitivity - 1).abs() < 1e-10).all()
    assert (sensitivities <= 1 + 1e-10).all()

    # Waves arriving along the horizontal arm bisector produce zero response in both polarizations.
    bisector = arms[:, 0] + arms[:, 1]
    bisector = bisector / bisector.antinorm()                                           # [2] Point-
    bisector_frame = (1 + bisector * first_axis).normalized()                           # [2] Rotor+
    blind_plus, blind_cross = core.wave_strains(bisector_frame)                         # [2] Plane+ <- Point-, Plane+ <- Point-
    blind_sensitivity = core.directional_sensitivity(
        core.arm_response(arms, blind_plus),
        core.arm_response(arms, blind_cross),
    )                                                                                   # [2] Scalar+
    assert (blind_sensitivity.abs() < 1e-10).all()

    # Spin-two law: rotating transverse axes by `angles` mixes plus and cross responses by `2 * angles`.
    mixed = (
        plus_response[0] * np.cos(2 * angles)[:, None, None]
        + cross_response[0] * np.sin(2 * angles)[:, None, None]
    )                                                                                   # [n_frames, n_sky_samples, 2 * n_sky_samples - 1] Scalar+
    assert ((swept_responses - mixed).abs() < 1e-10).all()

    # Hanford and Livingston are oriented nearly opposite, giving opposite-sign responses across >90% of the sky.
    assert np.mean(alignment < -0.5) > 0.9


def pinned() -> None:
    """Gallery animation for `examples.pinned`: Hanford's response and overhead ring over a half-turn polarization sweep."""
    from examples.relativity.ligo import render
    from examples.util.animation import pin_animation

    sites, arms = site_geometries()                                                     # [2] Rotor+, [2, 2] Point-
    sky_frames = sky_grid(n_sky_samples)                                                # [n_sky_samples, 2 * n_sky_samples - 1] Rotor+
    directions = sky_frames >> core.mv.x.dual()                                         # [n_sky_samples, 2 * n_sky_samples - 1] Point-
    angles = np.linspace(0, np.pi, n_frames, endpoint=False)
    rest_ring = horizontal_ring(sites[0], n_masses)                                     # [n_masses] Point-
    states = core.polarization_states(sites[0], arms[0], sky_frames, angles, rest_ring, strain_scale)
    pin_animation(
        render.polarization_animation(directions, sites[0], arms[0], detector_names[0], gallery_size, states),
        "ligo",
        frame_duration_ms,
        colors=n_gallery_colours,
    )


if __name__ == "__main__":
    main()
