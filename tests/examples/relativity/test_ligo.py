"""Coverage tests for LIGO gravitational-wave antenna patterns and sky-map / ring renderers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.ligo import core, render, scenarios


def test_pole_detector_matches_analytic_antenna_pattern():
    """For orthogonal horizontal arms at the north pole, squared sensitivity equals `((1 + cos(theta)**2) / 2)**2 * cos(2 * phi)**2 + cos(theta)**2 * sin(2 * phi)**2`."""
    pole = core.earth_frame(np.array(np.pi / 2), np.array(0))                           # [] Rotor+
    arms = core.detector_arms(pole, np.radians([0, 90]))                                # [2] Point-
    latitudes = np.radians(np.linspace(-80, 80, 9))[:, None]
    longitudes = np.radians(np.linspace(-170, 170, 13))[None, :]
    plus, cross = core.wave_strains(core.earth_frame(latitudes, longitudes))
    seen = core.directional_sensitivity(core.arm_response(arms, plus), core.arm_response(arms, cross))
    polar, azimuth = np.pi / 2 - latitudes, longitudes
    expected = ((1 + np.cos(polar) ** 2) / 2) ** 2 * np.cos(2 * azimuth) ** 2 + np.cos(polar) ** 2 * np.sin(2 * azimuth) ** 2
    assert ((seen.squared() - expected).abs() < 1e-10).all()
    assert ((core.mv.t & arms).abs() < 1e-12).all()


def test_ligo_figures_and_frame_render():
    sites, arms = scenarios.site_geometries()
    sky_frames = scenarios.sky_grid(5)
    first_axis = core.mv.x.dual()
    directions, zeniths = sky_frames >> first_axis, sites >> first_axis
    plus_response, cross_response = scenarios.sky_patterns(arms, sky_frames)
    sensitivities = core.directional_sensitivity(plus_response, cross_response)
    plt.close(
        render.patterns_figure(
            directions,
            plus_response,
            cross_response,
            sensitivities,
            zeniths,
            scenarios.detector_names,
        )
    )
    plt.close(
        render.network_figure(
            directions,
            sensitivities[0],
            sensitivities[1],
            zeniths,
            scenarios.detector_names,
        )
    )
    rest_ring = scenarios.horizontal_ring(sites[0], 8)
    states = core.polarization_states(sites[0], arms[0], sky_frames, np.zeros(1), rest_ring, scenarios.strain_scale)
    frames = list(
        render.polarization_animation(
            directions,
            sites[0],
            arms[0],
            scenarios.detector_names[0],
            scenarios.polarization_size,
            states,
        )
    )
    assert len(frames) == 1
