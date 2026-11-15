"""Tests for the horizontal sundial verify the sun's noon altitude, the seasonal shadow rays on their cones, the gnomonic hour-line angles, the static dial figure, and a single animation frame."""

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.sundial import core, render, scenarios


def test_noon_sun_altitude_matches_colatitude_plus_declination():
    pole, equator = core.sky(scenarios.latitude)                                         # [] Point-, Point-
    n_declinations = 5
    declinations = np.linspace(-scenarios.tilt, scenarios.tilt, n_declinations)          # [n_declinations]
    noon_suns = core.sun(pole, equator, declinations, np.zeros_like(declinations))       # [n_declinations] Point-
    expected_heights = np.sin(np.pi / 2 - scenarios.latitude + declinations)             # [n_declinations]
    assert (((core.dial & noon_suns) - expected_heights).abs() < 1e-12).all()


def test_shadows_lie_on_cones_and_hour_lines_follow_gnomonic_angle():
    nodus, seasons, days, suns, drawn_hours = scenarios.dial_scene()
    del seasons, drawn_hours
    pole, equator = core.sky(scenarios.latitude)                                         # [] Point-, Point-
    del equator
    rays = (nodus - core.shadow(nodus, days)).normalized()                               # [n_seasons, n_samples] Point-
    assert ((core.cone(pole, days[:, 0])[:, None](rays) & rays).abs() < 1e-10).all()
    offsets = (core.shadow(nodus, suns) - core.shadow(nodus, pole)).normalized()         # [n_hours, n_declinations] Point-
    expected_east = np.sin(scenarios.latitude) * np.sin(scenarios.hour_angles)[:, None]
    expected_north = np.cos(scenarios.hour_angles)[:, None]
    residuals = expected_north * (core.mv.x & offsets) - expected_east * (core.mv.y & offsets)  # [n_hours, n_declinations] Scalar+
    assert (residuals.abs() < 1e-10).all()


def test_dial_figure_and_first_year_frame_render():
    nodus, seasons, days, suns, drawn_hours = scenarios.dial_scene()
    pole, equator = core.sky(scenarios.latitude)                                         # [] Point-, Point-
    foot = core.shadow(nodus, pole)                                                      # [] Point-
    figure = render.dial_figure(nodus, foot, seasons, days, suns, drawn_hours)
    plt.close(figure)
    states = core.year_states(pole, equator, np.array([0.0]), scenarios.ray_angles[:4], scenarios.day_angles[:8])
    frame = next(render.year_animation(nodus, foot, seasons, days, suns, drawn_hours, scenarios.annual_envelope()[:2], states))
    assert frame.ndim == 3 and frame.shape[2] == 3
