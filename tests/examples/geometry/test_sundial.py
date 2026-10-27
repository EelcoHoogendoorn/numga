"""Tests for the horizontal sundial verify the sun's noon altitude, the seasonal shadow rays on their cones, the gnomonic hour-line angles, and the rendered figures."""

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.sundial import core, render, scenarios


def test_the_noon_sun_stands_a_quarter_turn_less_the_latitude_plus_the_declination_above_the_horizon():
    pole, equator = core.sky(scenarios.LATITUDE)
    declinations = np.linspace(-scenarios.TILT, scenarios.TILT, 5)
    noon = core.sun(pole, equator, declinations, np.zeros_like(declinations))   # [declinations] Point-
    elevations = np.arcsin((core.dial & noon).to_array())
    np.testing.assert_allclose(elevations, np.pi / 2 - scenarios.LATITUDE + declinations, atol=1e-12)


def test_shadows_lie_on_their_cones_and_hour_lines_on_the_gnomonic_angle():
    nodus, seasons, days, suns, _ = scenarios.dial()
    pole, _ = core.sky(scenarios.LATITUDE)
    rays = (nodus - core.shadow(nodus, days)).normalized()                      # [seasons, samples] Point-
    assert np.abs((core.cone(pole, days[:, 0])[:, None](rays) & rays).to_array()).max() < 1e-10
    offsets = core.shadow(nodus, suns) - core.shadow(nodus, pole)               # [hours, declinations] Point-
    angles = np.arctan2((core.mv.x & offsets).to_array(), (core.mv.y & offsets).to_array())
    expected = np.arctan2(np.sin(scenarios.LATITUDE) * np.sin(scenarios.HOUR_ANGLES), np.cos(scenarios.HOUR_ANGLES))
    assert np.abs(np.sin(angles - expected[:, None])).max() < 1e-10


def test_the_dial_and_a_day_of_the_year_draw():
    nodus, seasons, days, suns, hours = scenarios.dial()
    pole, _ = core.sky(scenarios.LATITUDE)
    foot = nodus - pole * ((core.dial & nodus) / (core.dial & pole))            # [] Point-
    plt.close(render.draw_dial(nodus, foot, seasons, days, suns, hours))
    declinations, rays, year_days = scenarios.year()
    assert len(list(render.animate_year(declinations[:1], rays[:1], year_days[:1], nodus, foot, seasons, days, suns, hours))) == 1
