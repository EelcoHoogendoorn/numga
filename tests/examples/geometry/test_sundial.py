"""A horizontal sundial: the sun's noon height, the seasons' shadows on their cones of rays, the hour lines
at the angle of gnomonics, and the figures."""

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.sundial import core, render, scenarios


def test_the_noon_sun_stands_a_quarter_turn_less_the_latitude_plus_the_declination_above_the_horizon():
    pole, equator = core.sky(scenarios.LATITUDE)
    declinations = np.linspace(-scenarios.TILT, scenarios.TILT, 5)
    noon = core.sun(pole, equator, declinations, np.zeros_like(declinations))   # [declinations] Point
    elevations = np.arcsin((core.dial & noon).to_array())
    np.testing.assert_allclose(elevations, np.pi / 2 - scenarios.LATITUDE + declinations, atol=1e-12)


def test_shadows_lie_on_their_cones_and_hour_lines_on_the_gnomonic_angle():
    nodus, seasons, days, suns, hours = scenarios.dial()
    pole, _ = core.sky(scenarios.LATITUDE)
    rays = (nodus - core.shadow(nodus, days)).normalized()                      # [seasons, samples] Point
    assert np.abs((core.cone(pole, days[:, 0])[:, None](rays) & rays).to_array()).max() < 1e-10
    foot = core.shadow(nodus, pole)
    angles = render.bearings(core.shadow(nodus, suns) - foot)                   # [hours, samples]
    hour_angles = scenarios.hour_angle(hours)
    expected = np.arctan2(np.sin(scenarios.LATITUDE) * np.sin(hour_angles), np.cos(hour_angles))
    assert np.abs(np.sin(angles - expected[:, None])).max() < 1e-10


def test_the_dial_and_a_day_of_the_year_draw():
    nodus, seasons, days, suns, hours = scenarios.dial()
    pole, _ = core.sky(scenarios.LATITUDE)
    foot = core.shadow(nodus, pole)
    plt.close(render.draw_dial(nodus, foot, seasons, days, suns, hours))
    declinations, rays, year_days = scenarios.year()
    assert len(render.animate_year(declinations[:1], rays[:1], year_days[:1], nodus, foot, seasons, days, suns, hours)) == 1
