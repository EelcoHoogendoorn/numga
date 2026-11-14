"""Daily swing precession of Foucault's pendulum matches the sine of the latitude clockwise in the north and agrees with spherical parallel-transport holonomy."""

import numpy as np

from examples.mechanics.foucault import core, render, scenarios


def test_daily_swing_precession_and_parallel_holonomy_agree():
    _, north, east = core.compass(scenarios.latitudes)                        # [n_latitudes] Point-, Point-, Point-
    floors = east.commutator(north)                                           # [n_latitudes] Bivector+
    expected = (floors * (-np.pi * np.sin(scenarios.latitudes))).exp() >> north  # [n_latitudes] Point-
    full_day = np.full(scenarios.n_latitudes, 2 * np.pi)
    day_ends = core.swing_direction(scenarios.latitudes, full_day)            # [n_latitudes] Point-
    assert ((day_ends - expected).antinorm() < 1e-10).all()

    early = core.swing_direction(scenarios.latitudes, np.full(scenarios.n_latitudes, 1e-3))  # [n_latitudes] Point-
    assert (-(early | east)[1:] > 0).all()

    holonomy = core.parallel_holonomy(scenarios.latitudes, scenarios.n_steps)  # [n_latitudes] Rotor+
    transported = holonomy >> north                                           # [n_latitudes] Point-
    assert ((transported - expected).antinorm() < 1e-4).all()
    assert ((core.turn_angle(holonomy, floors).sin() - np.sin(2 * np.pi * np.sin(scenarios.latitudes))).abs() < 1e-4).all()


def test_day_animation_frame_draws():
    vertical, north, east, states = scenarios.paris_day(1)
    frames = list(render.day_animation(vertical, north, east, states))
    assert len(frames) == 1
    assert frames[0].shape[1] == render.n_pixels
