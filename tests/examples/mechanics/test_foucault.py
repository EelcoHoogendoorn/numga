"""Foucault's pendulum: the swing turns by the felt part of the earth's turn, clockwise in the north, and a
direction carried around the latitude circle comes back turned the same way."""

import numpy as np

from examples.mechanics.foucault import core, render, scenarios


def test_a_day_turns_each_swing_by_the_sine_of_its_latitude_clockwise_and_the_holonomy_agrees():
    _, north, east = core.compass(scenarios.LATITUDES)
    expected = 2 * np.pi * np.sin(scenarios.LATITUDES)
    ends = core.swing(scenarios.LATITUDES, np.full(len(scenarios.LATITUDES), 2 * np.pi))
    assert np.abs(np.sin(render.turned(ends, north, east) - expected)).max() < 1e-10
    early = core.swing(scenarios.LATITUDES, np.full(len(scenarios.LATITUDES), 1e-3))
    assert (render.turned(early, north, east)[1:] > 0).all()
    carried = core.carried(scenarios.LATITUDES, scenarios.N_STEPS) >> north
    assert np.abs(np.sin(render.turned(carried, north, east) - expected)).max() < 1e-4


def test_a_frame_of_the_day_draws():
    earth, swings, vertical, north, east = scenarios.day()
    assert len(render.animate_day(earth[:2], swings[:2], vertical, north, east)) == 1
