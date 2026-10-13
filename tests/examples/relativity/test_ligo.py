"""Gravitational-wave antenna patterns: a detector's sensitivity agrees with the closed form for a
detector at the pole, and the figures draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.ligo import core, render, scenarios


def test_sensitivity_is_the_closed_form_for_a_detector_at_the_pole():
    """For arms along two horizontal axes at the pole, a wave from polar angle theta and azimuth phi
    from the first arm is seen with the squared sensitivity
    `((1 + cos(theta)**2) / 2)**2 * cos(2 * phi)**2 + cos(theta)**2 * sin(2 * phi)**2`."""
    pole = core.place(np.array(np.pi / 2), np.array(0))                      # [] Rotor
    arms = core.arms(pole, np.radians([0, 90]))                              # [2] Vector
    latitudes = np.radians(np.linspace(-80, 80, 9))[:, None]
    longitudes = np.radians(np.linspace(-170, 170, 13))[None, :]
    plus, cross = core.strains(core.place(latitudes, longitudes))
    seen = core.sensitivity(core.response(arms, plus), core.response(arms, cross))
    polar, azimuth = np.pi / 2 - latitudes, longitudes
    expected = ((1 + np.cos(polar)**2) / 2)**2 * np.cos(2 * azimuth)**2 + np.cos(polar)**2 * np.sin(2 * azimuth)**2
    assert np.abs(render.scalars(seen.squared()) - expected).max() < 1e-12


def test_figures_draw():
    sites, arms = scenarios.detectors()
    frames = scenarios.sky()
    directions, zeniths = frames >> core.x, sites >> core.x
    to_plus, to_cross = scenarios.patterns(arms, frames)
    sensitivities = core.sensitivity(to_plus, to_cross)
    plt.close(render.draw_patterns(directions, to_plus, to_cross, sensitivities, zeniths, scenarios.NAMES))
    plt.close(render.draw_network(directions, sensitivities[0], sensitivities[1], zeniths, scenarios.NAMES))
    responses, rings = scenarios.polarization(sites[0], arms[0], frames, np.zeros(1))
    assert len(render.animate_polarization(directions, responses, sites[0], arms[0], rings, scenarios.NAMES[0])) == 1
