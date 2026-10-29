"""The rainbow: each bow lies at Descartes's angle, the path inside the drop is the powers of one turn,
and the scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.rainbow import core, render, scenarios

forward, up = scenarios.forward, scenarios.up


def test_bows_lie_at_descartes_angles():
    """The extreme angle of the light leaving after one and two reflections, for every colour."""
    sight_cosines, _ = scenarios.sky(up * ((np.arange(scenarios.n_rays) + 0.5) / scenarios.n_rays))
    index = scenarios.indices[:, 0]
    reflections = np.arange(1, scenarios.n_reflections + 1)[:, None]          # [orders, 1]
    incidence = np.arccos(np.sqrt((index**2 - 1) / (reflections * (reflections + 2))))
    refraction = np.arcsin(np.sin(incidence) / index)
    deflection = 2 * (incidence - refraction) + reflections * (np.pi - 2 * refraction)
    cosines = sight_cosines.to_array()                                        # [orders, colours, rays]
    np.testing.assert_allclose([cosines[0].min(axis=-1), cosines[1].max(axis=-1)], -np.cos(deflection), atol=1e-6)


def test_light_inside_reflects_off_the_surface():
    """Each chord inside the drop, turned by the bounce, is the last one reflected off the surface where
    they meet, and the light meets the surface at the same angle at both ends of every chord."""
    offsets = up * np.linspace(0.1, 0.95, 7)
    vertex = core.entry(forward, offsets)
    chord = core.refract(forward, -vertex, 1.333)
    turn = core.bounce(vertex, chord)
    next_vertex = turn >> vertex
    np.testing.assert_allclose(((turn >> chord) + (next_vertex >> chord)).norm().to_array(), 0, atol=1e-12)
    np.testing.assert_allclose(chord.dot(vertex).to_array(), -chord.dot(next_vertex).to_array(), atol=1e-12)


def test_scenes_draw():
    offsets = up * ((np.arange(scenarios.n_rays) + 0.5) / scenarios.n_rays)
    sight_cosines, flux = scenarios.sky(offsets)
    figure = render.draw_sky(sight_cosines, flux, scenarios.wavelengths, scenarios.sun_elevation, scenarios.sun_width)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    swept_light = scenarios.sweep(up * np.array([0.3, 0.86]))
    bundle = scenarios.sweep(up * np.linspace(0.1, 0.9, 5))
    assert len(list(render.animate_sweep(swept_light, bundle, offsets, sight_cosines, scenarios.wavelengths))) == 2
