"""The rainbow: each bow lies at Descartes's angle, the path inside the drop is the powers of one turn,
and the scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.optics.rainbow import core, render, scenarios

forward, up = scenarios.forward, scenarios.up


def test_bows_lie_at_descartes_angles():
    """The extreme angle of the light leaving after one and two reflections, for every colour."""
    apart, _ = scenarios.sky(up * ((np.arange(scenarios.N_RAYS) + 0.5) / scenarios.N_RAYS))
    n = scenarios.indices()[:, 0]
    reflections = np.arange(1, scenarios.N_REFLECTIONS + 1)[:, None]          # [orders, 1]
    incidence = np.arccos(np.sqrt((n**2 - 1) / (reflections * (reflections + 2))))
    refraction = np.arcsin(np.sin(incidence) / n)
    deflection = 2 * (incidence - refraction) + reflections * (np.pi - 2 * refraction)
    cosines = apart.to_array()                                                # [orders, colours, rays]
    np.testing.assert_allclose([cosines[0].min(axis=-1), cosines[1].max(axis=-1)], -np.cos(deflection), atol=1e-6)


def test_light_inside_reflects_off_the_surface():
    """Each chord inside the drop, turned by the bounce, is the last one reflected off the surface where
    they meet, and the light meets the surface at the same angle at both ends of every chord."""
    offsets = up * np.linspace(0.1, 0.95, 7)
    vertex = core.entry(forward, offsets)
    chord = core.refract(forward, -vertex, 1.333)
    turn = core.bounce(vertex, chord)
    following = turn >> vertex
    np.testing.assert_allclose(((turn >> chord) + (following >> chord)).norm().to_array(), 0, atol=1e-12)
    np.testing.assert_allclose((chord | vertex).to_array(), -(chord | following).to_array(), atol=1e-12)


def test_scenes_draw():
    offsets = up * ((np.arange(scenarios.N_RAYS) + 0.5) / scenarios.N_RAYS)
    apart, flux = scenarios.sky(offsets)
    figure = render.draw_sky(apart, flux, scenarios.WAVELENGTHS, scenarios.SUN_ELEVATION, scenarios.SUN_WIDTH)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    swept = scenarios.sweep(up * np.array([0.3, 0.86]))
    bundle = scenarios.sweep(up * np.linspace(0.1, 0.9, 5))
    assert len(render.animate_sweep(swept, bundle, offsets, apart, scenarios.WAVELENGTHS)) == 2
