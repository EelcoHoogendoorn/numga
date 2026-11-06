"""The rainbow: the path inside the drop is the powers of one turn."""

from __future__ import annotations

import numpy as np

from examples.optics.rainbow import core, render, scenarios

forward, up = scenarios.forward, scenarios.up


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


def test_one_frame_of_the_sweep_draws():
    offsets = up * np.array([0.3, 0.86])
    sight_cosines, _ = scenarios.sky(offsets)
    next(render.animate_sweep(scenarios.sweep(offsets[:1]), scenarios.sweep(offsets), offsets, sight_cosines,
                              scenarios.wavelengths))
