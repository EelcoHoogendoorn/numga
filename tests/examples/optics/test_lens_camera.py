"""The lens camera stills, with their checks, and a short zoom animation."""

from __future__ import annotations


import matplotlib.pyplot as plt

from examples.optics.lens_camera import render, scenarios


def test_figure_and_animation_draw():
    """The stills run their checks; the figure and a short animation draw."""
    assert isinstance(render.draw_stills(scenarios.stills()), plt.Figure)
    frames = render.animate_camera(scenarios.motion(2), scenarios.SCENE)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
