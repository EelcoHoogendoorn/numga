"""The manipulability scenarios check their ellipsoids against the joint torques; the figure and
the sweep render."""

from __future__ import annotations

from itertools import islice

import matplotlib.pyplot as plt

from examples.mechanics.manipulability import render, scenarios


def test_figure_renders():
    """Both poses pass their checks, and draw."""
    poses = [scenarios.ellipsoids(angles, 0.25, 2.5) for angles in (scenarios.REACHING, scenarios.NEARLY_STRAIGHT)]
    figure = render.draw_ellipsoids(poses, scenarios.VIEW, scenarios.CENTRE, scenarios.EXTENT, 48)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)


def test_sweep_renders():
    frames = [render.frame(pose, scenarios.VIEW, scenarios.CENTRE, scenarios.EXTENT, 40) for pose in islice(scenarios.sweep(12), 3)]
    assert frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
