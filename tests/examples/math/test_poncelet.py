"""Coverage tests for Poncelet's porism verifying five-sided path closure around the inscribed
pentagon's envelope, non-closure around the scaled pentagon's envelope, and single-frame rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt

from numga import stack
from examples.math.poncelet import render, scenarios


def test_paths_close_about_inscribed_pentagon_and_miss_about_scaled_pentagon():
    outer, inner, inside = scenarios.scene()
    vertices = stack(list(scenarios.circuit_paths(outer, inner, inside, 3)))
    assert ((outer(vertices) & vertices).abs() < 1e-10).all()
    gaps = (vertices[:, scenarios.n_sides] - vertices[:, 0]).antinorm()
    assert (gaps[:, 0] < 1e-9).all()
    assert (gaps[:, 1] > 1e-2).all()


def test_one_frame_of_the_turn_draws():
    outer, inner, inside = scenarios.scene()
    first_vertices = next(scenarios.circuit_paths(outer, inner, inside, 1))
    figure = render.paths_figure(outer, inner, first_vertices)
    plt.close(figure)
    next(render.path_frames(outer, inner, [first_vertices]))
