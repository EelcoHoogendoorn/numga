"""A six-dimensional rotation: its bivector splits into the same three commuting planes by its wedge
powers and by the spectrum of the map `(Vector | B) | B`, and the orbit draws."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.math import invariant_decomposition


def test_main_passes_its_checks_and_draws():
    parts, points, circles = invariant_decomposition.main(60, 1)
    figure = invariant_decomposition.draw(points, circles, parts, 30)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(invariant_decomposition.animate(points, circles, parts, 3)) == 3
