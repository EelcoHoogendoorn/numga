"""Pose filtering on the motor manifold in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.geometry.kalman import render, scenarios


def test_filter_beats_dead_reckoning_and_renders():
    """Sparse noisy pose readings keep the filtered path far closer to the truth than the
    dead-reckoned one, and the figure renders."""
    tracking = scenarios.tracking()
    *_, dead_error, filtered_error, _ = tracking
    assert filtered_error.mean(axis=0).to_array() < 0.25 * dead_error.mean(axis=0).to_array()

    figure = render.draw_tracking(*tracking)
    assert isinstance(figure, plt.Figure)
