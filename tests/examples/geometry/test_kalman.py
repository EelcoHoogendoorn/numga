"""Pose filtering on the motor manifold in PGA2D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.kalman import core, render, scenarios


def test_filter_beats_dead_reckoning_and_renders():
    """Sparse noisy pose readings keep the filtered path far closer to the truth than the
    dead-reckoned one, and the figure renders."""
    tracking = scenarios.tracking()
    *_, dead_error, filtered_error, _ = tracking
    assert filtered_error.mean(axis=0).to_array() < 0.25 * dead_error.mean(axis=0).to_array()

    figure = render.draw_tracking(*tracking)
    assert isinstance(figure, plt.Figure)


def test_covariance_stays_symmetric():
    """Prediction pushes the covariance through the step on both sides and the update subtracts a
    symmetric term, so as a form on lines the covariance stays symmetric through the filter."""
    mv = core.mv
    turns = np.array([[0.3, -0.2, 0.5], [0.1, 0.4, -0.3]])                # [readings, steps]
    steps = ((mv.xy * turns - mv.wx) * 0.05).exp()                         # [readings, steps] Motor
    measurements = ((mv.xy * np.array([0.2, -0.1]) - mv.wx) * 0.5).exp()   # [readings] Motor
    noise = core.covariance(0.1, 0.2)
    for _, sigma in core.kalman_filter(mv.rotor(), noise, steps, measurements, noise, core.covariance(0.3, 0.1)):
        form = (core.Line & sigma).kernel
        np.testing.assert_allclose(form, np.swapaxes(form, -1, -2), atol=1e-14)
