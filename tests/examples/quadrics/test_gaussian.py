"""Tests for the Gaussian fit and its 1σ quadric."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.gaussian import render, scenarios
from functools import partial

from examples.quadrics.gaussian.core import density, fit_gaussian, mv


def test_density_peaks_at_the_mean_and_level_matches_the_covariance():
    """The density is one at the sample mean, and the level is the squared Mahalanobis distance less one."""
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(500, 2)) * [2.0, 0.5] + [1.0, -1.0]
    queries = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, -3.0]])
    precision, ellipse = fit_gaussian(mv.point(xy))
    samples = mv.point(np.concatenate([xy.mean(axis=0)[None], queries]))
    level = ellipse(samples) & samples
    np.testing.assert_allclose(density(precision, samples).to_array()[0], 1.0, atol=1e-12)
    offset = queries - xy.mean(axis=0)
    covariance = np.cov(xy.T, bias=True)
    mahalanobis = np.einsum("ni,ij,nj->n", offset, np.linalg.inv(covariance), offset)
    np.testing.assert_allclose(level.to_array()[1:], mahalanobis - 1.0, atol=1e-9)


def test_scenario_renders():
    points, precision, ellipse = scenarios.gaussian()
    figure = render.draw_gaussian(points, partial(density, precision), ellipse)
    assert isinstance(figure, plt.Figure)
