"""Coverage tests verify homogeneous Gaussian fitting, Mahalanobis quadric evaluation, and rendering."""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.gaussian import render, scenarios
from examples.quadrics.gaussian.core import gaussian_density, gaussian_fit, mv


def test_gaussian_fit_matches_mahalanobis_distance_and_mirrors():
    """The fitted density peaks at the sample mean, matches sample Mahalanobis distance, and commutes with reflection."""
    n_samples, n_queries = 8, 3
    rng = np.random.default_rng(0)
    coordinates = rng.normal(size=(n_samples, 2)) * [2.0, 0.5] + [1.0, -1.0]
    query_coordinates = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, -3.0]])
    samples = mv.point(coordinates)                                            # [n_samples] Point-
    queries = mv.point(query_coordinates)                                      # [n_queries] Point-
    precision, ellipsoid = gaussian_fit(samples)                               # [] Plane+ <- Point-, Plane+ <- Point-
    mean = samples.mean(axis=0)                                                # [] Point-

    assert (gaussian_density(precision, mean) - 1).abs() < 1e-8
    offsets = query_coordinates - coordinates.mean(axis=0)
    covariance = np.cov(coordinates.T, bias=True)
    mahalanobis = np.einsum("ni,ij,nj->n", offsets, np.linalg.inv(covariance), offsets)
    query_levels = ellipsoid(queries) & queries                                # [n_queries] Scalar+
    assert ((query_levels - (mahalanobis - 1)).abs() < 1e-8).all()

    mirrored_samples = mv.x >> samples                                         # [n_samples] Point-
    mirrored_queries = mv.x >> queries                                         # [n_queries] Point-
    mirrored_precision, mirrored_ellipsoid = gaussian_fit(mirrored_samples)    # [] Plane+ <- Point-, Plane+ <- Point-
    assert ((gaussian_density(mirrored_precision, mirrored_queries) - gaussian_density(precision, queries)).abs() < 1e-8).all()
    assert (((mirrored_ellipsoid(mirrored_queries) & mirrored_queries) - query_levels).abs() < 1e-8).all()


def test_gaussian_figure():
    """The scenario cloud fits and renders into a figure without error."""
    points = scenarios.point_cloud()                                           # [n_points] Point-
    precision, ellipsoid = gaussian_fit(points)                                # [] Plane+ <- Point-, Plane+ <- Point-
    fig = render.gaussian_figure(points, partial(gaussian_density, precision), ellipsoid)
    plt.close(fig)
