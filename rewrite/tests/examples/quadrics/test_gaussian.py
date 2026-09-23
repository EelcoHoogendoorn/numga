"""Tests for the Gaussian fit and its 1σ quadric."""

from __future__ import annotations

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.gaussian import render, scenarios
from examples.quadrics.gaussian.core import fit_gaussian, point


def test_density_peaks_at_the_mean_and_level_matches_the_covariance():
    """The density is one at the sample mean, and the level is the squared Mahalanobis distance less one."""
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(500, 2)) * [2.0, 0.5] + [1.0, -1.0]
    queries = np.array([[0.0, 0.0], [2.0, 1.0], [-1.0, -3.0]])
    density, level = fit_gaussian(point(xy), point(np.concatenate([xy.mean(axis=0)[None], queries])))
    np.testing.assert_allclose(density.to_array()[0], 1.0, atol=1e-12)
    offset = queries - xy.mean(axis=0)
    covariance = np.cov(xy.T, bias=True)
    mahalanobis = np.einsum("ni,ij,nj->n", offset, np.linalg.inv(covariance), offset)
    np.testing.assert_allclose(level.to_array()[1:], mahalanobis - 1.0, atol=1e-9)


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.quadrics.gaussian.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_scenario_renders():
    figure = render.draw_gaussian(*scenarios.gaussian())
    assert isinstance(figure, plt.Figure)
