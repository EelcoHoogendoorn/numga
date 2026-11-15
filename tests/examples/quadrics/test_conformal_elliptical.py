"""Coverage tests verify the thirteen cube-axis planes, their rotational closure, and front-hemisphere rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical import render
from examples.quadrics.conformal_elliptical.core import alignments, planes


def test_thirteen_distinct_unit_planes_closed_under_cube_rotations() -> None:
    """All thirteen planes have unit norm, remain pairwise distinct up to orientation, and permute under both generators."""
    assert planes.shape == (13,)
    assert ((planes.scalar_norm_squared() - 1).abs() < 1e-6).all()
    assert ((planes[:, None] | planes).abs() > 1 - 1e-6).sum() == 13
    assert alignments.shape == (2, 13, 13)
    assert ((alignments.max(axis=-1) - 1).abs() < 1e-6).all()
    permutations = alignments.argmax(axis=-1)
    assert np.array_equal(np.sort(permutations, axis=-1), np.broadcast_to(np.arange(13), (2, 13)))


def test_planes_figure_renders() -> None:
    """Front-hemisphere great-circle rendering returns a matplotlib figure."""
    figure = render.planes_figure(planes)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
