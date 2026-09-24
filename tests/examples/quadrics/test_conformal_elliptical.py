"""Tests for the octahedral mirror planes on the unit sphere."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.conformal_elliptical import render, scenarios
from examples.quadrics.conformal_elliptical.core import octahedral_planes


def test_thirteen_distinct_unit_planes():
    """The planes are unit normals, no two of them equal up to sign."""
    planes = octahedral_planes()
    assert planes.shape == (13,)
    np.testing.assert_allclose((planes | planes).to_array(), 1.0, atol=1e-12)
    alignment = (planes[:, None] | planes).abs().to_array()
    assert (alignment[~np.eye(13, dtype=bool)] < 1.0 - 1e-6).all()


def test_scenario_renders():
    figure = render.draw_octahedral_sphere(scenarios.octahedral_sphere())
    assert isinstance(figure, plt.Figure)
