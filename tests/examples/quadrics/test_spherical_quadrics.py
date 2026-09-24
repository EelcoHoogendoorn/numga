"""Tests for the spherical conic in Cl(3)."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor

from examples.quadrics.spherical_quadrics import render, scenarios
from examples.quadrics.spherical_quadrics.core import geodesic, great_circles, mv, points, polarity


def test_polarity_is_diagonal_on_the_basis():
    """The polarity maps each basis point to its polar basis plane, scaled by its eigenvalue."""
    C = polarity(np.array([1.0, 0.4, -0.8]))
    planes = Extensor.stack([mv.x, mv.y, mv.z])
    gram = C(points(np.eye(3))[:, None]) | planes
    np.testing.assert_allclose(gram.to_array(), np.diag([1.0, 0.4, -0.8]), atol=1e-14)


def test_geodesics_and_great_circles_stay_on_the_sphere():
    a, b = points(np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]]))
    arc = geodesic(a, b, np.linspace(0.0, 1.0, 7))
    np.testing.assert_allclose((-(arc | arc)).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((-(arc[-1] | b)).to_array(), 1.0, atol=1e-12)
    plane = mv.z
    circle = great_circles(plane, points(np.array([1.0, 0.0, 0.0])), np.linspace(0.0, 2 * np.pi, 9))
    np.testing.assert_allclose((circle & plane).to_array(), 0.0, atol=1e-12)


def test_scenario_renders():
    figure = render.draw_spherical_conic(*scenarios.spherical_conic())
    assert isinstance(figure, plt.Figure)
