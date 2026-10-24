"""Tests for the spherical conic in Cl(3)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor

from examples.quadrics.spherical_quadrics import render, scenarios
from examples.quadrics.spherical_quadrics.core import mv, points, polarity


def test_polarity_is_diagonal_on_the_basis():
    """The polarity maps each basis point to its polar basis plane, scaled by its eigenvalue."""
    C = polarity(np.array([1.0, 0.4, -0.8]))
    planes = Extensor.stack([mv.x, mv.y, mv.z])
    gram = C(points(np.eye(3))[:, None]) | planes
    np.testing.assert_allclose(gram.to_array(), np.diag([1.0, 0.4, -0.8]), atol=1e-14)


def test_the_oval_has_a_constant_focal_sum_and_its_tangents_lie_on_the_dual():
    curve, foci, _, semi_major, tangents, _, _, dual = scenarios.spherical_conic()
    focal_sum = sum((-(curve | focus)).clip(-1.0, 1.0).arccos() for focus in (foci[0], foci[1]))
    np.testing.assert_allclose(focal_sum.to_array(), 2.0 * semi_major, atol=1e-11)
    np.testing.assert_allclose((tangents & dual(tangents)).to_array(), 0.0, atol=1e-12)


def test_scenario_renders():
    curve, foci, sample, _, tangents, potential, conic, _ = scenarios.spherical_conic()
    figure = render.draw_spherical_conic(curve, foci, sample, tangents, potential, conic)
    assert isinstance(figure, plt.Figure)
