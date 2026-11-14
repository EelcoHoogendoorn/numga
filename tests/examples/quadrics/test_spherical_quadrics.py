"""Coverage tests for spherical conics in 3D spherical geometric algebra."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.quadrics.spherical_quadrics import render, scenarios
from examples.quadrics.spherical_quadrics.core import mv, principal_polarity


def test_principal_polarity_is_diagonal_on_coordinate_poles():
    """The principal polarity maps each coordinate pole to its coordinate plane scaled by its eigenvalue."""
    eigenvalues = np.array([1.0, 0.4, -0.8])
    conic = principal_polarity(eigenvalues)
    planes = stack([mv.x, mv.y, mv.z])
    poles = planes.dual()
    diagonal = conic(poles) & poles
    off_diagonal = conic(poles[:, None]) & poles[None, :]
    assert ((diagonal - eigenvalues).abs() < 1e-12).all()
    assert ((off_diagonal.sum(axis=-1) - diagonal).abs() < 1e-12).all()


def test_oval_focal_sum_and_dual_tangent_locus():
    """Points on the spherical oval satisfy the focal geodesic sum and tangent planes lie on the dual conic."""
    oval_points, foci, _, semi_major, tangents, conic, dual = scenarios.spherical_conic_scene()
    focal_sum = (oval_points[:, None].reverse() | foci).clip(-1.0, 1.0).arccos().sum(axis=-1)
    assert ((conic(oval_points) & oval_points).abs() < 1e-12).all()
    assert ((focal_sum - 2 * semi_major).abs() < 1e-11).all()
    assert ((tangents & dual(tangents)).abs() < 1e-12).all()


def test_conic_figure_renders():
    """The three-panel GPU spherical conic figure renders cleanly without writing files."""
    oval_points, foci, sample_point, _, tangents, conic, _ = scenarios.spherical_conic_scene()
    figure = render.conic_figure(oval_points, foci, sample_point, tangents, conic)
    plt.close(figure)
