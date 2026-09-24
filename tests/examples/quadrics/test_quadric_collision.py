"""Tests for ellipse collision from blends of dual quadrics in PGA2D."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadric_collision import render, scenarios
from examples.quadrics.quadric_collision.core import Line, cubic_peak, ellipse, motor, mv, blend, tangent_line


def test_dual_ellipse_tangency():
    """Tangent lines from the support formula satisfy L ∨ Q(L) = 0 for every normal."""
    placement = motor(1.5, -0.8, np.radians(35))
    Q = placement >> ellipse(2.0, 1.2)(placement << Line)
    phi = np.radians([0, 45, 90, 135, 210, 315])
    lines = tangent_line(Q, mv.x * np.cos(phi) + mv.y * np.sin(phi))
    np.testing.assert_allclose((lines & Q(lines)).to_array(), 0.0, atol=1e-10)


def test_tangent_line_ignores_the_scale_of_the_quadric():
    """Q and any nonzero multiple of it, negative included, are the same conic with the same tangents."""
    placement = motor(1.5, -0.8, np.radians(35))
    Q = placement >> ellipse(2.0, 1.2)(placement << Line)
    phi = np.radians([0, 45, 90, 135, 210, 315])
    normals = mv.x * np.cos(phi) + mv.y * np.sin(phi)
    np.testing.assert_allclose(tangent_line(Q * -2.5, normals).kernel, tangent_line(Q, normals).kernel, atol=1e-12)


def test_cubic_peak_matches_a_dense_sweep():
    """The closed-form peak of the blend's determinant agrees with a sampled maximum."""
    placement = motor(2.5, 0.7, np.radians(-20))
    Q1 = ellipse(2.0, 1.0)
    Q2 = placement >> ellipse(1.0, 0.5)(placement << Line)
    samples = mv.scalar(np.array([[0.0], [1.0], [2.0], [-1.0]]))
    parameter, maximum = cubic_peak(blend(Q1, Q2, samples).dual().det())
    sweep = np.linspace(0.0, 1.0, 20001)
    values = blend(Q1, Q2, mv.scalar(sweep[:, None])).dual().det().to_array()
    np.testing.assert_allclose(parameter.to_array(), sweep[values.argmax()], atol=0.001)
    np.testing.assert_allclose(maximum.to_array(), values.max(), atol=1e-8)


def test_scenario_renders():
    """The scenario classifies its three poses (its checks) and the figure draws."""
    figure = render.draw_collision(*scenarios.collision())
    assert isinstance(figure, plt.Figure)
