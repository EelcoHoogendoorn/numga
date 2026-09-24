"""Tests for ellipsoids as plane-to-point maps in PGA3D."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadrics import render, scenarios
from examples.quadrics.quadrics.core import mv, support_plane


def test_support_planes_are_tangent_and_face_their_normal():
    """Every support plane contains its own contact point and keeps the requested orientation."""
    dual = scenarios.body()
    normals = mv.x * np.array([1.0, 0.0, -1.0, 0.3]) + mv.y * np.array([0.0, 1.0, 2.0, -0.4]) + mv.z * np.array([0.0, 0.0, 0.5, 2.0])
    tangents = support_plane(dual, normals, mv.w)
    np.testing.assert_allclose((tangents & dual(tangents)).to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose((tangents | normals.normalized()).to_array(), (normals.normalized() | normals.normalized()).to_array(), atol=1e-12)


def test_scenarios_render():
    for figure in (
        render.draw_polar_reciprocity(*scenarios.polar_reciprocity()),
        render.draw_motor_transport(*scenarios.motor_transport()),
    ):
        assert isinstance(figure, plt.Figure)
