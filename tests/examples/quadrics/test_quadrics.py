"""Coverage tests for projective ellipsoids as plane-to-point maps."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadrics import render, scenarios
from examples.quadrics.quadrics.core import mv, support_plane


def test_support_planes_are_tangent_and_face_their_normal():
    """Every support plane contains its contact point, faces the requested direction, and solves back from its point."""
    dual = scenarios.rest_ellipsoid()                                              # [] Point- <- Plane+
    normals = (
        mv.x * np.array([1.0, 0.0, -1.0, 0.3])
        + mv.y * np.array([0.0, 1.0, 2.0, -0.4])
        + mv.z * np.array([0.0, 0.0, 0.5, 2.0])
    )                                                                              # [4] Plane+
    unit_normals = normals.normalized()                                            # [4] Plane+
    tangents = support_plane(dual, normals)                                        # [4] Plane+
    contacts = dual(tangents)                                                      # [4] Point-
    recovered = dual.solve(contacts)                                               # [4] Plane+
    assert ((tangents & contacts).abs() < 1e-12).all()
    assert (((tangents | unit_normals) - 1.0).abs() < 1e-12).all()
    np.testing.assert_allclose((recovered - tangents).kernel, 0, atol=1e-12)


def test_the_scenes_draw():
    dual, tangent, contact, _ = scenarios.polar_reciprocity()
    fig_reciprocity = render.draw_polar_reciprocity(dual, tangent, contact)
    fig_transport = render.draw_motor_transport(*scenarios.motor_transport())
    plt.close(fig_reciprocity)
    plt.close(fig_transport)
