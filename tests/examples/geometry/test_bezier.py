"""Rational Bézier curves: both arcs of a quadratic curve on one conic of the right kind, tangent planes
on the envelope, and curves of motors between their end placements."""

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.bezier import core, render, scenarios


def test_quadratic_curves_and_their_complements_lie_on_one_conic_of_the_right_kind():
    _, curves, complements, shapes, kinds = scenarios.conics()
    assert np.abs(render.scalars(curves & shapes[:, None](curves))).max() < 1e-10
    assert np.abs(render.scalars(complements & shapes[:, None](complements))).max() < 1e-10
    eigenvalues = render.scalars(kinds)                                        # [cases, 2]
    assert (eigenvalues[0] > 0).all()
    assert np.abs(eigenvalues[1]).min() < 1e-10 and eigenvalues[1].max() > 0
    assert eigenvalues[2].prod() < 0


def test_tangent_planes_touch_the_curve_and_lie_on_the_envelope():
    _, curves, _, shapes, _ = scenarios.conics()
    planes = scenarios.tangent_planes()                                         # [cases, tangents] Plane
    assert np.abs(render.scalars(planes & curves[:, ::scenarios.TANGENT_STRIDE])).max() < 1e-10
    assert np.abs(render.scalars(planes & shapes.inverse()[:, None](planes))).max() < 1e-10


def test_complementary_curves_pass_through_infinity_where_their_weights_cancel():
    _, curves, complements = scenarios.cubic()
    assert (render.homogeneous(curves[0])[..., 2] > 0).all()
    # The complements of the first and last case cross the plane at infinity; the middle one only nears it.
    weights = render.homogeneous(complements)[..., 2]                          # [cases, samples + 1]
    np.testing.assert_array_equal((np.diff(np.sign(weights), axis=-1) != 0).any(axis=-1), [True, False, True])


def test_curves_of_motors_end_at_the_placements_and_the_figures_draw():
    centres, placements, paths, frames = scenarios.motors()
    assert render.scalars((paths[:, [0, -1]] & centres[[0, -1]]).norm()).max() < 1e-10
    plt.close(render.draw_motors(centres, placements, paths, frames))
    plt.close(render.draw_curves(*scenarios.cubic(), scenarios.CUBIC_BOX))
    triangle, curves, complements, shapes, _ = scenarios.conics()
    figure = render.draw_conics(triangle, curves, complements, shapes, scenarios.CONIC_BOX)
    plt.close(render.draw_tangents(scenarios.tangent_planes(), figure))
    weighted, curves, complements, shapes = scenarios.sweep()
    assert len(render.animate_rays(weighted[:2], curves[:2], complements[:2], shapes[:2])) == 2
