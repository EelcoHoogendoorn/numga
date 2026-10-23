"""Rational quadratic Bézier curves: both arcs on the cone of their conic as the middle weight sweeps, and the
animation draws."""

import numpy as np

from examples.geometry.bezier import render, scenarios


def test_both_arcs_lie_on_the_cone_of_their_conic_and_the_animation_draws():
    weighted, curves, complements, shapes = scenarios.sweep()
    assert np.abs((shapes[:, None](curves) & curves).to_array()).max() < 1e-10
    assert np.abs((shapes[:, None](complements) & complements).to_array()).max() < 1e-10
    assert len(render.animate_rays(weighted[:2], curves[:2], complements[:2], shapes[:2])) == 2
