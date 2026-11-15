"""Rational quadratic Bézier curves: one frame of the sweep, at a few samples, and its picture drawn."""

from examples.geometry.bezier import render, scenarios


def test_one_frame_of_the_sweep_draws():
    weighted_controls, curves, complements, shapes = scenarios.conic_sweep(n_frames=1, n_samples=4)
    assert ((shapes[:, None](curves) & curves).abs() < 1e-10).all()
    next(render.animate_rays(weighted_controls, curves, complements, shapes))
