"""Rational quadratic Bézier curves: one frame of the sweep, at a few samples, and its picture drawn."""

from examples.geometry.bezier import render, scenarios


def test_one_frame_of_the_sweep_draws():
    weighted, curves, complements, shapes = scenarios.sweep(n_frames=1, n_samples=4)
    next(render.animate_rays(weighted, curves, complements, shapes))
