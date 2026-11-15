"""Tests for Rayleigh-Bénard convection check that gravity pulls on a parcel's mass alone and that the smallest
triangulated layer steps one frame and renders."""

import numpy as np

from examples.fluids.convection import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.ym                                                        # [] Bivector+
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in (core.mass, core.volume, core.entropy)]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0], atol=1e-12)


def test_the_layer_steps_one_frame_and_draws():
    opened, opening, vertices, liquid, frames = scenarios.turning_over(3, 3)
    next(frames)
    next(render.animate(opened, vertices, opening, scenarios.period, [next(frames)]))
