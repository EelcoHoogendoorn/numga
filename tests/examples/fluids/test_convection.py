"""Rayleigh-Benard convection: gravity's pull on a parcel's mass alone, and the layer on the smallest mesh, stepped one
frame and drawn."""

import numpy as np

from examples.fluids.convection import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.ym                                                        # [] Drive+
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in (core.mass, core.volume, core.entropy)]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0], atol=1e-12)


def test_the_layer_steps_one_frame_and_draws():
    opened, opening, vertices, liquid, frames = scenarios.turning_over(n_along=3, n_across=3)
    next(frames)
    next(render.animate(opened, vertices, opening, scenarios.period, [next(frames)]))
