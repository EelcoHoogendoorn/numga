"""Rayleigh-Benard convection: gravity's pull on a parcel's mass, and a layer turning over from a still, conducting
start."""

import numpy as np

from examples.fluids.convection import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.yt                                                        # [] Drive+
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in (core.time, core.volume, core.entropy)]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0], atol=1e-12)


def test_layer_turns_over_and_mixes_its_middle():
    opened, opening, vertices, liquid, frames = scenarios.turning_over()
    frames = list(frames)
    heights = (core.mv.y & vertices).to_array()
    temperatures = (opening * frames[-1][1]).to_array()
    drop = temperatures[np.isclose(heights, 0.25)].mean() - temperatures[np.isclose(heights, 0.75)].mean()
    assert abs(drop) < 0.05
    render.draw_layer(opened, vertices, scenarios.period)
    assert len(list(render.animate(opened, vertices, opening, scenarios.period, frames[-1:]))) == 1
