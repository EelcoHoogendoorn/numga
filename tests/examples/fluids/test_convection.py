"""Rayleigh-Benard convection: gravity's pull on a parcel's mass, the onset of convection between rigid plates, and a
layer turning over from a still, conducting start."""

import numpy as np

from examples.fluids.convection import core, render, scenarios


def test_gravity_pulls_mass_alone():
    pull = -core.mv.yt                                                        # [] Drive+
    readings = [(core.mv.y & pull.commutator(part)).to_array() for part in (core.time, core.volume, core.entropy)]
    np.testing.assert_allclose(np.ravel(readings), [-1.0, 0.0, 0.0], atol=1e-12)


def test_onset_between_rigid_plates():
    rates = scenarios.onset().to_array()
    assert rates[0] < 0 < rates[-1]
    np.testing.assert_allclose(np.interp(0.0, rates, scenarios.onset_rayleighs), 1707.8, rtol=0.01)


def test_layer_turns_over_and_mixes_its_middle():
    topology, vertices, _, frames = scenarios.turning_over()
    frames = list(frames)
    _, places, _, _, _ = core.layer(scenarios.n_along, scenarios.n_across, scenarios.period)
    temperatures = frames[-1][1].to_array()
    drop = temperatures[np.isclose(places[:, 1], 0.25)].mean() - temperatures[np.isclose(places[:, 1], 0.75)].mean()
    assert abs(drop) < 0.05
    render.draw_layer(topology, vertices, scenarios.n_across, scenarios.period)
    assert next(render.animate(topology, vertices, scenarios.n_across, scenarios.period, frames[-1:])).shape[-1] == 3
    render.draw_onset(scenarios.onset_rayleighs, core.mv.scalar(np.linspace(-1, 1, 5)[:, None]))
