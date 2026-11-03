"""A rubber band lifting a weight by heat: each vertex one point, its place and its entropy, its force and temperature
the two parts of one plane, the energy's change, its pull proportional to its temperature, warming as it is stretched and cooling as
it is let go, and the energy kept without drift."""

import numpy as np

from numga import stack
from examples.thermodynamics.rubber_band import core, render, scenarios


def test_pull_is_proportional_to_temperature():
    topology, rest, _, band = scenarios.hanging_band()
    # The band stretched to half again its length, its pull on the support at the room's temperature and at twice it,
    # the entropies raised by the heat capacity times the logarithm of two.
    stretched = rest + 0.5 * (core.mv.y & rest) * core.up                     # Point-[vertices]
    support = topology.reading(np.flatnonzero((core.mv.y & rest).to_array() == 0))
    held_potentials = [support * core.potentials(band, stretched + band.heat_capacity * band.cells * warmer
                                                 * core.entropy)[1] for warmer in (0.0, np.log(2))]   # Plane+[held]
    pulls = [((plane & core.entropy) * core.mv.s - plane).sites.sum() & core.up for plane in held_potentials]
    np.testing.assert_allclose((pulls[1] / pulls[0]).to_array(), 2.0, rtol=1e-12)
    # The band pulls the support down.
    assert pulls[0].to_array() < 0


def test_engine_lifts_the_weight_and_keeps_its_energy():
    topology, in_weight, states = scenarios.engine()
    states = list(states)
    kinetic, potential, thermal, heated, dissipated, handed = map(stack, zip(*(budget for *_, budget in states)))
    kept = (kinetic + potential + thermal + dissipated - heated - handed).to_array()
    assert np.abs(kept - kept[0]).max() < 1e-4 * np.abs(heated.to_array()).max()
    # The weight rises at every frame while the lamp is on, and is lower at the end than when the lamp went off.
    weight = topology.reading(np.unique(topology.elements[2][in_weight]))
    heights = np.array([(core.mv.y & (weight * places)).sites.mean().to_array() for places, *_ in states])
    n_lit_frames = scenarios.n_lit // scenarios.n_stride
    assert np.all(np.diff(heights)[:n_lit_frames] > 0)
    assert heights[-1] < heights[n_lit_frames]


def test_pulled_band_warms_and_let_go_cools():
    topology, _, states = scenarios.pull()
    states = list(states)
    kinetic, potential, thermal, heated, dissipated, handed = map(stack, zip(*(budget for *_, budget in states)))
    kept = (kinetic + potential + thermal + dissipated - heated - handed).to_array()
    assert np.abs(kept - kept[0]).max() < 1e-4
    # Pulled, the band warms above the room's temperature, most at its top, where it stretches most; let go, it cools
    # below it.
    changes = np.array([(temperatures - scenarios.room_temperature).sites.mean().to_array()
                        for _, temperatures, *_ in states])
    let_go = len(changes) // 2
    assert changes[:let_go].max() > 0 > changes[let_go:].min()
    pulled = states[np.argmax(changes)][1].to_array() - scenarios.room_temperature
    rows = (core.mv.y & states[0][0]).to_array()
    assert pulled[rows > np.median(rows)].mean() > pulled[rows < np.median(rows)].mean()


def test_scenes_draw():
    topology, rest, in_weight, _ = scenarios.hanging_band()
    render.draw_band(topology, in_weight, rest)
    warmth = core.mv.scalar(np.full((topology.n_vertices, 1), scenarios.room_temperature)).field()
    frames = render.animate(topology, in_weight, scenarios.lamp, scenarios.room_temperature, scenarios.lamp_temperature,
                            [(rest, warmth, 1.0, None)])
    assert next(frames).shape[-1] == 3
    pulled = render.animate_pulled(topology, in_weight, scenarios.room_temperature, scenarios.pull_swing,
                                   scenarios.pull_drop, [(rest, warmth, 0.0, None)])
    assert next(pulled).shape[-1] == 3
