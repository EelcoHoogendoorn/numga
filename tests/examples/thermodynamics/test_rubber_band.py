"""A rubber band lifting a weight by heat: each vertex one point, its place and its entropy, its force and temperature
the two parts of one plane, the energy's change, its pull proportional to its temperature, warming as it is stretched and cooling as
it is let go, and, isolated, its energy kept and its entropy grown."""

import numpy as np

from numga import stack
from examples.thermodynamics.rubber_band import core, render, scenarios


def test_pull_is_proportional_to_temperature():
    topology, rest, _, band = scenarios.hanging_band()
    # The band stretched to half again its length, its pull on the support at the room's temperature and at twice it,
    # the entropies raised by the heat capacity times the logarithm of two.
    stretched = rest + 0.5 * (core.mv.y & rest) * core.up                     # Point-[vertices]
    support = topology.reading(np.flatnonzero((core.mv.y & rest).to_array() == 0), 0)
    held_potentials = [support * core.potentials(band, stretched + band.heat_capacity * band.cells * warmer
                                                 * core.entropy) for warmer in (0.0, np.log(2))]   # Plane+[held]
    pulls = [((plane & core.entropy) * core.mv.s - plane).sites.sum() & core.up for plane in held_potentials]
    np.testing.assert_allclose((pulls[1] / pulls[0]).to_array(), 2.0, rtol=1e-12)
    # The band pulls the support down.
    assert pulls[0].to_array() < 0


def test_isolated_band_keeps_its_energy_and_grows_its_entropy():
    # The band let go at rest, unstretched, with nothing outside it doing work or moving heat: no lamp, no room to cool
    # it, no drag and no hand. It bounces under the weight, warming where it stretches, and conducts that warmth.
    topology, rest, _, band = scenarios.hanging_band()
    isolated = band._replace(cooling=0.0, drag=0.0)
    n_steps, dt, lamp, reach = 400, scenarios.dt, scenarios.lamp, scenarios.reach
    points, velocities = rest, 0 * rest                                       # Point-[vertices]
    kinetic, energies, entropies = [], [], []
    for _ in range(n_steps):
        points, velocities = core.step(isolated, points, velocities, lamp, 0.0, reach, 0.0, 0.0, dt)
        # The energy of the state: its motion's, gravity's, and its heat's, each cell's heat capacity times its
        # temperature; and the entropy it holds.
        temperatures = core.potentials(isolated, points) & core.entropy       # Scalar+[vertices]
        motion = (isolated.masses * (velocities.dual_inverse() & velocities)).sites.sum() / 2   # [] Scalar+
        height = -(isolated.masses * (isolated.pull & points)).sites.sum()    # [] Scalar+
        heat = isolated.heat_capacity * (isolated.cells * temperatures).sites.sum()   # [] Scalar+
        kinetic.append(motion)
        energies.append(motion + height + heat)
        entropies.append((core.mv.s & points).sites.sum())
    kinetic, energies, entropies = (stack(values).to_array() for values in (kinetic, energies, entropies))
    # The energy is kept without drift, within the square of the time step of the largest kinetic energy.
    assert np.abs(energies - energies[0]).max() < dt ** 2 * kinetic.max()
    # The entropy never falls, beyond round-off, and grows.
    round_off = 1e-12
    assert np.diff(entropies).min() > -round_off
    assert entropies[-1] - entropies[0] > 1e3 * round_off


def test_engine_lifts_the_weight_and_lets_it_down():
    topology, in_weight, states = scenarios.engine()
    states = list(states)
    # The weight rises at every frame while the lamp is on, and is lower at the end than when the lamp went off.
    weight = topology.reading(np.unique(topology.elements[2][in_weight]), 0)
    heights = np.array([(core.mv.y & (weight * places)).sites.mean().to_array() for places, *_ in states])
    lit = np.array([lit for *_, lit in states]) > 0
    assert np.all(np.diff(heights[lit]) > 0)
    assert heights[-1] < heights[lit][-1]


def test_pulled_band_warms_and_let_go_cools():
    topology, _, states = scenarios.pull()
    states = list(states)
    # Pulled, the band warms above the room's temperature, most at its top, where it stretches most; let go, it cools
    # below it.
    changes = np.array([(temperatures - scenarios.room_temperature).sites.mean().to_array()
                        for _, temperatures, _ in states])
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
                            [(rest, warmth, 1.0)])
    assert next(frames).shape[-1] == 3
    pulled = render.animate_pulled(topology, in_weight, scenarios.room_temperature, scenarios.pull_swing,
                                   scenarios.pull_drop, [(rest, warmth, 0.0)])
    assert next(pulled).shape[-1] == 3
