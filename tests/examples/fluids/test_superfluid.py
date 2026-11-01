"""Superfluid helium's two fluids: the drive's pushes on their parcels, their sounds against Donnelly and Barenghi's
measurements, and a heat pulse in a cavity divided by a wall of powder."""

import numpy as np

from numga import stack
from examples.fluids.superfluid import core, render, scenarios


def test_drives_push_parcels_by_what_they_hold():
    parcels, masses = scenarios.components(scenarios.at)
    drives = stack((-core.mv.xm, -core.mv.xs))                                # [potentials] Drive+
    forces = drives[:, None].commutator(parcels[None, :])                     # [potentials, components] Point-
    # The chemical potential's drive pushes each component by its mass, the temperature's the normal fluid alone by its
    # entropy.
    expected = -np.stack([masses, (core.mv.s & parcels).to_array()])
    np.testing.assert_allclose((core.mv.x & forces).to_array(), expected, rtol=1e-12)
    # The whole liquid is pushed by the fall of its pressure, the potentials read by its parcel.
    liquid = parcels.sum(axis=-1)
    np.testing.assert_allclose((core.mv.x & drives.commutator(liquid)).to_array(),
                               -(stack((core.mv.m, core.mv.s)) & liquid).to_array(), rtol=1e-12)


def test_sounds_match_measurements():
    free, held = scenarios.sounds()
    speeds, held_speeds = np.sqrt(np.abs(free.to_array())), np.sqrt(np.abs(held.to_array()))
    np.testing.assert_allclose(speeds[:, 1], scenarios.first_sounds, rtol=1e-10)
    np.testing.assert_allclose(speeds[:, 0], scenarios.second_sounds, rtol=0.06)
    np.testing.assert_allclose(held_speeds[:, 1], scenarios.fourth_sounds, rtol=0.01)


def test_heat_pulse_rings_as_second_sound_behind_the_powder():
    topology, coordinates, in_powder, rise = scenarios.scene()
    vertices = (core.mass + core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()
    _, _, cells = core.triangles(topology, vertices)
    liquid = scenarios.stiffness(scenarios.at)
    states = list(scenarios.pulse(topology, vertices, in_powder, rise))
    totals = stack([(cells * amounts).sites.sum() for amounts in (states[0], states[-1])])
    readings = (stack((core.mv.m, core.mv.s)) & totals[:, None]).to_array()
    np.testing.assert_allclose(readings[1], readings[0], rtol=1e-10, atol=1e-10)
    # The far chamber, from the powder's far face on, never holds any entropy.
    far = core.mv.scalar((coordinates[:, 0] >= scenarios.wall_middle + scenarios.wall_half_thickness)[:, None] * 1.0).field()
    far_entropies = stack([(far * cells * (core.mv.s & amounts)).sites.sum() for amounts in states]).to_array()
    assert np.abs(far_entropies).max() < 1e-12 * readings[0, 1]
    # A fifth of the way in, the ring's warmest place along the heater's row towards the powder is where second sound
    # has carried it, within a tenth.
    frame = len(states) // 5
    row = (coordinates[:, 1] == scenarios.heater[1]) & (coordinates[:, 0] > scenarios.heater[0]) & (
        coordinates[:, 0] < scenarios.wall_middle - scenarios.wall_half_thickness)
    warmth = (liquid(states[frame]) & core.entropy).to_array()
    reach = coordinates[row, 0][warmth[row].argmax()] - scenarios.heater[0]
    elapsed = frame * scenarios.stride * scenarios.courant * (scenarios.width / (scenarios.n_columns - 1)) / (
        scenarios.first_sounds[scenarios.at])
    second_sound = np.sqrt(scenarios.sounds()[0].to_array()[scenarios.at, 0])
    np.testing.assert_allclose(reach, second_sound * elapsed, rtol=0.1)


def test_scenes_draw():
    free, held = scenarios.sounds()
    render.draw_sounds(scenarios.temperatures, free, held, scenarios.first_sounds, scenarios.second_sounds,
                       scenarios.fourth_sounds)
    topology, coordinates, in_powder, rise = scenarios.scene()
    vertices = (core.mass + core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()
    render.draw_cavity(topology, vertices, in_powder)
    flat = core.mv.scalar(rise[:, None]).field()
    frames = list(render.animate(topology, vertices, in_powder, [(flat, 0 * flat)], scenarios.pulse_rise, 1.0))
    times = np.linspace(0, 1, 5)
    render.draw_fountain(times, core.mv.scalar(times[:, None]), core.mv.scalar(times[:, None]))
    assert frames[0].shape[-1] == 3
