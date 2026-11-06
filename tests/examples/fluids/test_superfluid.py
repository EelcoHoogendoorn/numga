"""Superfluid helium's two fluids: the drive's pushes on their parcels, their sounds against Donnelly and Barenghi's
measurements, and the heat pulse on the smallest cavity, stepped one frame and drawn."""

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


def test_the_pulse_steps_one_frame_and_draws():
    topology, coordinates, in_powder, rise = scenarios.scene(n_columns=5, n_rows=3)
    vertices = (core.mass + core.axis * coordinates[:, 0] + core.across * coordinates[:, 1]).field()
    liquid = scenarios.stiffness(scenarios.at)
    parcels, _ = scenarios.components(scenarios.at)
    states = scenarios.pulse(topology, vertices, in_powder, rise)
    next(states)
    amounts = next(states)
    readings = [(liquid(amounts) & core.entropy, liquid(amounts) & parcels.sum(axis=-1))]
    next(render.animate(topology, vertices, in_powder, readings, scenarios.pulse_rise, 1.0))
