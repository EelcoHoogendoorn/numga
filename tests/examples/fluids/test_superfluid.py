"""This module tests two-fluid drive forces, sound speeds, and cavity stepping and rendering for superfluid helium."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.fluids.superfluid import core, render, scenarios


def test_drives_push_parcels_by_what_they_hold():
    parcels, masses = scenarios.fluid_components()
    index = scenarios.cavity_index
    cavity_parcels = parcels[index]                                           # [2] State-
    drives = stack((-core.mv.xm, -core.mv.xs))                                # [2] Bivector+
    forces = drives[:, None].commutator(cavity_parcels[None, :])              # [2, 2] Point-
    # Chemical-potential drive pushes both components by their mass while temperature drive pushes only the normal fluid by its entropy.
    expected = -np.stack([masses[index], np.array([0.0, scenarios.densities[index] * scenarios.entropies[index]])])
    assert (((core.mv.x & forces) - expected).abs() < 1e-12).all()
    # Pairing with the total liquid parcel recovers the pressure force from the Gibbs-Duhem relation.
    liquid = cavity_parcels.sum(axis=-1)                                      # [] State-
    assert (((core.mv.x & drives.commutator(liquid)) + (stack((core.mv.m, core.mv.s)) & liquid)).abs() < 1e-12).all()


def test_sounds_match_measurements():
    bulk_sounds, powder_sounds = scenarios.sound_curves()
    assert ((bulk_sounds[:, 1] - scenarios.first_sounds ** 2).abs() < 1e-8 * scenarios.first_sounds ** 2).all()
    assert ((bulk_sounds[:, 0] - scenarios.second_sounds ** 2).abs() < 0.12 * scenarios.second_sounds ** 2).all()
    assert ((powder_sounds[:, 1] - scenarios.fourth_sounds ** 2).abs() < 0.02 * scenarios.fourth_sounds ** 2).all()
    fig = render.sounds_figure(
        scenarios.temperatures,
        bulk_sounds,
        powder_sounds,
        scenarios.first_sounds,
        scenarios.second_sounds,
        scenarios.fourth_sounds,
    )
    plt.close(fig)


def test_pulse_steps_one_frame_and_draws():
    topology, coordinates, powder_triangles, pulse_profile = scenarios.cavity_scene(5, 3)
    vertices = (core.origin + core.mv(core.Upright, coordinates).dual()).field()  # Point-[V]
    fig = render.cavity_figure(topology, vertices, powder_triangles)
    plt.close(fig)
    stiffness = scenarios.liquid_stiffnesses()[scenarios.cavity_index]
    parcels, masses = scenarios.fluid_components()
    thermometer = stiffness(core.entropy)                                     # [] Potential+
    barometer = stiffness(parcels[scenarios.cavity_index].sum(axis=-1))       # [] Potential+
    states = scenarios.heat_pulse(topology, vertices, powder_triangles, pulse_profile)
    next(states)
    frame_state = next(states)
    frame = next(
        render.cavity_animation(
            topology, vertices, powder_triangles, thermometer, barometer, scenarios.pulse_rise, 1.0, render.n_bands, [frame_state]
        )
    )
    assert frame.ndim == 3 and frame.shape[-1] == 3
