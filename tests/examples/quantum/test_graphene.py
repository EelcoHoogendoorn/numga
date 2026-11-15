"""Graphene: the Hamiltonian squares to the field's length, and transport around a circle turns a
frame by the solid angle it encloses."""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.graphene import core, render, scenarios


def test_the_hamiltonian_squares_to_the_fields_length():
    """Applying `hamiltonian(field)` twice multiplies every spinor by the field's squared norm."""
    field = core.mv(core.Field, np.array([0.3, -1.1, 0.7]))                     # [] Field+
    state = core.mv(core.Spinor, np.array([0.4, -0.2, 1.3, 0.5]))               # [] Spinor+
    operator = core.hamiltonian(field)                                          # [] Spinor+ <- Spinor+
    residual = operator(operator(state)) - field.scalar_norm_squared() * state  # [] Spinor+
    assert (residual.norm() < 1e-9).all()


def test_transport_around_a_circle_of_latitude_turns_by_the_enclosed_solid_angle():
    """Transporting a frame around latitude circles at polar angles `polar_angles` gives a holonomy rotor whose scalar part equals the cosine of half the enclosed solid angle."""
    mv = core.mv
    n_angles, n_steps = 3, 400
    polar_angles = np.array([0.3, 1.0, 2.0])
    azimuths = np.linspace(0.0, 2 * np.pi, n_steps + 1)[:, None]
    # The pseudospin along z tilted toward x by polar_angles, then turned about z around the circle.
    tilt_rotors = (mv.zx * (-polar_angles / 2)).exp()                           # [n_angles] Rotor+
    tilted_pseudospins = tilt_rotors >> core.first_sublattice                   # [n_angles] Pseudospin-
    circle_rotors = (mv.xy * (-azimuths / 2)).exp()                             # [n_steps + 1, 1] Rotor+
    directions = circle_rotors >> tilted_pseudospins                            # [n_steps + 1, n_angles] Pseudospin-
    holonomy = core.transport(directions)[-1]                                   # [n_angles] Rotor+
    expected_cosine = np.cos(np.pi * (1 - np.cos(polar_angles)))
    assert ((holonomy.select[0] - expected_cosine).abs() < 1e-3).all()


def test_the_scenes_draw_at_a_few_samples():
    """All four graphene figures draw cleanly on minimal momentum grids and loop step counts."""
    momenta, field, energies = scenarios.zone_bands(scenarios.zone_extent, 3, scenarios.gapless)
    plt.close(render.field_figure(momenta, field))
    plt.close(render.bands_figure(momenta, energies))
    plt.close(render.textures_figure(partial(core.upper_pseudospins, gap=scenarios.texture_gap), scenarios.valley_extent))
    radii, gaps = scenarios.radii[:2], scenarios.gaps[:2]
    plt.close(render.berry_figure(radii, gaps, *scenarios.loop_berry(radii, gaps, 4)))
