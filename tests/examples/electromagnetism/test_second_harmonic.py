"""Tests for second-harmonic generation cover the tetrahedral bond response, crystal rotation and
bond inversion, the [110]-cut polarization and power, pump-probe linear mixing, closed-form slice
growth, and figure rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core, render, scenarios

mv = core.mv
beam_fields = stack((core.horizontal_field, core.vertical_field))              # [2] Field+


def test_response_has_tetrahedral_component_law_and_ignores_magnetic_fields():
    crystal = core.crystal
    rng = np.random.default_rng(814)
    first, second = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    first_field = core.field_along((core.axes * first).sum(axis=-1))           # [7] Field+
    second_field = core.field_along((core.axes * second).sum(axis=-1))         # [7] Field+
    expected = (core.axes * np.stack((
        first[:, 1] * second[:, 2] + first[:, 2] * second[:, 1],
        first[:, 2] * second[:, 0] + first[:, 0] * second[:, 2],
        first[:, 0] * second[:, 1] + first[:, 1] * second[:, 0],
    ), axis=-1)).sum(axis=-1) & core.time                                      # [7] Polarization-
    magnetic_fields = mv(core.Field.derive.within("x y z"), rng.normal(size=(2, 7, 3)))
    assert ((crystal(first_field, second_field) - expected).norm_squared() < 1e-12).all()
    assert ((crystal(first_field)(second_field) - expected).norm_squared() < 1e-12).all()
    assert ((crystal(first_field + magnetic_fields[0], second_field + magnetic_fields[1]) - expected).norm_squared() < 1e-12).all()


def test_response_turns_with_crystal_and_reverses_with_bonds():
    crystal = core.crystal
    rng = np.random.default_rng(177)
    fields, probes = mv.bivector(rng.normal(size=(7, 6))), mv.bivector(rng.normal(size=(7, 6)))
    rotation = (mv.xy * 0.37).exp() * (mv.yz * 0.23).exp()
    turned_response = rotation >> crystal(rotation << core.Field, rotation << core.Field)
    rebuilt_response = core.bond_response(rotation >> core.bonds)
    assert ((turned_response(rotation >> fields, rotation >> probes) - (rotation >> crystal(fields, probes))).norm_squared() < 1e-8).all()
    assert ((turned_response(fields, probes) - rebuilt_response(fields, probes)).norm_squared() < 1e-12).all()
    assert ((core.bond_response(-core.bonds)(fields, probes) + crystal(fields, probes)).norm_squared() < 1e-12).all()


def test_110_cut_polarization_and_power():
    n_headings = 17
    headings = np.linspace(0, 2 * np.pi, n_headings)
    across, up = np.cos(headings), np.sin(headings)
    pumps, _ = scenarios.turning(1, n_headings)
    harmonic = core.second_harmonic(pumps, pumps)
    expected = -core.horizontal_polarization * (across * up) - core.vertical_polarization * (across**2 / 2)
    assert ((harmonic - expected).norm_squared() < 1e-10).all()
    power = (beam_fields & harmonic[..., None]).squared().sum(axis=-1)
    assert ((power - across**2 * (4 - 3 * across**2) / 4).abs() < 1e-10).all()


def test_pump_bound_response_matches_cross_term():
    pumps, probes, generated = scenarios.mixing(13)
    pump_and_probe = pumps[:, None] + probes
    cross_term = (core.second_harmonic(pump_and_probe, pump_and_probe)
                  - core.second_harmonic(pumps[:, None], pumps[:, None])
                  - core.second_harmonic(probes, probes))
    assert ((cross_term - generated).norm_squared() < 1e-12).all()


def test_uniform_growth_matches_closed_form_and_quasi_phase_matching_builds():
    n_slices = 128
    depths = (np.arange(n_slices) + 0.5) / n_slices
    mismatch = np.array([[0.0], [2 * np.pi], [4 * np.pi]])
    source = (core.horizontal_polarization + core.vertical_polarization) / np.sqrt(2)
    amplitude = core.growth(source, np.ones((3, n_slices)), mismatch, depths)
    edges = depths + 0.5 / n_slices
    exact = np.where(mismatch == 0, edges, (np.exp(1j * mismatch * edges) - 1) / (1j * np.where(mismatch == 0, 1, mismatch)))
    expected = source * np.stack((exact.real, exact.imag), axis=-1)
    assert ((amplitude - expected).norm_squared() < 1e-3).all()

    _, matched_amplitude = scenarios.phase_matching(256)
    power = matched_amplitude.scalar_norm_squared().sum(axis=-1)
    assert ((power[:, -1] - np.array([1.0, 0.0, (2 / np.pi) ** 2])).abs() < 1e-4).all()


def test_render_views_and_animation_frame():
    phase, pump, harmonic = scenarios.waveform(9)
    plt.close(render.draw_waveform(phase, pump, harmonic))
    pumps, states = scenarios.turning(1, 5)
    bonds, turned_harmonic = next(states)
    plt.close(render.draw_response(pumps, bonds, turned_harmonic))
    frame = next(render.animate(pumps, [(bonds, turned_harmonic)]))
    assert frame.ndim == 3 and frame.shape[-1] == 3
    plt.close(render.draw_mixing(*scenarios.mixing(5)))
    plt.close(render.draw_growth(*scenarios.phase_matching(8)))
