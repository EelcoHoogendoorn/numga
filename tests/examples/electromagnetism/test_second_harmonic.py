"""The frequency-doubling crystal: its response has the tetrahedral component law, turns with the
crystal and reverses with its bonds, the [110] cut has its analytic polarization and power, the
phase-matched growth follows the closed form, and the scenes draw."""

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.second_harmonic import core, render, scenarios

mv = core.mv


def test_response_has_the_tetrahedral_component_law_bound_at_once_or_one_slot_at_a_time():
    crystal = core.response(core.BONDS)
    rng = np.random.default_rng(814)
    first, second = rng.normal(size=(19, 3)), rng.normal(size=(19, 3))
    # The six permutations of three distinct axes share one unit coefficient.
    electric_planes = core.ga.subspace("tx ty tz")
    first_field, second_field = mv(electric_planes, first), mv(electric_planes, second)
    expected = mv(electric_planes, np.stack((
        first[:, 1] * second[:, 2] + first[:, 2] * second[:, 1],
        first[:, 2] * second[:, 0] + first[:, 0] * second[:, 2],
        first[:, 0] * second[:, 1] + first[:, 1] * second[:, 0],
    ), axis=-1))
    magnetic_fields = mv(core.ga.subspace("xy xz yz"), rng.normal(size=(2, 19, 3)))
    np.testing.assert_allclose((crystal(first_field, second_field) - expected).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((crystal(first_field)(second_field) - expected).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((crystal(first_field + magnetic_fields[0], second_field + magnetic_fields[1]) - expected).kernel, 0, atol=1e-12)


def test_the_response_turns_with_the_crystal_and_reverses_with_its_bonds():
    crystal = core.response(core.BONDS)
    rng = np.random.default_rng(177)
    fields, probes = mv.bivector(rng.normal(size=(23, 6))), mv.bivector(rng.normal(size=(23, 6)))
    rotation = (mv.xy * 0.37).exp() * (mv.yz * 0.23).exp()
    turned = core.turned(crystal, rotation)
    rebuilt = core.response(rotation >> core.BONDS)
    np.testing.assert_allclose(
        (turned(rotation >> fields, rotation >> probes) - (rotation >> crystal(fields, probes))).kernel,
        0, atol=1e-8)
    np.testing.assert_allclose((turned(fields, probes) - rebuilt(fields, probes)).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((core.response(-core.BONDS)(fields, probes) + crystal(fields, probes)).kernel, 0, atol=2e-12)


def test_110_cut_has_the_analytic_polarization_and_power():
    crystal = core.response(core.BONDS)
    headings = np.linspace(0, 2 * np.pi, 129)
    across, up = np.cos(headings), np.sin(headings)
    pumps, _ = scenarios.turning(1, len(headings))
    harmonic = core.doubled(crystal, pumps)
    expected = -core.HORIZONTAL * (across * up) - core.VERTICAL * (across**2 / 2)
    np.testing.assert_allclose((harmonic - expected).kernel, 0, atol=1e-6)
    np.testing.assert_allclose((harmonic | harmonic).to_array(), across**2 * (4 - 3 * across**2) / 4, atol=1e-6)


def test_pump_bound_response_is_the_mixed_part_of_the_doubled_field():
    crystal = core.response(core.BONDS)
    pumps, probes, generated = scenarios.mixing(49)
    change = (core.doubled(crystal, pumps[:, None] + probes)
              - core.doubled(crystal, pumps[:, None]) - core.doubled(crystal, probes))
    np.testing.assert_allclose((change - generated).kernel, 0, atol=1e-12)


def test_uniform_growth_follows_the_closed_form():
    slices = 512
    depths = (np.arange(slices) + 0.5) / slices
    mismatch = np.array([[0.0], [2 * np.pi], [4 * np.pi]])
    source = (core.HORIZONTAL + core.VERTICAL) / np.sqrt(2)
    amplitude = core.growth(source, np.ones((3, slices)), mismatch, depths)
    # The running sum of midpoint slices, against the integral up to each slice's far edge.
    edges = depths + 0.5 / slices
    exact = np.where(mismatch == 0, edges, (np.exp(1j * mismatch * edges) - 1) / (1j * np.where(mismatch == 0, 1, mismatch)))
    expected = source * np.stack((exact.real, exact.imag), axis=-1)
    np.testing.assert_allclose((amplitude - expected).kernel, 0, atol=1e-4)


def test_phase_matching_builds_while_uniform_mismatched_light_cancels():
    _, amplitude = scenarios.phase_matching(256)
    power = (amplitude | amplitude).sum(axis=-1)
    np.testing.assert_allclose(power[:, -1].to_array(), [1.0, 0.0, (2 / np.pi) ** 2], atol=1e-4)


def test_scenes_draw():
    pumps, frames = scenarios.turning(2, 49)
    figures = (render.draw_waveform(*scenarios.waveform(65)), render.draw_mixing(*scenarios.mixing(49)),
               render.draw_growth(*scenarios.phase_matching(256)))
    for figure in figures:
        plt.close(figure)
    images = render.animate(pumps, frames)
    assert len(images) == 2 and not np.array_equal(images[0], images[1])
