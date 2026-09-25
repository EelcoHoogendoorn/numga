"""The frequency-doubling crystal: its response has the tetrahedral component law, turns with the
crystal and reverses with its bonds, the [110] cut has its analytic polarization and power, the
phase-matched growth follows the closed form, and the scenes pass their checks and draw."""

import matplotlib.pyplot as plt
import numpy as np

from examples.electromagnetism.second_harmonic import core, render, scenarios

mv = core.mv


def test_response_has_the_tetrahedral_component_law_bound_at_once_or_one_slot_at_a_time():
    crystal = core.response(core.BONDS)
    rng = np.random.default_rng(814)
    first, second = rng.normal(size=(19, 3)), rng.normal(size=(19, 3))
    # The six permutations of three distinct axes share one unit coefficient.
    expected = mv.vector(np.stack((
        first[:, 1] * second[:, 2] + first[:, 2] * second[:, 1],
        first[:, 2] * second[:, 0] + first[:, 0] * second[:, 2],
        first[:, 0] * second[:, 1] + first[:, 1] * second[:, 0],
    ), axis=-1))
    np.testing.assert_allclose((crystal(mv.vector(first), mv.vector(second)) - expected).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((crystal(mv.vector(first))(mv.vector(second)) - expected).kernel, 0, atol=1e-12)


def test_the_response_turns_with_the_crystal_and_reverses_with_its_bonds():
    crystal = core.response(core.BONDS)
    rng = np.random.default_rng(177)
    fields, probes = mv.vector(rng.normal(size=(23, 3))), mv.vector(rng.normal(size=(23, 3)))
    rotation = (mv.xy * 0.37).exp() * (mv.yz * 0.23).exp()
    np.testing.assert_allclose(
        (core.turned(crystal, rotation)(rotation >> fields, rotation >> probes) - (rotation >> crystal(fields, probes))).kernel,
        0, atol=1e-8)
    np.testing.assert_allclose((core.response(-core.BONDS)(fields, probes) + crystal(fields, probes)).kernel, 0, atol=2e-12)


def test_110_cut_has_the_analytic_polarization_and_power():
    crystal = core.response(core.BONDS)
    headings = np.linspace(0, 2 * np.pi, 129)
    across, up = np.cos(headings), np.sin(headings)
    harmonic = core.doubled(crystal, core.HORIZONTAL * across + core.VERTICAL * up)
    expected = -core.HORIZONTAL * (across * up) - core.VERTICAL * (across**2 / 2)
    np.testing.assert_allclose((harmonic - expected).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((harmonic | harmonic).to_array(), across**2 * (4 - 3 * across**2) / 4, atol=1e-12)


def test_uniform_growth_follows_the_closed_form():
    depths = (np.arange(512) + 0.5) / 512
    mismatch = np.array([[0.0], [2 * np.pi], [4 * np.pi]])
    amplitude = core.growth(np.ones((3, 512)), mismatch, depths)
    # The running sum of midpoint slices, against the integral up to each slice's far edge.
    edges = depths + 0.5 / 512
    exact = np.where(mismatch == 0, edges, (np.exp(1j * mismatch * edges) - 1) / (1j * np.where(mismatch == 0, 1, mismatch)))
    expected = mv.scalar(exact.real[..., None]) + mv.xyz * exact.imag
    np.testing.assert_allclose((amplitude - expected).kernel, 0, atol=1e-4)


def test_scenes_pass_their_checks_and_draw():
    pumps, frames = scenarios.turning(2, 49)
    figures = (render.draw_waveform(*scenarios.waveform(65)), render.draw_mixing(*scenarios.mixing(49)),
               render.draw_growth(*scenarios.phase_matching(256)))
    for figure in figures:
        plt.close(figure)
    images = render.animate(pumps, frames)
    assert len(images) == 2 and not np.array_equal(images[0], images[1])
