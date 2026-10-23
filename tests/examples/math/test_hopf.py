"""The Hopf fibration: the fibre over a spin is the top eigenspace of the spin paired with the Hopf
form, any two fibres link once, a carried spinor turns by half the solid angle, and the
scenarios draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.hopf import core, render, scenarios


def test_the_fibre_is_the_top_eigenspace_of_the_paired_form():
    """spin.dot(HOPF) has eigenvalues -1, -1, 1, 1, and both spinors of the top pair have the spin;
    turning a spinor on the right in the xy plane keeps it on its fibre."""
    spin = core.mv.bivector([0.3, -0.5, 0.8]).normalized()
    values, spinors = spin.dot(core.HOPF).eigh()
    np.testing.assert_allclose(values.to_array(), [-1.0, -1.0, 1.0, 1.0], atol=1e-12)
    top = spinors[-2:]
    np.testing.assert_allclose((core.HOPF(top, top) - spin).kernel, 0.0, atol=1e-12)
    turned = core.fibre(spinors[-1], np.array([0.4, 2.1]))
    np.testing.assert_allclose((core.HOPF(turned, turned) - spin).kernel, 0.0, atol=1e-9)


def test_fibres_have_their_spins_and_link_once():
    spins, projected = scenarios.tori(np.pi * np.array([0.8, 0.5]), 5, 120)
    fibres = core.fibre(core.fibre_start(spins), np.linspace(0.0, 2 * np.pi, 121))
    np.testing.assert_allclose((core.HOPF(fibres, fibres) - spins[..., None]).kernel, 0.0, atol=1e-9)
    linking_numbers = core.linking(projected[[0, 0, 1], [0, 1, 2]], projected[[1, 0, 1], [3, 4, 0]])
    np.testing.assert_allclose(linking_numbers.to_array(), 1.0, atol=2e-2)


def test_a_spinor_carried_around_a_loop_turns_by_half_the_solid_angle():
    polar = 2.2
    _, carried, start_fibre = scenarios.lift(polar, 200)
    half_solid_angle = np.pi * (1 - np.cos(polar))
    np.testing.assert_allclose((carried[-1] - start_fibre[0] * (core.mv.xy * -half_solid_angle).exp()).kernel, 0.0, atol=1e-3)


def test_scenarios_draw():
    loop, carried, start_fibre = scenarios.lift(2.2, 200)
    figures = (render.draw_tori(*scenarios.tori(np.pi * np.array([0.8, 0.5]), 5, 120)),
               render.draw_lift(loop, core.stereographic(carried), core.stereographic(start_fibre)))
    for figure in figures:
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(list(render.animate_sweep(scenarios.sweep(scenarios.SPIRAL_POLARS, 1, 3, 60)))) == 3
