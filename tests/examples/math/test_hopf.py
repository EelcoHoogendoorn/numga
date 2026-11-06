"""The Hopf fibration: the fibre over a spin is the top eigenspace of the spin paired with the Hopf
form, any two fibres link once, and a carried spinor turns by half the solid angle."""

from __future__ import annotations

import numpy as np

from examples.math.hopf import core, render, scenarios


def test_the_fibre_is_the_top_eigenspace_of_the_paired_form():
    """spin.dot(core.hopf) has eigenvalues -1, -1, 1, 1, and both spinors of the top pair have the spin;
    turning a spinor on the right in the xy plane keeps it on its fibre."""
    spin = core.mv.vector([0.3, -0.5, 0.8]).dual().normalized()
    values, spinors = spin.dot(core.hopf).eigh()
    np.testing.assert_allclose(values.to_array(), [-1.0, -1.0, 1.0, 1.0], atol=1e-12)
    top = spinors[-2:]
    np.testing.assert_allclose((core.hopf(top, top) - spin).kernel, 0.0, atol=1e-12)
    turned = spinors[-1] * (core.mv.xy * np.array([0.4, 2.1])).exp()
    np.testing.assert_allclose((core.hopf(turned, turned) - spin).kernel, 0.0, atol=1e-9)


def test_fibres_have_their_spins_and_link_once():
    spins, projected = scenarios.tori(np.pi * np.array([0.8, 0.5]), 5, 120)
    fibres = core.fibre_start(spins)[..., None] * (core.mv.xy * np.linspace(0.0, 2 * np.pi, 121)).exp()
    np.testing.assert_allclose((core.hopf(fibres, fibres) - spins[..., None]).kernel, 0.0, atol=1e-9)
    linking_numbers = core.linking(projected[[0, 0, 1], [0, 1, 2]], projected[[1, 0, 1], [3, 4, 0]])
    np.testing.assert_allclose(linking_numbers.to_array(), 1.0, atol=2e-2)


def test_a_spinor_carried_around_a_loop_turns_by_half_the_solid_angle():
    polar = 2.2
    _, carried, start_fibre = scenarios.lift(polar, 200)
    half_solid_angle = np.pi * (1 - np.cos(polar))
    np.testing.assert_allclose((carried[-1] - start_fibre[0] * (core.mv.xy * -half_solid_angle).exp()).kernel, 0.0, atol=1e-3)


def test_the_tori_the_lift_and_one_frame_of_the_sweep_draw():
    render.draw_tori(*scenarios.tori(scenarios.polars[:1], 2, 8))
    loop, carried, start_fibre = scenarios.lift(scenarios.loop_polar, 8)
    render.draw_lift(loop, core.stereographic(carried), core.stereographic(start_fibre))
    next(render.animate_sweep(scenarios.sweep(scenarios.spiral_polars, 1, 1, 8)))
