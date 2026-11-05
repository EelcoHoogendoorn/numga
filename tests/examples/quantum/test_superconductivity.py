"""The superconducting condensate: its linear response and its motion under a mirror, and its figures after one frame."""

import numpy as np
import pytest

from examples.quantum.superconductivity import core, render, scenarios

ROUND_OFF = 1e-11


@pytest.fixture(scope="module")
def quench():
    return scenarios.quench(scenarios.QUENCH_RATIO, n_steps=scenarios.STEPS_PER_FRAME)


def test_response_is_the_derivative_and_contains_the_phase_zero_mode(quench):
    model, initial, equilibrium, _, _ = quench
    local, feedback = model.response(equilibrium)
    delta = (core.mv.yz * 0.11 + core.mv.zx * 0.07).exp() >> (initial - equilibrium)
    response = local(delta) + feedback(delta.sites.mean())
    # The vector field is quadratic: its central difference is exactly its derivative.
    difference = (model.rate(equilibrium + delta) - model.rate(equilibrium - delta)) / 2
    np.testing.assert_allclose((response - difference).kernel, 0, atol=ROUND_OFF, rtol=0)

    # Turning every pairing phase together gives another equilibrium. Local precession
    # alone moves this disturbance; the changed pairing field cancels it exactly.
    phase = -core.mv.xy.commutator(equilibrium)
    local_phase = local(phase)
    assert np.max(np.abs(local_phase.kernel)) > 0.1
    np.testing.assert_allclose((local_phase + feedback(phase.sites.mean())).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_precession_is_mirror_covariant(quench):
    # Mirroring the condensate, its occupation planes and its pairing map with it, mirrors its motion.
    model, initial, _, _, _ = quench
    for mirror in (core.mv.x, core.mv.z, core.mv.vector([0.6, 0.0, 0.8])):
        mirrored = core.Condensate(mirror >> model.dispersion, mirror >> model.pairing(mirror << core.Spin))
        np.testing.assert_allclose((mirrored.rate(mirror >> initial) - (mirror >> model.rate(initial))).kernel,
                                   0, atol=ROUND_OFF, rtol=0)


def test_the_figures_and_one_frame_draw(quench):
    model, initial, equilibrium, times, history = quench
    gaps, equilibrium_gap = model.gap(history), model.gap(equilibrium)
    render.draw_equilibrium(model.dispersion, initial)
    next(render.animate_quench(model.dispersion, history, gaps, times))
    render.draw_gap(times, gaps, equilibrium_gap)
    disturbance = scenarios.response(model, equilibrium, initial - equilibrium, n_steps=scenarios.STEPS_PER_FRAME)
    render.draw_response(times, gaps - equilibrium_gap, model.gap(disturbance), scenarios.RESPONSE_NAMES)
