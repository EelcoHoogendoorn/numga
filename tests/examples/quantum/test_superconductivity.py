"""Self-consistency and collective superconducting response."""

import numpy as np
import pytest

from numga import stack
from examples.quantum.superconductivity import core, scenarios

ROUND_OFF = 1e-11


@pytest.fixture(scope="module")
def quench():
    return scenarios.quench(scenarios.QUENCH_RATIO)


def test_equilibrium_gap_and_phase_symmetry(quench):
    model, _, equilibrium, _, _ = quench
    np.testing.assert_allclose(model.rate(equilibrium).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(equilibrium.scalar_norm_squared().kernel, 0.25, atol=ROUND_OFF, rtol=0)
    coupling = scenarios.COUPLING * scenarios.QUENCH_RATIO
    continuum_gap = scenarios.CUTOFF / np.sinh(2 * scenarios.CUTOFF / coupling)
    # This tolerance tests the finite energy quadrature against the continuum gap equation.
    np.testing.assert_allclose(model.gap(equilibrium).kernel[0], continuum_gap, rtol=1e-5)

    phase = 0.73
    rotor = (core.mv.xy * (-phase / 2)).exp()
    rotated = rotor >> equilibrium
    np.testing.assert_allclose(model.rate(rotated).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((model.gap(rotated) - (rotor >> model.gap(equilibrium))).kernel,
                               0, atol=ROUND_OFF, rtol=0)


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


def test_collective_response_predicts_a_weak_quench():
    model, initial, equilibrium, _, history = scenarios.quench(scenarios.SMALL_QUENCH_RATIO)
    disturbance = scenarios.response(model, equilibrium, initial - equilibrium)
    exact = model.gap(history) - model.gap(equilibrium)
    predicted = model.gap(disturbance)
    scale = np.max(np.abs(exact.kernel))
    collective_error = np.max(np.abs((exact - predicted[:, 0]).kernel)) / scale
    frozen_error = np.max(np.abs((exact - predicted[:, 1]).kernel)) / scale
    assert collective_error < 0.005
    assert frozen_error > 0.2


def test_midpoint_converges_at_second_order(quench):
    model, initial, _, _, _ = quench
    duration = 2.0
    steps = (40, 80, 160)
    ends = [tuple(core.evolution(initial, model.rate, duration / count, count, count,
                                 scenarios.MIDPOINT_ITERATIONS))[-1] for count in steps]
    coarse = np.linalg.norm((ends[0] - ends[1]).kernel)
    fine = np.linalg.norm((ends[1] - ends[2]).kernel)
    assert 3.8 < coarse / fine < 4.2


def test_figures_and_animation_draw(quench):
    import matplotlib.pyplot as plt
    from examples.quantum.superconductivity import render

    model, initial, equilibrium, times, history = quench
    figures = [render.draw_equilibrium(model.dispersion, initial),
               render.draw_gap(times, model.gap(history), model.gap(equilibrium))]
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)
    frames = render.animate_quench(model.dispersion, history[:2], model.gap(history[:2]), times[:2])
    assert len(frames) == 2
    assert np.any(frames[0] != frames[1])
