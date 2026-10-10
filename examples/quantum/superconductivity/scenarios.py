"""A pairing-strength quench and the collective response of a superconducting condensate."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.superconductivity import core

LEVELS = 240
CUTOFF = 5.0
COUPLING = 4.4
QUENCH_RATIO = 0.88
SMALL_QUENCH_RATIO = 0.99
EQUILIBRIUM_ITERATIONS = 160
DT = 0.05
STEPS = 480
STRIDE = 2
MIDPOINT_ITERATIONS = 3
DURATION_MS = 50
FEEDBACK_STRENGTHS = np.array([1.0, 0.0])
RESPONSE_NAMES = ("collective response", "fixed pairing field")


# --- math -----------------------------------------------------------------------------
def quench(ratio: float) -> tuple[core.Condensate, core.Spin, core.Spin, np.ndarray, core.Spin]:
    dispersion = core.energy_levels(LEVELS, CUTOFF)
    # Remove occupation; only the transverse coherence contributes to pairing.
    transverse = core.Spin - core.mv.z * (core.mv.z | core.Spin)
    before = core.Condensate(dispersion, COUPLING * transverse)
    after = core.Condensate(dispersion, ratio * before.pairing)
    initial = before.equilibrium(core.mv.x, EQUILIBRIUM_ITERATIONS)
    equilibrium = after.equilibrium(core.mv.x, EQUILIBRIUM_ITERATIONS)
    history = stack(core.evolution(initial, after.rate, DT, STEPS, STRIDE, MIDPOINT_ITERATIONS))
    times = np.arange(STEPS // STRIDE + 1) * STRIDE * DT
    return after, initial, equilibrium, times, history


def response(
    model: core.Condensate, equilibrium: core.Spin, disturbance: core.Spin,
) -> core.Spin:
    local, feedback = model.response(equilibrium)
    # Batch the full response and the approximation that holds the pairing field fixed.
    feedbacks = feedback * FEEDBACK_STRENGTHS
    disturbance = disturbance.broadcast_to(FEEDBACK_STRENGTHS.shape)

    def rate(delta: core.Spin) -> core.Spin:
        return local(delta) + feedbacks(delta.sites.mean())

    return stack(core.evolution(disturbance, rate, DT, STEPS, STRIDE, MIDPOINT_ITERATIONS))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quantum.superconductivity import render

    model, initial, equilibrium, times, history = quench(QUENCH_RATIO)
    save_figure(render.draw_equilibrium(model.dispersion, initial), "superconductivity_equilibrium")
    save_animation(render.animate_quench(model.dispersion, history, model.gap(history), times),
                   "superconductivity_quench", DURATION_MS)
    save_figure(render.draw_gap(times, model.gap(history), model.gap(equilibrium)),
                "superconductivity_gap")

    model, initial, equilibrium, times, history = quench(SMALL_QUENCH_RATIO)
    disturbance = response(model, equilibrium, initial - equilibrium)
    exact = model.gap(history) - model.gap(equilibrium)
    predicted = model.gap(disturbance)
    save_figure(render.draw_response(times, exact, predicted, RESPONSE_NAMES),
                "superconductivity_response")


if __name__ == "__main__":
    main()
