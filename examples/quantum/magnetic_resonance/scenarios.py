"""Scenes for magnetic resonance: a spin nutating under a steady drive, the absorption and
dispersion lines and their broadening under strong drive, a spin echo from an ensemble of spins in
a field that is not quite uniform, and the echo and the free decay over a range of delays, built as
maps from state to state.

Times are in microseconds and frequencies in radians per microsecond, typical of an electron spin
in a solid: `T1` is 10 us and `T2` is 4 us.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.magnetic_resonance import core

mv = core.mv
T1, T2 = 10.0, 4.0
# All spins along the field.
EQUILIBRIUM = core.state(mv.z)                     # [] State


# --- math -----------------------------------------------------------------------------
def nutation(detunings: np.ndarray, drive: float, seconds: float, dt: float, every: int):
    """Spins switched on to a steady drive at the given detunings, from equilibrium: their states
    every few steps, and the steady states they settle into."""
    rates = core.generator(mv.scalar(detunings[:, None]), mv.scalar([drive]), T1, T2)   # [detunings] State <- State
    states = list(core.evolve(core.evolution(rates, dt), EQUILIBRIUM.broadcast_to(detunings.shape), int(seconds / dt) + 1))
    settled = core.steady(rates)

    # --- checks
    # After several T1 the driven spins have reached the steady states the solve predicts.
    np.testing.assert_allclose(states[-1].kernel, settled.kernel, atol=1e-2)
    return stack(states[::every]), settled                                              # [times, detunings] State, [detunings] State


def lines(detunings: np.ndarray, drives: np.ndarray) -> core.State:
    """The steady states over a grid of detunings and drive strengths, in one solve."""
    rates = core.generator(mv.scalar(detunings[:, None, None]), mv.scalar(drives[None, :, None]), T1, T2)
    settled = core.steady(rates)                                                        # [detunings, drives] State

    # --- checks
    # The steady states do not change, and they match the closed form: along the field
    # `(1 + (D * T2) ** 2) / d` and absorbing `W * T2 / d`, with `d = 1 + (D * T2) ** 2 + W**2 * T1 * T2`.
    np.testing.assert_allclose(rates(settled).kernel, 0.0, atol=1e-12)
    D, W = detunings[:, None], drives[None, :]
    d = 1 + (D * T2) ** 2 + W**2 * T1 * T2
    r = core.bloch(settled)
    np.testing.assert_allclose((r | mv.z).to_array(), (1 + (D * T2) ** 2) / d, atol=1e-12)
    np.testing.assert_allclose(-(r | mv.y).to_array(), W * T2 / d, atol=1e-12)
    return settled


def echo(spins: int, spread: float, delay: float, seconds: float, dt: float, every: int, seed: int):
    """A spin echo: an ensemble of spins with detunings scattered by an uneven field is tipped
    into the transverse plane, left to fan out for the delay, turned over by a pulse, and left to
    refocus. Returns the detunings and the ensemble every few steps."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, spins)
    # No drive between the pulses; the spins start along the field.
    rates = core.generator(mv.scalar(detunings[:, None]), mv.scalar([0.0]), T1, T2)    # [spins] State <- State
    start = EQUILIBRIUM.broadcast_to(detunings.shape)                                    # [spins] State
    waits = round(delay / dt)
    states = core.echo(core.evolution(rates, dt), start, waits, int(seconds / dt) - waits)
    ensemble = stack(list(states)[::every])                                              # [times, spins] State

    # --- checks
    # At twice the delay the spins have refocused: the mean transverse Bloch vector has decayed only
    # by the true dephasing, `np.exp(-2 * delay / T2)`, not by the spread of the field.
    r = core.bloch(ensemble[int(round(2 * delay / (dt * every)))])
    transverse = np.hypot((r | mv.x).to_array().mean(), (r | mv.y).to_array().mean())
    np.testing.assert_allclose(transverse, np.exp(-2 * delay / T2), rtol=0.02)
    return detunings, ensemble


def echo_decay(spins: int, spread: float, dt: float, count: int, seed: int):
    """The echo and the free decay of an ensemble with detunings scattered by an uneven field, as
    maps from state to state: one step on the open state, doubled for delays from dt to
    `2 ** (count - 1) * dt`, composed with the pulses, and averaged over the spins. Returns the times after
    the first pulse, and the states the two sequences leave from equilibrium."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, spins)
    rates = core.generator(mv.scalar(detunings[:, None]), mv.scalar([0.0]), T1, T2)     # [spins] State <- State
    waiting = stack(list(core.doublings(core.evolution(rates, dt), count)))              # [delays, spins] State <- State
    tip = core.pulse(np.pi / 2) >> core.State                                            # [] State <- State
    turn = core.pulse(np.pi) >> core.State                                               # [] State <- State
    echo = waiting(turn(waiting(tip))).mean(axis=-1)                                     # [delays] State <- State
    decay = waiting(waiting(tip)).mean(axis=-1)                                          # [delays] State <- State
    times = 2 * dt * 2.0 ** np.arange(count)
    echoed, faded = echo(EQUILIBRIUM), decay(EQUILIBRIUM)                                # [delays] State

    # --- checks
    # The echo has lost only the true dephasing, `np.exp(-times / T2)`, whatever the spread of the field.
    r = core.bloch(echoed)
    np.testing.assert_allclose(np.hypot((r | mv.x).to_array(), (r | mv.y).to_array()), np.exp(-times / T2), rtol=1e-5)
    return times, echoed, faded


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.quantum.magnetic_resonance import render

    save_figure(render.draw_nutation(*nutation(np.array([0.0, 0.5, 1.0]), 1.0, 40.0, 0.02, 5)), "resonance_nutation")
    detunings, drives = np.linspace(-3.0, 3.0, 241), np.array([0.05, 0.3, 1.0])
    save_figure(render.draw_lines(detunings, drives, lines(detunings, drives)), "resonance_lines")
    detunings, ensemble = echo(300, 2.0, 2.5, 8.0, 0.01, 4, 0)
    save_figure(render.draw_echo(ensemble, 0.04, 2.5, T2), "resonance_echo")
    save_animation(render.animate_echo(detunings, ensemble, 0.04, 2.5, T2, 2), "resonance_echo", 40)
    save_figure(render.draw_decays(*echo_decay(300, 2.0, 0.01, 11, 0), T2), "resonance_echo_decay")
