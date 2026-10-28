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
# All spins along the field, turning in the magnet's plane.
EQUILIBRIUM = core.state(core.MAGNET_PLANE.dual())  # [] State-
# Nutation: the detunings, the drive, the duration and time step, and the steps between states kept.
NUTATION_DETUNINGS = np.array([0.0, 0.5, 1.0])
NUTATION_DRIVE, NUTATION_SECONDS, NUTATION_DT, NUTATION_EVERY = 1.0, 40.0, 0.02, 5
# The lines: detunings, and the drive strengths from weak to saturating.
N_LINE_DETUNINGS = 241
LINE_DETUNINGS = np.linspace(-3.0, 3.0, N_LINE_DETUNINGS)
LINE_DRIVES = np.array([0.05, 0.3, 1.0])
# The echo: the spins, the spread of their detunings, the delay, the duration and time step, the steps
# between states kept, the seed; the steps between frames drawn, and a frame's duration in milliseconds.
N_SPINS, SPREAD, DELAY, ECHO_SECONDS, ECHO_DT, ECHO_EVERY, SEED = 300, 2.0, 2.5, 8.0, 0.01, 4, 0
FRAME_STRIDE, FRAME_DURATION = 2, 40
# The echo and the free decay over doubling delays: the shortest delay's steps, and the doublings.
N_DOUBLINGS = 11


# --- math -----------------------------------------------------------------------------
def nutation(detunings: np.ndarray, drive: float, seconds: float, dt: float, every: int):
    """Spins switched on to a steady drive at the given detunings, from equilibrium: their states
    every few steps, the steady states they settle into, and the last state reached."""
    driven_rates = core.generator(detunings, drive, T1, T2)                              # [detunings] State- <- State-
    start = EQUILIBRIUM.broadcast_to(detunings.shape)                                    # [detunings] State-
    states = list(core.evolve(core.evolution(driven_rates, dt), start, int(seconds / dt) + 1))
    return stack(states[::every]), core.steady(driven_rates), states[-1]                 # [times, detunings] State-, [detunings] State-, [detunings] State-


def lines(detunings: np.ndarray, drives: np.ndarray):
    """The steady states over a grid of detunings and drive strengths, in one solve, and their rates of
    change, which vanish."""
    rates = core.generator(detunings[:, None], drives[None, :], T1, T2)                 # [detunings, drives] State- <- State-
    steady_states = core.steady(rates)                                                   # [detunings, drives] State-
    return steady_states, rates(steady_states)                                           # [detunings, drives] State-, [detunings, drives] State-


def echo(n_spins: int, spread: float, delay: float, seconds: float, dt: float, every: int, seed: int):
    """A spin echo: an ensemble of spins with detunings scattered by an uneven field is tipped
    into the transverse plane, left to fan out for the delay, turned over by a pulse, and left to
    refocus. Returns the detunings and the ensemble every few steps."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, n_spins)
    # No drive between the pulses; the spins start along the field.
    free_rates = core.generator(detunings, 0.0, T1, T2)                                  # [spins] State- <- State-
    start = EQUILIBRIUM.broadcast_to(detunings.shape)                                    # [spins] State-
    n_delay_steps = round(delay / dt)
    states = core.echo(core.evolution(free_rates, dt), start, n_delay_steps, int(seconds / dt) - n_delay_steps)
    return detunings, stack(list(states)[::every])                                       # [spins], [times, spins] State-


def echo_decay(n_spins: int, spread: float, dt: float, n_doublings: int, seed: int):
    """The echo and the free decay of an ensemble with detunings scattered by an uneven field, as
    maps from state to state: one step as a map on states, doubled for delays from dt to
    `2 ** (n_doublings - 1) * dt`, composed with the pulses, and averaged over the spins. Returns the times after
    the first pulse, and the states the two sequences leave from equilibrium."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, n_spins)
    free_rates = core.generator(detunings, 0.0, T1, T2)                                  # [spins] State- <- State-
    waiting = stack(list(core.doublings(core.evolution(free_rates, dt), n_doublings)))   # [delays, spins] State- <- State-
    tip = core.pulse(np.pi / 2) >> core.State                                            # [] State- <- State-
    turn = core.pulse(np.pi) >> core.State                                               # [] State- <- State-
    echo_map = waiting(turn(waiting(tip))).mean(axis=-1)                                 # [delays] State- <- State-
    decay_map = waiting(waiting(tip)).mean(axis=-1)                                      # [delays] State- <- State-
    times = 2 * dt * 2.0 ** np.arange(n_doublings)
    return times, echo_map(EQUILIBRIUM), decay_map(EQUILIBRIUM)                          # [delays], [delays] State-, [delays] State-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.magnetic_resonance import render

    history, steady_states, last = nutation(NUTATION_DETUNINGS, NUTATION_DRIVE, NUTATION_SECONDS, NUTATION_DT, NUTATION_EVERY)
    save_figure(render.draw_nutation(history, steady_states), "resonance_nutation")
    line_states, line_rates = lines(LINE_DETUNINGS, LINE_DRIVES)
    save_figure(render.draw_lines(LINE_DETUNINGS, LINE_DRIVES, line_states), "resonance_lines")
    detunings, ensemble = echo(N_SPINS, SPREAD, DELAY, ECHO_SECONDS, ECHO_DT, ECHO_EVERY, SEED)
    frame_time = ECHO_DT * ECHO_EVERY
    save_figure(render.draw_echo(ensemble, frame_time, DELAY, T2), "resonance_echo")
    save_animation(render.animate_echo(detunings, ensemble, frame_time, DELAY, T2, FRAME_STRIDE), "resonance_echo", FRAME_DURATION)
    times, echo_states, decay_states = echo_decay(N_SPINS, SPREAD, ECHO_DT, N_DOUBLINGS, SEED)
    save_figure(render.draw_decays(times, echo_states, decay_states, T2), "resonance_echo_decay")

    # checks
    # After several T1 the driven spins have reached the steady states the solve predicts.
    np.testing.assert_allclose(last.kernel, steady_states.kernel, atol=1e-2)
    # The steady states do not change, and they match the closed form: along the field
    # `(1 + (D * T2) ** 2) / d` and absorbing `W * T2 / d`, with `d = 1 + (D * T2) ** 2 + W**2 * T1 * T2`.
    # A spin's parts along the field and against the drive are its pairings with the planes xy and zx.
    np.testing.assert_allclose(line_rates.kernel, 0.0, atol=1e-12)
    D, W = LINE_DETUNINGS[:, None], LINE_DRIVES[None, :]
    d = 1 + (D * T2) ** 2 + W**2 * T1 * T2
    spins = 2 * line_states.select[2]
    np.testing.assert_allclose((mv.xy & spins).to_array(), (1 + (D * T2) ** 2) / d, atol=1e-12)
    np.testing.assert_allclose(-(mv.zx & spins).to_array(), W * T2 / d, atol=1e-12)
    # At twice the delay the spins have refocused: the mean transverse spin has decayed only by the true
    # dephasing, `np.exp(-2 * delay / T2)`, not by the spread of the field.
    spins = 2 * ensemble[int(round(2 * DELAY / frame_time))].select[2]
    transverse = np.hypot((mv.yz & spins).to_array().mean(), (mv.zx & spins).to_array().mean())
    np.testing.assert_allclose(transverse, np.exp(-2 * DELAY / T2), rtol=0.02)
    # Over every delay the echo has lost only the true dephasing, `np.exp(-times / T2)`, whatever the
    # spread of the field.
    spins = 2 * echo_states.select[2]
    np.testing.assert_allclose(np.hypot((mv.yz & spins).to_array(), (mv.zx & spins).to_array()), np.exp(-times / T2), rtol=1e-5)


if __name__ == "__main__":
    main()
