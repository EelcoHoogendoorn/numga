"""Scenes for magnetic resonance cover driven spin nutation, power-broadened absorption and
dispersion lines, a Hahn spin echo in an inhomogeneous field, and the echo and free decay built as
maps from state to state across exponentially doubled delays.

Times are measured in microseconds and angular frequencies in radians per microsecond, typical of an
electron spin in a solid with `t1 = 10` us and `t2 = 4` us.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.quantum.magnetic_resonance import core

mv = core.mv
t1, t2 = 10.0, 4.0
# All spins start along the field, turning in the magnet's plane.
equilibrium = core.spin_state(core.magnet_plane.dual())                              # [] State-
# Nutation: three detunings, steady drive, duration, time step, and subsampling stride.
nutation_detunings = np.array([0.0, 0.5, 1.0])
nutation_drive, nutation_seconds, nutation_dt, nutation_stride = 1.0, 40.0, 0.02, 5
# Spectral lines: detuning sweep and drive strengths from weak to saturating.
n_line_detunings = 241
line_detunings = np.linspace(-3.0, 3.0, n_line_detunings)
line_drives = np.array([0.05, 0.3, 1.0])
# Spin echo: spin count, detuning spread, pulse delay, duration, time step, subsampling stride, and RNG seed.
n_spins, spread, delay, echo_seconds, echo_dt, echo_stride, seed = 300, 2.0, 2.5, 8.0, 0.01, 4, 0
frame_stride, frame_duration_ms = 2, 40
# Echo and free decay across exponentially doubled delays.
n_doublings = 11


# --- math -----------------------------------------------------------------------------
def nutation_run(detunings: np.ndarray, drive: float, seconds: float, dt: float,
                 stride: int) -> tuple[core.State, core.State, core.State]:
    """Steps spins from `equilibrium` under `drive` at `detunings` (`[n_detunings]`) and pairs the
    subsampled trajectory (`[n_samples, n_detunings] State-`) with the solved stationary states and
    final stepped state."""
    driven_rates = core.rate_map(detunings, drive, core.magnet_plane, core.drive_plane, core.observer,
                                 t1, t2)                                             # [...] State- <- State-
    start = equilibrium.broadcast_to(detunings.shape)                                # [...] State-
    n_steps = int(seconds / dt) + 1
    states = list(core.trajectory(core.step_evolution(driven_rates, dt), start, n_steps))
    return stack(states[::stride]), core.steady_state(driven_rates, core.observer), states[-1]


def spectral_lines(detunings: np.ndarray, drives: np.ndarray) -> tuple[core.State, core.State]:
    """Solves the stationary states (`[..., ...] State-`) across the grid of `detunings` and `drives`
    together with their vanishing rates of change."""
    rates = core.rate_map(detunings[:, None], drives[None, :], core.magnet_plane, core.drive_plane,
                          core.observer, t1, t2)                                     # [..., ...] State- <- State-
    steady_states = core.steady_state(rates, core.observer)                          # [..., ...] State-
    return steady_states, rates(steady_states)


def echo_stream(n_spins: int, spread: float, delay: float, seconds: float, dt: float, stride: int,
                seed: int) -> tuple[np.ndarray, Iterator[core.State]]:
    """Draws `n_spins` inhomogeneous detunings of width `spread` and streams the Hahn echo ensemble
    (`Iterator[State-]`, each `[n_spins] State-`) every `stride` steps."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, n_spins)
    free_rates = core.rate_map(detunings, 0.0, core.magnet_plane, core.drive_plane, core.observer,
                               t1, t2)                                               # [n_spins] State- <- State-
    start = equilibrium.broadcast_to(detunings.shape)                                # [n_spins] State-
    n_delay_steps = round(delay / dt)
    states = core.echo_trajectory(core.step_evolution(free_rates, dt), core.drive_plane, start,
                                  n_delay_steps, int(seconds / dt) - n_delay_steps)
    subsampled = (state for index, state in enumerate(states) if index % stride == 0)
    return detunings, subsampled


def echo_decay_curves(n_spins: int, spread: float, dt: float, n_doublings: int,
                      seed: int) -> tuple[np.ndarray, core.State, core.State]:
    """Composes `n_doublings` doubled waiting maps with quarter-turn and half-turn pulse maps across
    `n_spins` inhomogeneous detunings, returning total elapsed times alongside the refocused echo and
    free-decay states (`[n_doublings] State-`)."""
    detunings = np.random.default_rng(seed).normal(0.0, spread, n_spins)
    free_rates = core.rate_map(detunings, 0.0, core.magnet_plane, core.drive_plane, core.observer,
                               t1, t2)                                               # [n_spins] State- <- State-
    waiting = stack(list(core.doubled_spans(core.step_evolution(free_rates, dt), n_doublings)))  # [n_doublings, n_spins] State- <- State-
    tip = core.drive_pulse(core.drive_plane, np.pi / 2) >> core.State                # [] State- <- State-
    turn = core.drive_pulse(core.drive_plane, np.pi) >> core.State                   # [] State- <- State-
    echo_map = waiting(turn(waiting(tip))).mean(axis=-1)                             # [n_doublings] State- <- State-
    decay_map = waiting(waiting(tip)).mean(axis=-1)                                  # [n_doublings] State- <- State-
    times = 2 * dt * 2.0 ** np.arange(n_doublings)
    return times, echo_map(equilibrium), decay_map(equilibrium)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quantum.magnetic_resonance import render
    from examples.util.animation import save_animation, save_figure

    history, steady_states, final_state = nutation_run(
        nutation_detunings, nutation_drive, nutation_seconds, nutation_dt, nutation_stride
    )
    save_figure(render.nutation_figure(history, steady_states), "resonance_nutation")
    line_states, line_rates = spectral_lines(line_detunings, line_drives)
    save_figure(render.lines_figure(line_detunings, line_drives, line_states), "resonance_lines")

    detunings, echo_states_iter = echo_stream(n_spins, spread, delay, echo_seconds, echo_dt, echo_stride, seed)
    sample_dt = echo_dt * echo_stride
    recorded_echo: list[core.State] = []

    def recording() -> Iterator[core.State]:
        for state in echo_states_iter:
            recorded_echo.append(state)
            yield state

    save_animation(
        render.echo_frames(detunings, sample_dt, delay, t2, frame_stride, recording()),
        "resonance_echo",
        frame_duration_ms,
    )
    ensemble = stack(recorded_echo)                                                  # [..., n_spins] State-
    save_figure(render.echo_figure(ensemble, sample_dt, delay, t2), "resonance_echo")

    times, echo_states, decay_states = echo_decay_curves(n_spins, spread, echo_dt, n_doublings, seed)
    save_figure(render.decays_figure(times, echo_states, decay_states, t2), "resonance_echo_decay")

    # checks
    # After several `t1` intervals the driven spins settle onto the solved stationary states.
    assert ((final_state - steady_states).antinorm() < 1e-2).all()
    # Stationary states have vanishing generator rates and match the analytical Bloch line shapes.
    assert (line_rates.antinorm() < 1e-12).all()
    grid_detunings, grid_drives = line_detunings[:, None], line_drives[None, :]
    denominator = 1 + (grid_detunings * t2) ** 2 + grid_drives**2 * t1 * t2
    line_spins = 2 * line_states.select.antibivector()                               # [..., ...] Spin-
    assert (((mv.xy & line_spins) - (1 + (grid_detunings * t2) ** 2) / denominator).abs() < 1e-12).all()
    assert ((-(mv.zx & line_spins) - grid_drives * t2 / denominator).abs() < 1e-12).all()
    # At `2 * delay` the ensemble refocuses so its mean transverse length reflects only `t2` dephasing.
    refocused_mean = ensemble[round(2 * delay / sample_dt)].mean(axis=-1).select.antibivector()
    refocused_transverse = (refocused_mean - (mv.xy >> refocused_mean)).antinorm()
    assert ((refocused_transverse - np.exp(-2 * delay / t2)).abs() < 0.02).all()
    # Across all doubled delays the composed echo map decays solely at rate `1 / t2`.
    echo_spins = echo_states.select.antibivector()                                   # [n_doublings] Spin-
    echo_transverse = (echo_spins - (mv.xy >> echo_spins)).antinorm()                # [n_doublings] Scalar+
    assert ((echo_transverse - np.exp(-times / t2)).abs() < 1e-5).all()


def pinned() -> None:
    """Regenerates the gallery's spin-echo animation (`resonance_echo.gif`)."""
    from examples.quantum.magnetic_resonance import render
    from examples.util.animation import pin_animation

    detunings, echo_states_iter = echo_stream(n_spins, spread, delay, echo_seconds, echo_dt, echo_stride, seed)
    sample_dt = echo_dt * echo_stride
    pin_animation(
        render.echo_frames(detunings, sample_dt, delay, t2, frame_stride, echo_states_iter),
        "resonance_echo",
        frame_duration_ms,
    )


if __name__ == "__main__":
    main()
