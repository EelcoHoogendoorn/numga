"""Drawing for magnetic resonance: Bloch-ball trajectories, absorption and dispersion lines, and the
spin echo."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.animation import capture
from examples.quantum.magnetic_resonance import core

COLOURS = ("#c0392b", "#7d3c98", "#2e86c1")


def components(rho: core.State) -> np.ndarray:
    """The Bloch vector's x, y and z components."""
    return core.bloch(rho).cast(core.ga.subspace("x y z")).kernel


def ball(ax) -> None:
    """The unit Bloch sphere, faintly, with its axes."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.85", linewidth=0.4)
    for axis in np.eye(3):
        ax.plot(*np.stack([-axis, axis]).T, color="0.6", linewidth=0.6)
    # The sphere fills the axes: limits at its radius, and the box zoomed past its default padding.
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.set_axis_off()


def draw_nutation(states: Iterable[core.State], settled: core.State) -> plt.Figure:
    """Each spin's Bloch vector from equilibrium, nutating about the drive and spiralling into its
    steady state, marked with a dot; the states are consumed as they are yielded."""
    paths, ends = components(stack(list(states))), components(settled)  # [times, detunings, 3], [detunings, 3]
    figure = plt.figure(figsize=(6.5, 6.5))
    ax = figure.add_axes((0, 0, 1, 1), projection="3d")
    ball(ax)
    for index, colour in enumerate(COLOURS[: paths.shape[1]]):
        ax.plot(*paths[:, index].T, color=colour, linewidth=1.0)
        ax.scatter(*ends[index], color=colour, s=30)
    ax.view_init(elev=18, azim=-50)
    return figure


def draw_lines(detunings: np.ndarray, drives: np.ndarray, settled: core.State) -> plt.Figure:
    """Absorption and dispersion against detuning, one curve per drive strength, each divided by
    the drive so weak drives stay visible: a stronger drive broadens and flattens the line."""
    r = components(settled)                                              # [detunings, drives, 3]
    figure, (absorbing, dispersing) = plt.subplots(1, 2, figsize=(11, 4.2))
    for index, (drive, colour) in enumerate(zip(drives, COLOURS)):
        absorbing.plot(detunings, -r[:, index, 1] / drive, color=colour, label=f"drive {drive:g} rad/µs")
        dispersing.plot(detunings, r[:, index, 0] / drive, color=colour)
    absorbing.set_title("absorption, per unit drive")
    dispersing.set_title("dispersion, per unit drive")
    for ax in (absorbing, dispersing):
        ax.set_xlabel("detuning (rad/µs)")
        ax.axhline(0, color="0.8", linewidth=0.6)
    absorbing.legend()
    figure.tight_layout()
    return figure


def signal(ensemble: core.State) -> np.ndarray:
    """The ensemble's mean transverse Bloch vector, its length at each time: what a pick-up coil sees."""
    r = components(ensemble)                                             # [times, spins, 3]
    return np.hypot(r[..., 0].mean(axis=-1), r[..., 1].mean(axis=-1))


def draw_echo(states: Iterable[core.State], dt: float, delay: float, t2: float) -> plt.Figure:
    """The signal after the first pulse, for an ensemble recorded every dt: it vanishes as the spins
    fan out, and returns at twice the delay after the second pulse, reduced only by the true
    dephasing."""
    ensemble = stack(list(states))                                       # [times, spins] State
    times = np.arange(ensemble.shape[0]) * dt
    figure, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(times, signal(ensemble), color="#2e86c1", label="mean transverse spin")
    ax.plot(times, np.exp(-times / t2), "--", color="0.5", label="exp(-t / T2)")
    for when, name in ((delay, "π pulse"), (2 * delay, "echo")):
        ax.axvline(when, color="0.7", linewidth=0.8)
        ax.annotate(name, (when, 0.95), textcoords="offset points", xytext=(4, 0))
    ax.set_xlabel("time (µs)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    figure.tight_layout()
    return figure


def transverse(states: core.State) -> np.ndarray:
    """The length of each state's transverse Bloch vector: what a pick-up coil sees of it."""
    r = components(states)
    return np.hypot(r[..., 0], r[..., 1])


def draw_decays(times: np.ndarray, echoed: core.State, faded: core.State, t2: float) -> plt.Figure:
    """The signal at each time after the first pulse, on a logarithmic time axis: with a half-turn
    pulse halfway, the echo, which fades only at the rate 1 / T2; without it, the free decay, which
    the uneven field ends within about a microsecond."""
    fine = np.geomspace(times[0], times[-1], 200)
    figure, ax = plt.subplots(figsize=(8, 3.8))
    ax.semilogx(times, transverse(echoed), "o-", color="#2e86c1", label="echo")
    ax.semilogx(times, transverse(faded), "o-", color="#c0392b", label="free decay")
    ax.semilogx(fine, np.exp(-fine / t2), "--", color="0.5", label="exp(-t / T2)")
    ax.set_xlabel("time after the first pulse (µs)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    figure.tight_layout()
    return figure


def animate_echo(detunings: np.ndarray, states: Iterable[core.State], dt: float, delay: float, t2: float,
                 every: int) -> list[np.ndarray]:
    """The spins seen from above the field, fanning out and refocusing, beside the signal so far;
    one frame for every few recorded states."""
    ensemble = stack(list(states))                                       # [times, spins] State
    times = np.arange(ensemble.shape[0]) * dt
    r = components(ensemble)                                             # [times, spins, 3]
    strength = signal(ensemble)
    order = np.argsort(detunings)
    frames = []
    for index in range(0, len(times), every):
        figure, (top, trace) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": (1, 1.4)})
        top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.8"))
        top.scatter(r[index, order, 0], r[index, order, 1], c=detunings[order], cmap="coolwarm", s=6)
        top.set_xlim(-1.1, 1.1)
        top.set_ylim(-1.1, 1.1)
        top.set_aspect("equal")
        top.set_axis_off()
        top.set_title(f"t = {times[index]:.2f} µs")
        trace.plot(times[: index + 1], strength[: index + 1], color="#2e86c1")
        trace.plot(times, np.exp(-times / t2), "--", color="0.8")
        trace.axvline(delay, color="0.85", linewidth=0.8)
        trace.set_xlim(times[0], times[-1])
        trace.set_ylim(0, 1.05)
        trace.set_xlabel("time (µs)")
        trace.set_title("mean transverse spin")
        figure.tight_layout()
        frames.append(capture(figure))
        plt.close(figure)
    return frames
