"""Pair coherence across the energy band and the gap it produces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.animation import capture
from examples.quantum.superconductivity import core

INK = "#263b51"
PAIR = "#198b87"
FIXED = "#d17c24"
DISPLAY_LEVELS = 33
DISPLAY_CUTOFF = 3.0


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.tick_params(color="0.8")


def draw_equilibrium(dispersion: core.Spin, spins: core.Spin) -> Figure:
    energies = (dispersion | core.mv.z).to_array()
    coherence = spins.cast(core.ga.subspace("x y z")).kernel
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), layout="constrained")
    axes[0].plot(energies, coherence[:, 2] + 0.5, color=INK, linewidth=2)
    axes[0].set(ylabel="occupation per electron state", ylim=(-0.04, 1.04))
    axes[1].plot(energies, coherence[:, 0], color=PAIR, linewidth=2)
    axes[1].set(ylabel="pair coherence along x", ylim=(-0.02, 0.54))
    for axis in axes:
        axis.set(xlabel=r"energy from chemical potential, $\xi/E_0$", xlim=(energies[0], energies[-1]))
        axis.axvline(0, color="0.85", linewidth=0.8, zorder=0)
        plain_axes(axis)
    return figure


def animate_quench(
    dispersion: core.Spin, spins: core.Spin, gaps: core.Spin, times: np.ndarray,
) -> list[np.ndarray]:
    """Arrows in each energy level's pairing plane, beside their collective gap."""
    energies = (dispersion | core.mv.z).to_array()
    transverse = spins.cast(core.ga.subspace("x y")).kernel
    amplitudes = gaps.norm().to_array()
    visible = np.flatnonzero(np.abs(energies) < DISPLAY_CUTOFF)
    indices = visible[np.linspace(0, len(visible) - 1, DISPLAY_LEVELS, dtype=int)]
    energies, transverse = energies[indices], transverse[:, indices]
    origins = np.stack([energies, np.zeros_like(energies), np.zeros_like(energies)], axis=-1)
    colours = plt.colormaps["coolwarm"](np.linspace(0.08, 0.92, len(energies)))

    figure = plt.figure(figsize=(9.0, 3.7), dpi=90)
    axis = figure.add_axes((0.01, 0.08, 0.57, 0.85), projection="3d")
    curve = figure.add_axes((0.68, 0.22, 0.29, 0.64))
    axis.plot(energies, np.zeros_like(energies), np.zeros_like(energies), color="0.7", linewidth=0.8)
    circles = np.linspace(0, 2 * np.pi, 100)
    for energy in energies[::8]:
        axis.plot(np.full_like(circles, energy), 0.5 * np.cos(circles), 0.5 * np.sin(circles),
                  color="0.88", linewidth=0.6)
    axis.set(xlim=(-DISPLAY_CUTOFF, DISPLAY_CUTOFF), ylim=(-0.53, 0.53), zlim=(-0.53, 0.53),
             xlabel=r"$\xi/E_0$", ylabel=r"$s_x$", zlabel=r"$s_y$",
             yticks=[-0.5, 0, 0.5], zticks=[-0.5, 0, 0.5])
    axis.set_box_aspect((2.4, 1, 1))
    axis.view_init(elev=24, azim=-58)
    axis.grid(False)
    for dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        dimension.set_pane_color((1, 1, 1, 0))
    stems = Line3DCollection(np.stack([origins, origins], axis=1), colors=colours, linewidths=1.8)
    axis.add_collection3d(stems)
    tips_artist, = axis.plot([], [], [], "o", color=PAIR, markersize=3)
    tip_path, = axis.plot([], [], [], color=INK, alpha=0.22, linewidth=0.7)
    gap_line, = curve.plot([], [], color=PAIR, linewidth=2)
    gap_tip, = curve.plot([], [], "o", color=PAIR, markersize=6)
    curve.set(xlim=(times[0], times[-1]), ylim=(0, amplitudes.max() * 1.12),
              xlabel=r"$t E_0/\hbar$", ylabel=r"$|\Delta|/E_0$")
    plain_axes(curve)

    frames = []
    for index, coherence in enumerate(transverse):
        tips = np.column_stack([energies, coherence])
        # Short wings mark the phase direction even where neighbouring stems overlap.
        wings = coherence * 0.83
        sideways = np.column_stack([-coherence[:, 1], coherence[:, 0]]) * 0.08
        first = np.column_stack([energies, wings + sideways])
        second = np.column_stack([energies, wings - sideways])
        segments = np.stack([np.stack([origins, tips], axis=1),
                             np.stack([first, tips], axis=1),
                             np.stack([second, tips], axis=1)], axis=1)
        stems.set_segments(segments.reshape(-1, 2, 3))
        stems.set_color(np.repeat(colours, 3, axis=0))
        tips_artist.set_data_3d(*tips.T)
        tip_path.set_data_3d(*tips.T)
        gap_line.set_data(times[:index + 1], amplitudes[:index + 1])
        gap_tip.set_data([times[index]], [amplitudes[index]])
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def draw_gap(times: np.ndarray, gaps: core.Spin, equilibrium_gap: core.Spin) -> Figure:
    amplitudes = gaps.norm().to_array()
    equilibrium = equilibrium_gap.norm().to_array()
    figure, axis = plt.subplots(figsize=(7.5, 2.8), layout="constrained")
    axis.plot(times, amplitudes, color=PAIR, linewidth=2, label="isolated condensate")
    axis.axhline(equilibrium, color=INK, linestyle="--", linewidth=1, label="final ground-state gap")
    axis.set(xlim=(times[0], times[-1]), xlabel=r"$t E_0/\hbar$", ylabel=r"$|\Delta|/E_0$")
    axis.legend(frameon=False, ncol=2)
    plain_axes(axis)
    return figure


def draw_response(
    times: np.ndarray, exact: core.Spin, predicted: core.Spin, names: tuple[str, str],
) -> Figure:
    exact_amplitude = (exact | core.mv.x).to_array()
    predicted_amplitude = (predicted | core.mv.x).to_array()
    figure, axis = plt.subplots(figsize=(7.5, 3.0), layout="constrained")
    axis.plot(times, exact_amplitude, color=INK, linewidth=3, alpha=0.5, label="nonlinear evolution")
    for values, name, colour in zip(predicted_amplitude.T, names, (PAIR, FIXED)):
        axis.plot(times, values, color=colour, linewidth=1.5, linestyle="--", label=name)
    axis.axhline(0, color="0.85", linewidth=0.8, zorder=0)
    axis.set(xlim=(times[0], times[-1]), xlabel=r"$t E_0/\hbar$", ylabel=r"$\delta\Delta_x/E_0$")
    axis.legend(frameon=False, ncol=3, loc="upper center", fontsize=9)
    plain_axes(axis)
    return figure
