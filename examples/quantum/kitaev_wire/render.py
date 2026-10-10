"""Spatial Majorana mode weights, excitation energies and transport leakage."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.figure import Figure

from examples.animation import capture
from examples.quantum.kitaev_wire import core

INK = "#263b51"
SLOW = "#148d88"
FAST = "#d06c36"
GATE = LinearSegmentedColormap.from_list("gate", ["white", "#e8e4f5"])


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.tick_params(color="0.8")


def shade(chemical: np.ndarray, hopping: float) -> np.ndarray:
    """How deep each site lies in the topological regime, from zero at the threshold 2 * hopping
    to one at the lowest chemical potential."""
    return np.clip((2 * hopping - chemical) / (2 * hopping - chemical.min()), 0, 1)


def draw_formation(chemical_potentials: np.ndarray, energies: core.Scalar,
                   density: core.Scalar, hopping: float) -> Figure:
    values = energies.to_array() / hopping
    weights = density.to_array()
    figure, (spectrum, modes) = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    # Each excitation has two real quadratures with the same frequency.
    spectrum.plot(chemical_potentials / hopping, values[:, ::2], color=INK, linewidth=1)
    spectrum.axvline(2, color="0.65", linestyle=":", linewidth=1)
    spectrum.set(xlim=(chemical_potentials[0] / hopping, chemical_potentials[-1] / hopping),
                 xlabel=r"chemical potential inside / hopping, $\mu/t$", ylabel=r"excitation energy, $E/t$")
    modes.imshow(weights, aspect="auto", origin="upper", interpolation="bilinear",
                 extent=(-0.5, weights.shape[-1] - 0.5, chemical_potentials[-1] / hopping,
                         chemical_potentials[0] / hopping),
                 cmap="magma", norm=PowerNorm(0.55, vmin=0, vmax=weights.max()))
    modes.axhline(2, color="white", linestyle=":", linewidth=0.8, alpha=0.6)
    modes.set(xlabel="site", ylabel=r"$\mu/t$")
    plain_axes(spectrum)
    return figure


def animate_transport(
    progress: np.ndarray, potentials: core.Scalar, history: core.Majorana,
    target: core.Majorana, share: core.Scalar, hopping: float,
) -> list[np.ndarray]:
    weights = history.scalar_norm_squared().to_array()
    expected = target.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    leakage = 1 - share.to_array()
    sites = np.arange(weights.shape[-1])
    maximum = max(weights.max(), expected.max()) * 1.15
    regime = shade(chemical, hopping)

    figure, axes = plt.subplots(3, 1, figsize=(8.4, 5.5), dpi=90,
                                gridspec_kw={"height_ratios": [1, 1, 0.7]})
    figure.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.10, hspace=0.70)
    lines, references, gates, fills = [], [], [], []
    for case, (axis, colour) in enumerate(zip(axes[:2], (SLOW, FAST))):
        # A two-row mesh shaded between its vertices fills the panel with the gate profile.
        gates.append(axis.pcolormesh(sites, [0, maximum], np.broadcast_to(regime[0], (2, len(sites))),
                                     shading="gouraud", cmap=GATE, vmin=0, vmax=1, zorder=0))
        line, = axis.plot(sites, weights[0, case], color=colour, linewidth=2, label="evolving mode")
        reference, = axis.plot(sites, expected[0], color=INK, linestyle="--", linewidth=1,
                                label="instantaneous boundary mode")
        fills.append(axis.fill_between(sites, weights[0, case], color=colour, alpha=0.25))
        lines.append(line)
        references.append(reference)
        axis.set(xlim=(sites[0], sites[-1]), ylim=(0, maximum), xlabel="site", ylabel="mode weight")
        plain_axes(axis)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8, ncol=2)
    traces = [axes[2].plot([], [], color=colour, linewidth=2)[0] for colour in (SLOW, FAST)]
    cursor = axes[2].axvline(0, color="0.7", linewidth=0.8)
    axes[2].set(xlim=(0, 1), ylim=(-0.02, 1.02), xlabel="fraction of gate motion", ylabel="bulk weight")
    plain_axes(axes[2])

    frames = []
    for frame, fraction in enumerate(progress):
        for case, colour in enumerate((SLOW, FAST)):
            lines[case].set_ydata(weights[frame, case])
            references[case].set_ydata(expected[frame])
            gates[case].set_array(np.broadcast_to(regime[frame], (2, len(sites))))
            fills[case].remove()
            fills[case] = axes[case].fill_between(sites, weights[frame, case], color=colour, alpha=0.25)
            traces[case].set_data(progress[:frame + 1], leakage[:frame + 1, case])
        cursor.set_xdata([fraction, fraction])
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def draw_transport(progress: np.ndarray, history: core.Majorana, potentials: core.Scalar,
                   hopping: float) -> Figure:
    weights = history.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    sites = np.arange(weights.shape[-1])
    figure, axes = plt.subplots(1, weights.shape[1], figsize=(9, 3.8), layout="constrained", sharey=True)
    for case, axis in enumerate(axes):
        axis.imshow(weights[:, case], origin="lower", aspect="auto", interpolation="bilinear", cmap="magma",
                     extent=(sites[0], sites[-1], 0, 1), norm=PowerNorm(0.5, vmin=0, vmax=weights.max()))
        axis.contour(sites, progress, chemical, levels=[2 * hopping], colors="white",
                      linewidths=0.7, linestyles="--", alpha=0.7)
        axis.set(xlabel="site")
    axes[0].set_ylabel("fraction of gate motion")
    return figure


def draw_overlap(separations: np.ndarray, profile_separations: np.ndarray,
                 splitting: core.Scalar, potentials: core.Scalar, modes: core.Majorana,
                 hopping: float) -> Figure:
    energies = splitting.to_array() / hopping
    weights = modes.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    regime = shade(chemical, hopping)
    sites = np.arange(weights.shape[-1])
    figure = plt.figure(figsize=(9, 3.8), layout="constrained")
    grid = figure.add_gridspec(len(profile_separations), 2, width_ratios=[1, 1.1])
    spectrum = figure.add_subplot(grid[:, 0])
    spectrum.semilogy(separations, energies, color=INK, linewidth=2)
    spectrum.set(xlabel="gate separation (sites)", ylabel=r"lowest excitation energy, $E/t$")
    plain_axes(spectrum)
    for row, separation in enumerate(profile_separations):
        axis = figure.add_subplot(grid[row, 1])
        axis.imshow(regime[row][None, :], aspect="auto", origin="lower", interpolation="bilinear",
                     extent=(sites[0], sites[-1], 0, weights.max() * 1.1), cmap=GATE, vmin=0, vmax=1,
                     zorder=0)
        for density, colour in zip(weights[row], (SLOW, FAST)):
            axis.fill_between(sites, density, color=colour, alpha=0.35)
            axis.plot(sites, density, color=colour, linewidth=1.5)
        axis.set(xlim=(sites[-1] - separations[-1] - 20, sites[-1]), ylim=(0, weights.max() * 1.1),
                  ylabel=f"{separation:g} sites", yticks=[])
        plain_axes(axis)
    axis.set_xlabel("site")
    return figure
