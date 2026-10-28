"""Energy levels, electron configurations, and spin exchange in the Hubbard dimer."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_dimer import core

COLOURS = ("#7d3c98", "#2e86c1", "#c0392b")
CONFIGURATION_LABELS = (
    r"$L\uparrow\,L\downarrow$", r"$L\uparrow\,R\uparrow$",
    r"$L\uparrow\,R\downarrow$", r"$L\downarrow\,R\uparrow$",
    r"$L\downarrow\,R\downarrow$", r"$R\uparrow\,R\downarrow$",
)
# The size of the text and the width of the curves in the plots.
FONT_SIZE, LINE_WIDTH = 7, 1.1


# --- plumbing -------------------------------------------------------------------------
def plain_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Label a quantity plot's axes in the plots' text size, and keep only the left and bottom spines."""
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE)
    ax.spines[["top", "right"]].set_visible(False)


def draw_spectrum(repulsion: np.ndarray, hopping: float, energies: core.Scalar, exchange: np.ndarray,
                  perturbative: np.ndarray, strong_repulsion: np.ndarray) -> plt.Figure:
    """The six two-electron levels, and the energy needed to turn the singlet into a triplet."""
    figure, (levels, gap) = plt.subplots(1, 2, figsize=(5.6, 2.4), layout="constrained")
    ratios = repulsion / hopping
    energy_values = energies.to_array() / hopping                                                          # [samples, modes]

    ground_line, = levels.plot(ratios, energy_values[:, 0], color=COLOURS[0], linewidth=LINE_WIDTH)
    triplet_lines = levels.plot(ratios, energy_values[:, 1:4], color=COLOURS[1], linewidth=LINE_WIDTH)
    charge_lines = levels.plot(ratios, energy_values[:, 4:], color=COLOURS[2], linewidth=LINE_WIDTH)
    # The energy's zero, where the triplets stay.
    levels.axhline(0, color="black", linewidth=0.6, zorder=0)
    levels.set(xlim=(ratios[0], ratios[-1]), xticks=[], yticks=[])
    plain_axes(levels, r"repulsion / hopping, $U/t$", r"energy / hopping, $E/t$")
    levels.legend((ground_line, triplet_lines[0], charge_lines[0]),
                  ("singlet ground state", "three triplets", "two excited singlets"), loc="upper left", frameon=False,
                  fontsize=FONT_SIZE)

    gap.plot(ratios, exchange / hopping, color=COLOURS[0], linewidth=LINE_WIDTH, label="from the singlet block")
    gap.plot(strong_repulsion / hopping, perturbative / hopping, "--", color="0.5", linewidth=LINE_WIDTH, label=r"$4t^2/U$")
    gap.set(xlim=(ratios[0], ratios[-1]), ylim=(0, exchange[0] / hopping * 1.05), xticks=[], yticks=[])
    plain_axes(gap, r"repulsion / hopping, $U/t$", r"singlet-triplet gap / hopping, $J/t$")
    gap.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE)
    return figure


def draw_ground_states(names: tuple[str, ...], repulsion_ratios: np.ndarray, probabilities: core.Scalar) -> plt.Figure:
    """How likely each electron configuration is in the ground state, one row per case."""
    figure, ax = plt.subplots(figsize=(4.4, 1.9), layout="constrained")
    # Each cell a rectangle of its own, the first case on top.
    heatmap = ax.pcolormesh(probabilities.to_array(), cmap="Purples", vmin=0, vmax=0.5)
    ax.invert_yaxis()
    ax.set(xticks=np.arange(len(CONFIGURATION_LABELS)) + 0.5, yticks=np.arange(len(names)) + 0.5)
    ax.set_xticklabels(CONFIGURATION_LABELS, fontsize=FONT_SIZE)
    ax.set_yticklabels([f"{name}\n$U/t = {ratio:g}$" for name, ratio in zip(names, repulsion_ratios)], fontsize=FONT_SIZE)
    ax.set_xlabel("electron configuration", fontsize=FONT_SIZE)
    ax.tick_params(length=0)
    bar = figure.colorbar(heatmap, ax=ax, shrink=0.85, pad=0.02, ticks=[0, 0.25, 0.5])
    bar.set_label("probability", fontsize=FONT_SIZE)
    bar.ax.tick_params(labelsize=FONT_SIZE)
    return figure


def draw_exchange(phase: np.ndarray, probabilities: core.Scalar, spin_only: np.ndarray) -> plt.Figure:
    """Spin swapping and temporary double occupancy for each case, left to right, on each case's own
    exchange timescale."""
    values = probabilities.to_array()                                                                      # [cases, times, configurations]
    figure, axes = plt.subplots(1, len(values), figsize=(2.5 * len(values), 2.3), layout="constrained", sharey=True)
    for ax, case in zip(axes, values):
        swapped, = ax.plot(phase / np.pi, case[:, 3], color=COLOURS[0], linewidth=LINE_WIDTH)
        approximation, = ax.plot(phase / np.pi, spin_only, "--", color="0.5", linewidth=LINE_WIDTH)
        charge, = ax.plot(phase / np.pi, case[:, 0] + case[:, 5], color=COLOURS[2], linewidth=LINE_WIDTH)
        # Probabilities read off at none, a half and certainty, along one cycle of the exchange.
        ax.set(xlim=(phase[0] / np.pi, phase[-1] / np.pi), ylim=(0, 1.05), xticks=[], yticks=[0, 0.5, 1])
        plain_axes(ax, r"exchange phase, $J\tau/(\pi\hbar)$", "")
    axes[0].set_ylabel("probability", fontsize=FONT_SIZE)
    figure.legend((swapped, approximation, charge),
                  ("spins swapped", "spin-only prediction", "both electrons on one site"),
                  loc="outside lower center", ncol=3, frameon=False, fontsize=FONT_SIZE)
    return figure


def draw_transfer(asymmetries: np.ndarray, transferred: core.Scalar, names: tuple[str, ...], repulsion_ratios: np.ndarray) -> plt.Figure:
    """How much charge the ground state moves to the lower site as the asymmetry grows, one curve per
    regime."""
    figure, ax = plt.subplots(figsize=(4.2, 2.3), layout="constrained")
    for curve, name, ratio, colour in zip(transferred.to_array(), names, repulsion_ratios, COLOURS):
        ax.plot(asymmetries, curve, color=colour, linewidth=LINE_WIDTH, label=f"{name}, $U/t = {ratio:g}$")
    # The charge moves over where the asymmetry matches each case's repulsion, ticked; from none to both electrons.
    ax.set(xlim=(asymmetries[0], asymmetries[-1]), ylim=(0, 2.05), xticks=repulsion_ratios, yticks=[0, 1, 2])
    plain_axes(ax, r"site asymmetry / hopping, $\Delta v/t$", r"occupation difference, $n_L - n_R$")
    ax.legend(frameon=False, loc="lower right", fontsize=FONT_SIZE)
    return figure


# The configurations that can ever be occupied from spin up on the left and spin down on the right.
DIALS = (0, 2, 3, 5)
# The side of each dial, the room for the row labels on the left and the column labels below, in inches, and the
# pixels per inch.
DIAL_INCHES, LABEL_INCHES, FOOT_INCHES, DIAL_DPI = 1.2, 0.7, 0.3, 80


def animate_dials(rows: list[str], cosine_amplitudes: core.Scalar, sine_amplitudes: core.Scalar,
                  triplet_cosine: core.Scalar, triplet_sine: core.Scalar) -> Iterator[np.ndarray]:
    """Each configuration's amplitude as an arrow on its own dial, its cosine part across and its sine
    part up, one labelled row per case, frame by frame. The arrow is drawn as the triplet's share,
    followed by the rest, the singlet block's, from its tip; the path the tip takes over the whole run
    is traced faintly."""
    total = np.stack([cosine_amplitudes.to_array(), sine_amplitudes.to_array()], axis=-1)[..., DIALS, :]   # [cases, times, dials, 2]
    triplet = np.stack([triplet_cosine.to_array(), triplet_sine.to_array()], axis=-1)[..., DIALS, :]       # [cases, times, dials, 2]
    count, columns = len(rows), len(DIALS)
    width, height = LABEL_INCHES + DIAL_INCHES * columns, DIAL_INCHES * count + FOOT_INCHES
    figure = plt.figure(figsize=(width, height), dpi=DIAL_DPI)
    turn = np.linspace(0, 2 * np.pi, 120)
    arrows = []
    for row, row_label in enumerate(rows):
        for column, label in enumerate(np.array(CONFIGURATION_LABELS)[list(DIALS)]):
            # Each dial in its own square, a small gap left between neighbours.
            ax = figure.add_axes(((LABEL_INCHES + DIAL_INCHES * column) / width, 1 - (row + 1) * DIAL_INCHES / height,
                                  0.95 * DIAL_INCHES / width, 0.95 * DIAL_INCHES / height))
            ax.plot(np.cos(turn), np.sin(turn), color="0.85", linewidth=0.5)
            ax.plot(*total[row, :, column].T, color=COLOURS[0], linewidth=0.4, alpha=0.3)
            arrows.append((ax.plot([], [], color="0.55", linewidth=1.3, animated=True)[0],
                           ax.plot([], [], color=COLOURS[0], linewidth=1.3, animated=True)[0],
                           ax.plot([], [], "o", color="0.1", markersize=2.5, animated=True)[0], row, column))
            ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[], yticks=[])
            ax.set_xlabel(label if row == count - 1 else "", fontsize=FONT_SIZE)
            ax.set_ylabel(row_label if column == 0 else "", fontsize=FONT_SIZE)
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_visible(False)
    # The dials, labels and traced paths are drawn once; each frame draws only the arrows over them.
    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(figure.bbox)
    for index in range(total.shape[1]):
        figure.canvas.restore_region(background)
        for triplet_line, rest_line, tip, row, column in arrows:
            start, end = triplet[row, index, column], total[row, index, column]
            triplet_line.set_data([0, start[0]], [0, start[1]])
            rest_line.set_data([start[0], end[0]], [start[1], end[1]])
            tip.set_data([end[0]], [end[1]])
            for artist in (triplet_line, rest_line, tip):
                artist.axes.draw_artist(artist)
        yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
