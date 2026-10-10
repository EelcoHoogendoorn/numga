"""Energy levels, electron configurations, and spin exchange in the Hubbard dimer."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_dimer import core

COLOURS = ("#7d3c98", "#2e86c1", "#c0392b")
CONFIGURATION_LABELS = (
    r"$L\uparrow\,L\downarrow$", r"$L\uparrow\,R\uparrow$",
    r"$L\uparrow\,R\downarrow$", r"$L\downarrow\,R\uparrow$",
    r"$L\downarrow\,R\downarrow$", r"$R\uparrow\,R\downarrow$",
)


# --- plumbing -------------------------------------------------------------------------
def draw_spectrum(repulsion: np.ndarray, hopping: float, energies: core.Scalar, exchange: np.ndarray,
                  perturbative: np.ndarray, strong_repulsion: np.ndarray) -> plt.Figure:
    """The six two-electron levels, and the energy needed to turn the singlet into a triplet."""
    figure, (levels, gap) = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    ratios = repulsion / hopping
    energy_values = energies.to_array() / hopping                              # [samples, modes]

    ground_line, = levels.plot(ratios, energy_values[:, 0], color=COLOURS[0])
    triplet_lines = levels.plot(ratios, energy_values[:, 1:4], color=COLOURS[1])
    charge_lines = levels.plot(ratios, energy_values[:, 4:], color=COLOURS[2])
    levels.set(xlabel=r"repulsion / hopping, $U/t$", ylabel=r"energy / hopping, $E/t$", xlim=(ratios[0], ratios[-1]))
    levels.legend((ground_line, triplet_lines[0], charge_lines[0]),
                  ("singlet ground state", "three triplets", "two excited singlets"), loc="upper left", frameon=False)

    gap.plot(ratios, exchange / hopping, color=COLOURS[0], label="from the singlet block")
    gap.plot(strong_repulsion / hopping, perturbative / hopping, "--", color="0.5", label=r"$4t^2/U$")
    gap.set(xlabel=r"repulsion / hopping, $U/t$", ylabel=r"singlet-triplet gap / hopping, $J/t$",
            xlim=(ratios[0], ratios[-1]), ylim=(0, exchange[0] / hopping * 1.05))
    gap.legend(loc="upper right", frameon=False)
    return figure


def draw_ground_states(names: tuple[str, ...], repulsion_ratios: np.ndarray, probabilities: core.Scalar) -> plt.Figure:
    """How likely each electron configuration is in the ground state, one row per case."""
    figure, ax = plt.subplots(figsize=(8.5, 3.4), layout="constrained")
    heatmap = ax.imshow(probabilities.to_array(), cmap="Purples", vmin=0, vmax=0.5, aspect="auto")
    ax.set(xticks=np.arange(len(CONFIGURATION_LABELS)), xticklabels=CONFIGURATION_LABELS,
           yticks=np.arange(len(names)),
           yticklabels=[f"{name}\n$U/t = {ratio:g}$" for name, ratio in zip(names, repulsion_ratios)],
           xlabel="electron configuration")
    ax.tick_params(length=0)
    figure.colorbar(heatmap, ax=ax, label="probability", shrink=0.85, pad=0.02)
    return figure


def draw_exchange(phase: np.ndarray, probabilities: core.Scalar, spin_only: np.ndarray) -> plt.Figure:
    """Spin swapping and temporary double occupancy for each case, left to right, on each case's own
    exchange timescale."""
    values = probabilities.to_array()                                         # [cases, times, configurations]
    figure, axes = plt.subplots(1, len(values), figsize=(4.1 * len(values), 3.9), layout="constrained", sharey=True)
    for ax, case in zip(axes, values):
        swapped, = ax.plot(phase / np.pi, case[:, 3], color=COLOURS[0])
        approximation, = ax.plot(phase / np.pi, spin_only, "--", color="0.5")
        charge, = ax.plot(phase / np.pi, case[:, 0] + case[:, 5], color=COLOURS[2])
        ax.set(xlabel=r"exchange phase, $J\tau/(\pi\hbar)$", xlim=(phase[0] / np.pi, phase[-1] / np.pi), ylim=(-0.02, 1.05))
    axes[0].set_ylabel("probability")
    figure.legend((swapped, approximation, charge),
                  ("spins swapped", "spin-only prediction", "both electrons on one site"),
                  loc="outside lower center", ncol=3, frameon=False)
    return figure


def draw_transfer(asymmetries: np.ndarray, transferred: core.Scalar, names: tuple[str, ...], repulsion_ratios: np.ndarray) -> plt.Figure:
    """How much charge the ground state moves to the lower site as the asymmetry grows, one curve per
    regime."""
    figure, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    for curve, name, ratio, colour in zip(transferred.to_array(), names, repulsion_ratios, COLOURS):
        ax.plot(asymmetries, curve, color=colour, linewidth=1.5, label=f"{name}, $U/t = {ratio:g}$")
    ax.set(xlim=(asymmetries[0], asymmetries[-1]), ylim=(-0.05, 2.05), xlabel=r"site asymmetry / hopping, $\Delta v/t$",
           ylabel=r"occupation difference, $n_L - n_R$")
    ax.legend(frameon=False, loc="lower right")
    return figure


# The configurations that can ever be occupied from spin up on the left and spin down on the right.
DIALS = (0, 2, 3, 5)


def animate_dials(rows: list[str], cosine_amplitudes: core.Scalar, sine_amplitudes: core.Scalar,
                  triplet_cosine: core.Scalar, triplet_sine: core.Scalar) -> list[np.ndarray]:
    """Each configuration's amplitude as an arrow on its own dial, its cosine part across and its sine
    part up, one labelled row per case, frame by frame. The arrow is drawn as the triplet's share,
    followed by the rest, the singlet block's, from its tip; the path the tip takes over the whole run
    is traced faintly."""
    total = np.stack([cosine_amplitudes.to_array(), sine_amplitudes.to_array()], axis=-1)[..., DIALS, :]   # [cases, times, dials, 2]
    triplet = np.stack([triplet_cosine.to_array(), triplet_sine.to_array()], axis=-1)[..., DIALS, :]       # [cases, times, dials, 2]
    count, columns = len(rows), len(DIALS)
    width, height = 1.9 * columns + 1.4, 1.9 * count + 0.4
    figure = plt.figure(figsize=(width, height), dpi=80)
    turn = np.linspace(0, 2 * np.pi, 120)
    arrows = []
    for row, row_label in enumerate(rows):
        for column, label in enumerate(np.array(CONFIGURATION_LABELS)[list(DIALS)]):
            ax = figure.add_axes(((1.4 + 1.9 * column) / width, 1 - (row + 1) * 1.9 / height, 1.8 / width, 1.8 / height))
            ax.plot(np.cos(turn), np.sin(turn), color="0.85", linewidth=0.8)
            ax.plot(*total[row, :, column].T, color=COLOURS[0], linewidth=0.6, alpha=0.3)
            arrows.append((ax.plot([], [], color="0.55", linewidth=2.0, animated=True)[0],
                           ax.plot([], [], color=COLOURS[0], linewidth=2.0, animated=True)[0],
                           ax.plot([], [], "o", color="0.1", markersize=4, animated=True)[0], row, column))
            ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[], yticks=[])
            ax.set_xlabel(label if row == count - 1 else "", fontsize=9)
            ax.set_ylabel(row_label if column == 0 else "", fontsize=9)
            ax.set_aspect("equal")
            for side in ax.spines.values():
                side.set_visible(False)
    # The dials, labels and traced paths are drawn once; each frame draws only the arrows over them.
    figure.canvas.draw()
    background = figure.canvas.copy_from_bbox(figure.bbox)
    frames = []
    for index in range(total.shape[1]):
        figure.canvas.restore_region(background)
        for triplet_line, rest_line, tip, row, column in arrows:
            start, end = triplet[row, index, column], total[row, index, column]
            triplet_line.set_data([0, start[0]], [0, start[1]])
            rest_line.set_data([start[0], end[0]], [start[1], end[1]])
            tip.set_data([end[0]], [end[1]])
            for artist in (triplet_line, rest_line, tip):
                artist.axes.draw_artist(artist)
        frames.append(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
    plt.close(figure)
    return frames
