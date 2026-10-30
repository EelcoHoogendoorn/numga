"""Energy levels, electron configurations, and spin exchange in the Hubbard dimer."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.quantum.hubbard_dimer import core
from examples.util import charts
from examples.util.wgsl import planar
from numga import stack

COLOURS = ("#7d3c98", "#2e86c1", "#c0392b")
CONFIGURATION_LABELS = (
    r"$L\uparrow\,L\downarrow$", r"$L\uparrow\,R\uparrow$",
    r"$L\uparrow\,R\downarrow$", r"$L\downarrow\,R\uparrow$",
    r"$L\downarrow\,R\downarrow$", r"$R\uparrow\,R\downarrow$",
)


# --- plumbing -------------------------------------------------------------------------
def plain_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Label a quantity plot's axes, and keep only the left and bottom spines."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)


@charts.style()
def draw_spectrum(repulsion: np.ndarray, hopping: float, energies: core.Scalar, exchange: np.ndarray,
                  perturbative: np.ndarray, strong_repulsion: np.ndarray) -> plt.Figure:
    """The six two-electron levels, and the energy needed to turn the singlet into a triplet."""
    # The whole column: two panels side by side, each with a long axis label and a legend of long names.
    figure, (levels, gap) = charts.subplots(charts.DESIGN_WIDTH, 340, ncols=2, layout="constrained")
    ratios = repulsion / hopping
    energy_values = energies.to_array() / hopping                                                          # [samples, modes]

    ground_line, = levels.plot(ratios, energy_values[:, 0], color=COLOURS[0])
    triplet_lines = levels.plot(ratios, energy_values[:, 1:4], color=COLOURS[1])
    charge_lines = levels.plot(ratios, energy_values[:, 4:], color=COLOURS[2])
    # The energy's zero, where the triplets stay, drawn as thin as the axes.
    levels.axhline(0, color="black", linewidth=charts.FRAME_WIDTH, zorder=0)
    levels.set(xlim=(ratios[0], ratios[-1]), xticks=[], yticks=[],
               xlabel=r"repulsion / hopping, $U/t$", ylabel=r"energy / hopping, $E/t$")
    levels.spines[["top", "right"]].set_visible(False)
    levels.legend((ground_line, triplet_lines[0], charge_lines[0]),
                  ("singlet ground state", "three triplets", "two excited singlets"), loc="upper left", frameon=False)

    gap.plot(ratios, exchange / hopping, color=COLOURS[0], label="from the singlet block")
    gap.plot(strong_repulsion / hopping, perturbative / hopping, "--", color="0.5", label=r"$4t^2/U$")
    gap.set(xlim=(ratios[0], ratios[-1]), ylim=(0, exchange[0] / hopping * 1.05), xticks=[], yticks=[],
            xlabel=r"repulsion / hopping, $U/t$", ylabel=r"singlet-triplet gap / hopping, $J/t$")
    gap.spines[["top", "right"]].set_visible(False)
    gap.legend(loc="upper right", frameon=False)
    return figure


@charts.style()
def draw_ground_states(names: tuple[str, ...], repulsion_ratios: np.ndarray, probabilities: core.Scalar) -> plt.Figure:
    """How likely each electron configuration is in the ground state, one row per case."""
    # Room for the six configurations' names along the bottom, and for the cases' two-line names and the bar.
    figure, ax = charts.subplots(480, 210, layout="constrained")
    # Each cell a rectangle of its own, the first case on top.
    heatmap = ax.pcolormesh(probabilities.to_array(), cmap="Purples", vmin=0, vmax=0.5)
    ax.invert_yaxis()
    ax.set(xticks=np.arange(len(CONFIGURATION_LABELS)) + 0.5, yticks=np.arange(len(names)) + 0.5)
    ax.set_xticklabels(CONFIGURATION_LABELS)
    ax.set_yticklabels([f"{name}\n$U/t = {ratio:g}$" for name, ratio in zip(names, repulsion_ratios)])
    ax.set_xlabel("electron configuration")
    ax.tick_params(length=0)
    bar = figure.colorbar(heatmap, ax=ax, shrink=0.85, pad=0.02, ticks=[0, 0.25, 0.5])
    bar.set_label("probability")
    return figure


@charts.style()
def draw_exchange(phase: np.ndarray, probabilities: core.Scalar, spin_only: np.ndarray) -> plt.Figure:
    """Spin swapping and temporary double occupancy for each case, left to right, on each case's own
    exchange timescale."""
    values = probabilities.to_array()                                                                      # [cases, times, configurations]
    # The whole column for the cases side by side, over a legend of three long names in one row.
    figure, axes = charts.subplots(charts.DESIGN_WIDTH, 260, ncols=len(values), layout="constrained", sharey=True)
    for ax, case in zip(axes, values):
        swapped, = ax.plot(phase / np.pi, case[:, 3], color=COLOURS[0])
        approximation, = ax.plot(phase / np.pi, spin_only, "--", color="0.5")
        charge, = ax.plot(phase / np.pi, case[:, 0] + case[:, 5], color=COLOURS[2])
        # Probabilities read off at none, a half and certainty, along one cycle of the exchange.
        ax.set(xlim=(phase[0] / np.pi, phase[-1] / np.pi), ylim=(0, 1.05), xticks=[], yticks=[0, 0.5, 1])
        plain_axes(ax, r"exchange phase, $J\tau/(\pi\hbar)$", "")
    axes[0].set_ylabel("probability")
    figure.legend((swapped, approximation, charge),
                  ("spins swapped", "spin-only prediction", "both electrons on one site"),
                  loc="outside lower center", ncol=3, frameon=False)
    return figure


@charts.style()
def draw_transfer(asymmetries: np.ndarray, transferred: core.Scalar, names: tuple[str, ...], repulsion_ratios: np.ndarray) -> plt.Figure:
    """How much charge the ground state moves to the lower site as the asymmetry grows, one curve per
    regime."""
    # Three curves and their legend, and the height the long vertical label needs.
    figure, ax = charts.subplots(440, 280, layout="constrained")
    for curve, name, ratio, colour in zip(transferred.to_array(), names, repulsion_ratios, COLOURS):
        ax.plot(asymmetries, curve, color=colour, label=f"{name}, $U/t = {ratio:g}$")
    # The charge moves over where the asymmetry matches each case's repulsion, ticked; from none to both electrons.
    ax.set(xlim=(asymmetries[0], asymmetries[-1]), ylim=(0, 2.05), xticks=repulsion_ratios, yticks=[0, 1, 2])
    plain_axes(ax, r"site asymmetry / hopping, $\Delta v/t$", r"occupation difference, $n_L - n_R$")
    ax.legend(frameon=False, loc="lower right")
    return figure


# The configurations that can ever be occupied from spin up on the left and spin down on the right.
DIALS = (0, 2, 3, 5)
# The side of each dial, and the room for the row labels on the left and the column labels below, in pixels.
dial_pixels, label_pixels, foot_pixels = 100, 50, 30
n_rim_samples = 120
mv = planar.mv
axes = stack((mv.x, mv.y))                                                   # [2] Plane+


def animate_dials(rows: list[str], cosine_amplitudes: core.Scalar, sine_amplitudes: core.Scalar,
                  triplet_cosine: core.Scalar, triplet_sine: core.Scalar) -> Iterator[np.ndarray]:
    """Each configuration's amplitude as an arrow on its own dial, its cosine part across and its sine
    part up, one labelled row per case, frame by frame. The arrow is drawn as the triplet's share,
    followed by the rest, the singlet block's, from its tip; the path the tip takes over the whole run
    is traced faintly."""
    readings = planar.context.include_from_algebra(stack((cosine_amplitudes, sine_amplitudes), axis=-1))
    ends = planar.origin + (axes.dual() * readings).sum(axis=-1)              # [cases, times, configurations] Point-
    triplet_readings = planar.context.include_from_algebra(stack((triplet_cosine, triplet_sine), axis=-1))
    starts = planar.origin + (axes.dual() * triplet_readings).sum(axis=-1)     # [cases, times, configurations] Point-
    total = (axes & ends[..., DIALS, None]).to_array()                         # [cases, times, dials, 2]
    triplet = (axes & starts[..., DIALS, None]).to_array()
    count, columns = len(rows), len(DIALS)
    width, height = label_pixels + dial_pixels * columns, dial_pixels * count + foot_pixels
    with charts.style():
        figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI))
        turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples) / 2)).exp()
        rim = planar.origin + (turns >> mv.x.dual())
        outline = (axes & rim[..., None]).to_array()
        arrows = []
        for row, row_label in enumerate(rows):
            for column, label in enumerate(np.array(CONFIGURATION_LABELS)[list(DIALS)]):
                # Each dial in its own square, a small gap left between neighbours.
                ax = figure.add_axes(((label_pixels + dial_pixels * column) / width, 1 - (row + 1) * dial_pixels / height,
                                      0.95 * dial_pixels / width, 0.95 * dial_pixels / height))
                # The dial's rim and the traced path faint, as thin as the frames.
                ax.plot(*outline.T, color="0.85", linewidth=charts.FRAME_WIDTH)
                ax.plot(*total[row, :, column].T, color=COLOURS[0], linewidth=charts.FRAME_WIDTH, alpha=0.3)
                arrows.append((ax.plot([], [], color="0.55", animated=True)[0],
                               ax.plot([], [], color=COLOURS[0], animated=True)[0],
                               ax.plot([], [], "o", color="0.1", animated=True)[0], row, column))
                ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), xticks=[], yticks=[])
                ax.set_xlabel(label if row == count - 1 else "")
                ax.set_ylabel(row_label if column == 0 else "")
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
