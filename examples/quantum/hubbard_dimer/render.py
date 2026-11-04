"""Energy levels, electron configurations, and spin exchange in the Hubbard dimer."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.quantum.hubbard_dimer import core
from examples.util import charts
from examples.util.wgsl import planar, simplices
from examples.util.wgsl.camera import panel
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
n_rim_samples = 119
mv = planar.mv
axes = stack((mv.x, mv.y))                                                   # [2] Plane+
# The rim's and the triplet share's inks, the tip's, and how much the traced path covers.
RIM, TRIPLET, TIP = np.full(3, 0.85), np.full(3, 0.55), np.full(3, 0.1)
TRACE_OPACITY = 0.3


def animate_dials(rows: list[str], cosine_amplitudes: core.Scalar, sine_amplitudes: core.Scalar,
                  triplet_cosine: core.Scalar, triplet_sine: core.Scalar) -> Iterator[np.ndarray]:
    """Each configuration's amplitude as an arrow on its own dial, its cosine part across and its sine
    part up, one labelled row per case, frame by frame. The arrow is drawn as the triplet's share,
    followed by the rest, the singlet block's, from its tip; the path the tip takes over the whole run
    is traced faintly."""
    count, columns = len(rows), len(DIALS)
    # The dials laid out in one panel, a dial's span apart: a dial of radius one fills most of its square. The columns
    # across and the rows up, the first row on top.
    span = 2.2 / 0.95
    across, up = (np.arange(columns) + 0.5) * span, (count - np.arange(count) - 0.5) * span   # [dials], [cases]
    centres = planar.origin + mv.x.dual() * across + mv.y.dual() * up[:, None]  # [cases, dials] Point-
    # Each configuration's amplitude as a move on its dial, its cosine part across and its sine part up: numbers of
    # the chart, read out of the dimer's algebra.
    readings = stack((cosine_amplitudes, sine_amplitudes), axis=-1).to_array()   # [cases, times, configurations, 2]
    triplet_readings = stack((triplet_cosine, triplet_sine), axis=-1).to_array() # [cases, times, configurations, 2]
    n_times = readings.shape[1]
    # The amplitudes of the configurations shown, a dial's in a row, its times along it.
    shown = (np.arange(count)[:, None, None], np.arange(n_times)[None, None, :], np.array(DIALS)[None, :, None])
    ends = centres[..., None] + (axes.dual() * readings[shown]).sum(axis=-1)   # [cases, dials, times] Point-
    starts = centres[..., None] + (axes.dual() * triplet_readings[shown]).sum(axis=-1)  # [cases, dials, times] Point-
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False) / 2)).exp()
    rims = centres[..., None] + (turns >> mv.x.dual())                         # [cases, dials, rim samples] Point-
    width, height = label_pixels + dial_pixels * columns, dial_pixels * count + foot_pixels
    colour = np.array(to_rgb(COLOURS[0]))
    n_dials = count * columns
    with charts.style():
        figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI))
        ax = figure.add_axes((label_pixels / width, foot_pixels / height, columns * dial_pixels / width,
                              count * dial_pixels / height))
        ax.set(xlim=(0, columns * span), ylim=(0, count * span), aspect="equal")
        camera = panel(ax, planar.mv.x, planar.mv.y, planar.mv.w)
        # The dials' rims and the traced paths faint, as thin as the frames.
        faces = simplices.composited([
            simplices.strokes(rims.reshape(-1), simplices.Topology.loops([n_rim_samples] * n_dials), RIM,
                              charts.FRAME_PX, camera),
            simplices.strokes(ends.reshape(-1), simplices.Topology.chains([n_times] * n_dials), colour, charts.FRAME_PX,
                              camera, opacity=TRACE_OPACITY)])
        simplices.overlay(ax, [faces])
        # The arrows and their tips over the faces, drawn anew each frame.
        arrows_shown = simplices.overlay(ax, [np.zeros(camera.shape + (4,))])
        arrows_shown.set_animated(True)
        # Each case's name left of its row, each configuration's under its column.
        for height_at, row_label in zip(up, rows):
            ax.annotate(row_label, (0, height_at), xytext=(-4, 0), textcoords="offset points", rotation=90,
                        ha="right", va="center", multialignment="center")
        for across_at, label in zip(across, np.array(CONFIGURATION_LABELS)[list(DIALS)]):
            ax.annotate(label, (across_at, 0), xytext=(0, -6), textcoords="offset points", ha="center", va="top")
        # The labels are drawn once; each frame draws only the dials over them, the arrows on their faces.
        figure.canvas.draw()
        background = figure.canvas.copy_from_bbox(figure.bbox)
        arrows = simplices.Topology.chains([3] * n_dials)
        shares = np.tile(np.stack((TRIPLET, colour)), (n_dials, 1))
        for index in range(n_times):
            figure.canvas.restore_region(background)
            tips = ends[..., index]                                             # [cases, dials] Point-
            shafts = stack((centres, starts[..., index], tips), axis=-1)       # [cases, dials, 3] Point-
            arrows_shown.set_data(simplices.composited([
                simplices.strokes(shafts.reshape(-1), arrows, shares, charts.LINE_PX, camera),
                simplices.markers(tips.reshape(-1), TIP, charts.MARKER_PX / 2, camera)]))
            ax.draw_artist(arrows_shown)
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
