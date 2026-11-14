"""Energy levels, electron configurations, and spin exchange in the Hubbard dimer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util import charts
from examples.util.panels import overlay, panel
from numga import stack
from numga.render import drawing, planar, simplices
from numga.topology import Topology

if TYPE_CHECKING:
    from examples.quantum.hubbard_dimer import core

colours = ("#7d3c98", "#2e86c1", "#c0392b")
configuration_labels = (
    "L\u2191 L\u2193", "L\u2191 R\u2191",
    "L\u2191 R\u2193", "L\u2193 R\u2191",
    "L\u2193 R\u2193", "R\u2191 R\u2193",
)


# --- plumbing -------------------------------------------------------------------------
def styled_axes(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Labels a quantity plot's axes while keeping only the left and bottom spines."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)


@charts.style()
def spectrum_figure(repulsion: np.ndarray, hopping: float, energies: core.Scalar, exchange_gap: np.ndarray,
                    perturbative: np.ndarray, strong_repulsion: np.ndarray) -> plt.Figure:
    """Six two-electron energy levels appear beside the singlet-triplet gap across repulsion strength."""
    figure, (levels, gap) = charts.subplots(charts.DESIGN_WIDTH, 340, ncols=2, layout="constrained")
    ratios = repulsion / hopping
    energy_values = energies.to_array() / hopping                                                          # [..., 6]

    ground_line, = levels.plot(ratios, energy_values[:, 0], color=colours[0])
    triplet_lines = levels.plot(ratios, energy_values[:, 1:4], color=colours[1])
    charge_lines = levels.plot(ratios, energy_values[:, 4:], color=colours[2])
    # Zero energy marks the level where the three triplets remain independent of repulsion.
    levels.axhline(0, color="black", linewidth=charts.FRAME_WIDTH, zorder=0)
    levels.set(xlim=(ratios[0], ratios[-1]), xticks=[], yticks=[],
               xlabel="repulsion / hopping", ylabel="energy / hopping")
    levels.spines[["top", "right"]].set_visible(False)
    levels.legend((ground_line, triplet_lines[0], charge_lines[0]),
                  ("singlet ground state", "three triplets", "two excited singlets"), loc="upper left", frameon=False)

    gap.plot(ratios, exchange_gap / hopping, color=colours[0], label="from the singlet block")
    gap.plot(strong_repulsion / hopping, perturbative / hopping, "--", color="0.5", label="strong-repulsion limit")
    gap.set(xlim=(ratios[0], ratios[-1]), ylim=(0, exchange_gap[0] / hopping * 1.05), xticks=[], yticks=[],
            xlabel="repulsion / hopping", ylabel="singlet-triplet gap / hopping")
    gap.spines[["top", "right"]].set_visible(False)
    gap.legend(loc="upper right", frameon=False)
    return figure


@charts.style()
def ground_states_figure(names: tuple[str, ...], repulsion_ratios: np.ndarray, probabilities: core.Scalar) -> plt.Figure:
    """Ground-state configuration probabilities form a six-column heatmap with one row per regime."""
    figure, ax = charts.subplots(480, 210, layout="constrained")
    heatmap = ax.pcolormesh(probabilities.to_array(), cmap="Purples", vmin=0, vmax=0.5)
    ax.invert_yaxis()
    ax.set(xticks=np.arange(len(configuration_labels)) + 0.5, yticks=np.arange(len(names)) + 0.5)
    ax.set_xticklabels(configuration_labels)
    ax.set_yticklabels([f"{name}\nrepulsion {ratio:g}" for name, ratio in zip(names, repulsion_ratios)])
    ax.set_xlabel("electron configuration")
    ax.tick_params(length=0)
    bar = figure.colorbar(heatmap, ax=ax, shrink=0.85, pad=0.02, ticks=[0, 0.25, 0.5])
    bar.set_label("probability")
    return figure


@charts.style()
def exchange_figure(phase: np.ndarray, probabilities: core.Scalar, spin_only: np.ndarray) -> plt.Figure:
    """Spin swapping and temporary double occupancy trace one exchange cycle per repulsion regime."""
    values = probabilities.to_array()                                                                      # [..., ..., 6]
    n_cases = len(values)
    figure, axes = charts.subplots(charts.DESIGN_WIDTH, 260, ncols=n_cases, layout="constrained", sharey=True)
    for ax, case in zip(axes, values):
        swapped, = ax.plot(phase / np.pi, case[:, 3], color=colours[0])
        approximation, = ax.plot(phase / np.pi, spin_only, "--", color="0.5")
        charge, = ax.plot(phase / np.pi, case[:, 0] + case[:, 5], color=colours[2])
        ax.set(xlim=(phase[0] / np.pi, phase[-1] / np.pi), ylim=(0, 1.05), xticks=[], yticks=[0, 0.5, 1])
        styled_axes(ax, "exchange phase / half-turns", "")
    axes[0].set_ylabel("probability")
    figure.legend((swapped, approximation, charge),
                  ("spins swapped", "spin-only prediction", "both electrons on one site"),
                  loc="outside lower center", ncol=3, frameon=False)
    return figure


@charts.style()
def transfer_figure(asymmetries: np.ndarray, charge_differences: core.Scalar, names: tuple[str, ...], repulsion_ratios: np.ndarray) -> plt.Figure:
    """Ground-state occupation difference rises toward two electrons as site asymmetry outweighs repulsion."""
    figure, ax = charts.subplots(440, 280, layout="constrained")
    for curve, name, ratio, colour in zip(charge_differences.to_array(), names, repulsion_ratios, colours):
        ax.plot(asymmetries, curve, color=colour, label=f"{name}, repulsion {ratio:g}")
    ax.set(xlim=(asymmetries[0], asymmetries[-1]), ylim=(0, 2.05), xticks=repulsion_ratios, yticks=[0, 1, 2])
    styled_axes(ax, "site asymmetry / hopping", "occupation difference")
    ax.legend(frameon=False, loc="lower right")
    return figure


# The four configurations reachable from spin up on the left and spin down on the right.
shown_dials = (0, 2, 3, 5)
dial_pixels, label_pixels, foot_pixels = 100, 50, 30
n_rim_samples = 119
mv = planar.mv
axes = stack((mv.x, mv.y))                                                   # [2] Plane+
rim_ink, triplet_ink, tip_ink = np.full(3, 0.85), np.full(3, 0.55), np.full(3, 0.1)
trace_opacity = 0.3


def dials_animation(rows: list[str], configurations: core.Pair, zero_triplet: core.Pair,
                    states: Iterable[tuple[core.Pair, core.Pair]]) -> Iterator[np.ndarray]:
    """Yields RGB frames showing each configuration's cosine and sine amplitudes as a two-part arrow
    (triplet share plus remaining singlet-block share) on planar dials, one row per regime."""
    n_cases, n_columns = len(rows), len(shown_dials)
    span = 2.2 / 0.95
    across = (np.arange(n_columns) + 0.5) * span                             # [n_columns]
    up = (n_cases - np.arange(n_cases) - 0.5) * span                         # [n_cases]
    centres = planar.origin + mv.x.dual() * across + mv.y.dual() * up[:, None]  # [n_cases, n_columns] Point-
    rim_angles = np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False)
    turns = (mv.xy * (-rim_angles / 2)).exp()
    rim_offsets = turns >> mv.x.dual()                                       # [n_rim_samples] Point-
    rims = centres[..., None] + rim_offsets                                  # [n_cases, n_columns, n_rim_samples] Point-
    width, height = label_pixels + dial_pixels * n_columns, dial_pixels * n_cases + foot_pixels
    colour = np.array(to_rgb(colours[0]))
    n_dials = n_cases * n_columns
    dial_selection = (np.arange(n_cases)[:, None], np.array(shown_dials)[None, :])
    with charts.style():
        figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI))
        ax = figure.add_axes((label_pixels / width, foot_pixels / height, n_columns * dial_pixels / width,
                              n_cases * dial_pixels / height))
        ax.set(xlim=(0, n_columns * span), ylim=(0, n_cases * span), aspect="equal")
        camera = panel(ax, planar.mv.x, planar.mv.y, planar.mv.w)
        # Dial rims sit fixed in the background while the arrow and faint tip trail update each frame.
        faces = drawing.composited([
            simplices.strokes(rims.reshape(-1), Topology.loops([n_rim_samples] * n_dials), rim_ink,
                              charts.FRAME_PX, camera)])
        overlay(ax, [faces])
        arrow_overlay = overlay(ax, [np.zeros(camera.shape + (4,))])
        arrow_overlay.set_animated(True)
        for height_at, row_label in zip(up, rows):
            ax.annotate(row_label, (0, height_at), xytext=(-4, 0), textcoords="offset points", rotation=90,
                        ha="right", va="center", multialignment="center")
        for across_at, label in zip(across, np.array(configuration_labels)[list(shown_dials)]):
            ax.annotate(label, (across_at, 0), xytext=(0, -6), textcoords="offset points", ha="center", va="top")
        figure.canvas.draw()
        background = figure.canvas.copy_from_bbox(figure.bbox)
        arrows = Topology.chains([3] * n_dials)
        shares = np.tile(np.stack((triplet_ink, colour)), (n_dials, 1))
        tip_history = []
        for cosine_part, sine_part in states:
            figure.canvas.restore_region(background)
            triplet_cosine = zero_triplet * zero_triplet.reverse().scalar_product(cosine_part)   # [n_cases] Bivector+
            triplet_sine = zero_triplet * zero_triplet.reverse().scalar_product(sine_part)   # [n_cases] Bivector+
            cosine_amp = configurations.reverse().scalar_product(cosine_part[..., None])   # [n_cases, 6] Scalar+
            sine_amp = configurations.reverse().scalar_product(sine_part[..., None])   # [n_cases, 6] Scalar+
            triplet_cos_amp = configurations.reverse().scalar_product(triplet_cosine[..., None])   # [n_cases, 6] Scalar+
            triplet_sin_amp = configurations.reverse().scalar_product(triplet_sine[..., None])   # [n_cases, 6] Scalar+
            readings = stack((cosine_amp, sine_amp), axis=-1).to_array()     # [n_cases, 6, 2]
            triplet_readings = stack((triplet_cos_amp, triplet_sin_amp), axis=-1).to_array()   # [n_cases, 6, 2]
            tips = centres + (axes.dual() * readings[dial_selection]).sum(axis=-1)    # [n_cases, n_columns] Point-
            starts = centres + (axes.dual() * triplet_readings[dial_selection]).sum(axis=-1)   # [n_cases, n_columns] Point-
            tip_history.append(tips)
            layers = []
            if len(tip_history) >= 2:
                trail_points = stack(tip_history, axis=-1)                   # [n_cases, n_columns, ...] Point-
                layers.append(simplices.strokes(
                    trail_points.reshape(-1), Topology.chains([len(tip_history)] * n_dials),
                    colour, charts.FRAME_PX, camera, opacity=trace_opacity))
            shafts = stack((centres, starts, tips), axis=-1)                 # [n_cases, n_columns, 3] Point-
            layers.append(simplices.strokes(shafts.reshape(-1), arrows, shares, charts.LINE_PX, camera))
            layers.append(simplices.markers(tips.reshape(-1), tip_ink, charts.MARKER_PX / 2, camera))
            arrow_overlay.set_data(drawing.composited(layers))
            ax.draw_artist(arrow_overlay)
            yield np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
        plt.close(figure)
