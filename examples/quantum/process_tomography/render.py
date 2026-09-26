"""Probe probabilities, reconstructed channels and predictions drawn from quantum states."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from examples.animation import capture
from examples.quantum.process_tomography import core

CHANNEL_COLOURS = ("#3275a8", "#d16a30", "#7b62a3")
PROBE_COLOURS = ("#3275a8", "#d16a30", "#438e65", "#7b62a3")
PROBE_NAMES = ("A", "B", "C", "D")


# --- plumbing -------------------------------------------------------------------------
def bloch(states: core.State) -> np.ndarray:
    """The Bloch vectors of normalized states, in an explicit spatial blade layout."""
    return 2 * states.cast(core.ga.subspace("x y z")).kernel                  # [..., 3]


def bloch_axes(axes: Axes3D, title: str) -> None:
    """One camera and scale shared by all channel views."""
    axes.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1),
             xlabel="x", ylabel="y", zlabel="z", title=title)
    axes.set_xticks([-1, 0, 1])
    axes.set_yticks([-1, 0, 1])
    axes.set_zticks([-1, 0, 1])
    axes.set_box_aspect((1, 1, 1), zoom=0.85)
    axes.view_init(elev=24, azim=-54)
    axes.grid(False)
    axes.xaxis.pane.fill = False
    axes.yaxis.pane.fill = False
    axes.zaxis.pane.fill = False


def draw_experiment(prepared: core.State, probabilities: core.Scalar,
                    labels: tuple[str, ...]) -> plt.Figure:
    """Four tetrahedral preparations and each channel's measured outcome probabilities."""
    probes = bloch(prepared)                                                   # [preparations, 3]
    measured = probabilities.to_array()                                      # [channels, preparations, outcomes]
    figure = plt.figure(figsize=(13, 3.7), layout="constrained")
    figure.get_layout_engine().set(wspace=0.12)
    probe_axes = figure.add_subplot(1, len(labels) + 1, 1, projection="3d")
    # Joining the four probes makes their coverage of space visible.
    first, second = np.triu_indices(len(probes), k=1)
    edges = np.stack((probes[first], probes[second]), axis=1)                  # [edges, 2, 3]
    probe_axes.add_collection3d(Line3DCollection(edges, colors="0.75", linewidths=1))
    probe_axes.scatter(*probes.T, c=PROBE_COLOURS, s=40, depthshade=False)
    for point, name, colour in zip(probes, PROBE_NAMES, PROBE_COLOURS):
        probe_axes.text(*(point * 1.12), name, color=colour, fontsize=10)
    bloch_axes(probe_axes, "Prepared states")

    table_axes = []
    probability_scale = LogNorm(vmin=measured[measured > 0].min(), vmax=0.5)
    for index, (label, table) in enumerate(zip(labels, measured)):
        axes = figure.add_subplot(1, len(labels) + 1, index + 2)
        table_axes.append(axes)
        heatmap = axes.imshow(table, origin="upper", norm=probability_scale, cmap="turbo")
        axes.set(xticks=np.arange(len(PROBE_NAMES)), xticklabels=PROBE_NAMES,
                 yticks=np.arange(len(PROBE_NAMES)), yticklabels=PROBE_NAMES,
                 xlabel="Measurement outcome", ylabel="Prepared state", title=label)
        axes.tick_params(length=0)
    figure.colorbar(heatmap, ax=table_axes, label="Probability (log scale)", shrink=0.72, pad=0.02)
    return figure


def draw_channels(surface: core.State, outputs: core.State, prepared: core.State,
                  transformed: core.State, labels: tuple[str, ...]) -> plt.Figure:
    """The sphere of pure inputs and its reconstructed images, with matching probe points."""
    sphere = bloch(surface)                                                    # [latitudes + 1, longitudes + 1, 3]
    images = bloch(outputs)                                                    # [channels, latitudes + 1, longitudes + 1, 3]
    probes = bloch(prepared)                                                   # [preparations, 3]
    mapped = bloch(transformed)                                                # [channels, preparations, 3]
    figure = plt.figure(figsize=(4.1 * len(labels), 4.3), layout="constrained")
    for index, (label, image, points, colour) in enumerate(zip(labels, images, mapped, CHANNEL_COLOURS)):
        axes = figure.add_subplot(1, len(labels), index + 1, projection="3d")
        # The reference mesh and the channel's image use the same sampled directions.
        axes.plot_wireframe(*np.moveaxis(sphere, -1, 0), rcount=7, ccount=9,
                            color="0.75", linewidth=0.5, alpha=0.45)
        axes.plot_surface(*np.moveaxis(image, -1, 0), color=colour,
                          alpha=0.12, linewidth=0, shade=False)
        axes.plot_wireframe(*np.moveaxis(image, -1, 0), rcount=7, ccount=9,
                            color=colour, linewidth=0.75, alpha=0.8)
        # Hollow and filled markers identify corresponding preparations and outputs.
        axes.scatter(*probes.T, edgecolors=PROBE_COLOURS, facecolors="none",
                     s=55, linewidths=1.3, depthshade=False)
        axes.scatter(*points.T, c=PROBE_COLOURS, s=30, depthshade=False)
        bloch_axes(axes, label)
    markers = (
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="none",
               markeredgecolor="0.4", label="Prepared"),
        Line2D([], [], linestyle="none", marker="o", color="0.4", label="After channel"),
    )
    figure.legend(handles=markers, loc="outside lower center", ncol=2, frameon=False)
    return figure


def draw_predictions(observed: core.Scalar, predicted: core.Scalar,
                     labels: tuple[str, ...]) -> plt.Figure:
    """Predicted and measured probabilities for preparations absent from the reconstruction."""
    measured = observed.to_array()                                             # [channels, heldout, outcomes]
    inferred = predicted.to_array()                                            # [channels, heldout, outcomes]
    limit = 1.05 * max(float(measured.max()), float(inferred.max()))
    figure, axes = plt.subplots(figsize=(4.7, 4.5), layout="constrained")
    axes.plot([0, limit], [0, limit], color="0.7", linewidth=1, zorder=0)
    for label, actual, estimate, colour in zip(labels, measured, inferred, CHANNEL_COLOURS):
        axes.scatter(actual, estimate, s=22, color=colour, alpha=0.75,
                     linewidths=0, label=label)
    axes.set(xlim=(0, limit), ylim=(0, limit), aspect="equal",
             xlabel="Observed probability", ylabel="Predicted probability",
             title="Unseen preparations")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper left", frameon=False, fontsize=9)
    return figure


def print_completeness(spectra: core.Scalar, labels: tuple[str, ...]) -> None:
    """Print each probe set's singular values, including unobserved directions."""
    for label, spectrum in zip(labels, spectra.to_array()):
        print(f"{label}: {np.array2string(spectrum, precision=4, suppress_small=True)}")


def animate(surface: core.State, prepared: core.State,
            frames: Iterator[tuple[core.State, core.State]],
            labels: tuple[str, ...]) -> list[np.ndarray]:
    """Successive applications of each reconstructed channel to the same input sphere."""
    images = []
    for output, transformed in frames:
        figure = draw_channels(surface, output, prepared, transformed, labels)
        images.append(capture(figure))
        plt.close(figure)
    return images
