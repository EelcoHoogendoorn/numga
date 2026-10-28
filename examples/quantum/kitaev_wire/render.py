"""Spatial Majorana mode weights, excitation energies and transport leakage."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.figure import Figure

from examples.util.animation import capture
from examples.util.wgsl import pinned
from examples.quantum.kitaev_wire import core

INK = "#263b51"
SLOW = "#148d88"
FAST = "#d06c36"
GATE = LinearSegmentedColormap.from_list("gate", ["white", "#e8e4f5"])
WEIGHTS = plt.get_cmap("magma")
# The mode weight at the top of the transport panels: above the most any one site holds, as the fast move squeezes
# the mode against the moving end.
WEIGHT_LIMIT = 0.6
# The size of the text and the width of the curves in the plots.
FONT_SIZE, LINE_WIDTH = 7, 1.1


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes) -> None:
    """A quantity plot's labels and ticks in the plots' text size, with faint left and bottom spines only."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.tick_params(color="0.8", labelsize=FONT_SIZE)
    axis.xaxis.label.set_fontsize(FONT_SIZE)
    axis.yaxis.label.set_fontsize(FONT_SIZE)


def shade(chemical: np.ndarray, hopping: float) -> np.ndarray:
    """How deep each site lies in the topological regime, from zero at the threshold 2 * hopping
    to one at the lowest chemical potential."""
    return np.clip((2 * hopping - chemical) / (2 * hopping - chemical.min()), 0, 1)


def weight_image(axis: plt.Axes, weights: np.ndarray, extent: tuple[float, float, float, float], gamma: float):
    """Mode weights, rows down from the top, one pixel per site and sample, held pixel for pixel with the axes'
    labels kept; the image."""
    colours = WEIGHTS(PowerNorm(gamma, vmin=0, vmax=weights.max())(weights))[..., :3]
    image = pinned(axis, colours, extent)
    axis.set_axis_on()
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0, labelsize=FONT_SIZE)
    return image


def draw_formation(chemical_potentials: np.ndarray, energies: core.Scalar,
                   density: core.Scalar, hopping: float) -> Figure:
    values = energies.to_array() / hopping
    weights = density.to_array()
    ratios = chemical_potentials / hopping
    figure, (spectrum, modes) = plt.subplots(1, 2, figsize=(3.6, 1.75), width_ratios=[1.6, 1], layout="constrained")
    # Each excitation has two real quadratures with the same frequency.
    spectrum.plot(ratios, values[:, ::2], color=INK, linewidth=0.7)
    spectrum.axvline(2, color="0.65", linestyle=":", linewidth=0.7)
    # Read off: the threshold at twice the hopping, and zero energy.
    spectrum.set(xlim=(ratios[0], ratios[-1]), ylim=(0, None), xticks=[2], yticks=[0],
                 xlabel=r"chemical potential inside, $\mu/t$", ylabel=r"excitation energy, $E/t$")
    plain_axes(spectrum)
    # One row per chemical potential, the outside value on top, half a sample beyond the first and last.
    half_row = (ratios[0] - ratios[-1]) / (len(ratios) - 1) / 2
    weight_image(modes, weights, (-0.5, weights.shape[-1] - 0.5, ratios[-1] - half_row, ratios[0] + half_row), 0.55)
    modes.axhline(2, color="white", linestyle=":", linewidth=0.6, alpha=0.6)
    modes.set(xticks=[], yticks=[2])
    modes.set_xlabel("site", fontsize=FONT_SIZE)
    modes.set_ylabel(r"$\mu/t$", fontsize=FONT_SIZE)
    return figure


def animate_transport(
    progress: np.ndarray, potentials: core.Scalar, history: core.Majorana,
    target: core.Majorana, share: core.Scalar, hopping: float,
) -> Iterator[np.ndarray]:
    weights = history.scalar_norm_squared().to_array()
    expected = target.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    leakage = 1 - share.to_array()
    sites = np.arange(weights.shape[-1])
    regime = shade(chemical, hopping)

    figure, axes = plt.subplots(3, 1, figsize=(3.6, 3.3), dpi=90,
                                gridspec_kw={"height_ratios": [1, 1, 0.7]})
    figure.subplots_adjust(left=0.1, right=0.97, top=0.98, bottom=0.09, hspace=0.6)
    lines, references, gates, fills = [], [], [], []
    for case, (axis, colour) in enumerate(zip(axes[:2], (SLOW, FAST))):
        # A two-row mesh shaded between its vertices fills the panel with the gate profile.
        gates.append(axis.pcolormesh(sites, [0, WEIGHT_LIMIT], np.broadcast_to(regime[0], (2, len(sites))),
                                     shading="gouraud", cmap=GATE, vmin=0, vmax=1, zorder=0))
        line, = axis.plot(sites, weights[0, case], color=colour, linewidth=LINE_WIDTH)
        reference, = axis.plot(sites, expected[0], color=INK, linestyle="--", linewidth=0.6)
        fills.append(axis.fill_between(sites, weights[0, case], color=colour, alpha=0.25, linewidth=0))
        lines.append(line)
        references.append(reference)
        axis.set(xlim=(sites[0], sites[-1]), ylim=(0, WEIGHT_LIMIT), xticks=[], yticks=[], xlabel="site", ylabel="mode weight")
        plain_axes(axis)
    traces = [axes[2].plot([], [], color=colour, linewidth=LINE_WIDTH)[0] for colour in (SLOW, FAST)]
    cursor = axes[2].axvline(0, color="0.7", linewidth=0.6)
    # The bulk weight from none to all of the mode, along the gate's motion.
    axes[2].set(xlim=(0, 1), ylim=(0, 1.02), xticks=[], yticks=[0, 1], xlabel="fraction of gate motion", ylabel="bulk weight")
    plain_axes(axes[2])

    for frame, fraction in enumerate(progress):
        for case, colour in enumerate((SLOW, FAST)):
            lines[case].set_ydata(weights[frame, case])
            references[case].set_ydata(expected[frame])
            gates[case].set_array(np.broadcast_to(regime[frame], (2, len(sites))))
            fills[case].remove()
            fills[case] = axes[case].fill_between(sites, weights[frame, case], color=colour, alpha=0.25, linewidth=0)
            traces[case].set_data(progress[:frame + 1], leakage[:frame + 1, case])
        cursor.set_xdata([fraction, fraction])
        yield capture(figure)
    plt.close(figure)


def draw_transport(progress: np.ndarray, history: core.Majorana, potentials: core.Scalar,
                   hopping: float) -> Figure:
    """The mode weight along the wire, the gate's motion upward, one panel per speed, the slow one on the left;
    the gate's topological threshold dashed."""
    weights = history.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    sites = np.arange(weights.shape[-1])
    figure, axes = plt.subplots(1, weights.shape[1], figsize=(2.5, 1.65), layout="constrained")
    # One row per frame, the start at the bottom, half a frame beyond the first and last.
    half_row = (progress[1] - progress[0]) / 2
    extent = (sites[0] - 0.5, sites[-1] + 0.5, progress[0] - half_row, progress[-1] + half_row)
    for case, axis in enumerate(axes):
        weight_image(axis, weights[::-1, case], extent, 0.5)
        axis.contour(sites, progress, chemical, levels=[2 * hopping], colors="white",
                     linewidths=0.5, linestyles="--", alpha=0.7)
        axis.set(xticks=[], yticks=[])
        axis.set_xlabel("site", fontsize=FONT_SIZE)
    axes[0].set_ylabel("fraction of gate motion", fontsize=FONT_SIZE)
    return figure


def animate_overlap(separations: np.ndarray, splitting: core.Scalar, potentials: core.Scalar,
                    modes: core.Majorana, hopping: float) -> Iterator[np.ndarray]:
    """The two end modes at each separation of the segment's ends, beside the lowest excitation energy against
    the separation, marked at the separation of the moment."""
    energies = splitting.to_array() / hopping
    weights = modes.scalar_norm_squared().to_array()
    regime = shade(potentials.to_array(), hopping)
    sites = np.arange(weights.shape[-1])
    top = weights.max() * 1.1
    figure, (spectrum, profile) = plt.subplots(1, 2, figsize=(4.2, 1.9), dpi=90, width_ratios=[1, 1.1],
                                               layout="constrained")
    spectrum.semilogy(separations, energies, color=INK, linewidth=LINE_WIDTH)
    marker = spectrum.scatter([], [], color=INK, edgecolors="white", s=18, linewidths=0.5, zorder=3)
    # The ends of the separations swept; the energy's decades show the exponential.
    spectrum.set(xlabel="gate separation (sites)", ylabel=r"lowest excitation, $E/t$",
                 xticks=[separations.min(), separations.max()])
    plain_axes(spectrum)
    # A two-row mesh shaded between its vertices fills the panel with the gate profile.
    shading = profile.pcolormesh(sites, [0, top], np.broadcast_to(regime[0], (2, len(sites))),
                                 shading="gouraud", cmap=GATE, vmin=0, vmax=1, zorder=0)
    lines = [profile.plot(sites, density, color=colour, linewidth=0.8)[0]
             for density, colour in zip(weights[0], (SLOW, FAST))]
    fills = [profile.fill_between(sites, density, color=colour, alpha=0.35, linewidth=0)
             for density, colour in zip(weights[0], (SLOW, FAST))]
    profile.set(xlim=(sites[-1] - separations.max() - 20, sites[-1]), ylim=(0, top),
                xlabel="site", ylabel="mode weight", xticks=[], yticks=[])
    plain_axes(profile)
    # Lay the figure out once; only the marker, the gate and the modes change between frames.
    figure.canvas.draw()
    figure.set_layout_engine("none")

    for frame, separation in enumerate(separations):
        marker.set_offsets([[separation, energies[frame]]])
        shading.set_array(np.broadcast_to(regime[frame], (2, len(sites))))
        for end, colour in enumerate((SLOW, FAST)):
            lines[end].set_ydata(weights[frame, end])
            fills[end].remove()
            fills[end] = profile.fill_between(sites, weights[frame, end], color=colour, alpha=0.35, linewidth=0)
        yield capture(figure)
    plt.close(figure)
