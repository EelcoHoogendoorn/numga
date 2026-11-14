"""Spatial Majorana mode weights, excitation energies, and transport leakage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.figure import Figure

from examples.util.animation import capture
from examples.util.panels import pinned
from examples.quantum.kitaev_wire import core
from examples.util import charts

ink_colour = "#263b51"
slow_colour = "#148d88"
fast_colour = "#d06c36"
gate_cmap = LinearSegmentedColormap.from_list("gate", ["white", "#e8e4f5"])
weights_cmap = plt.get_cmap("magma")
# The mode weight at the top of the transport panels: above the most any one site holds, as the fast move squeezes
# the mode against the moving end.
weight_limit = 0.6


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes) -> None:
    """A quantity plot shows faint left and bottom spines only."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.tick_params(color="0.8")


def topological_depth(chemical: np.ndarray, hopping: float) -> np.ndarray:
    """Each site's topological depth runs from zero at threshold `2 * hopping` to one at the lowest chemical potential."""
    return np.clip((2 * hopping - chemical) / (2 * hopping - chemical.min()), 0, 1)


def weight_image(axis: plt.Axes, weights: np.ndarray, extent: tuple[float, float, float, float], gamma: float):
    """Mode weights draw rows down from the top, one pixel per site and sample, keeping the axis labels."""
    colours = weights_cmap(PowerNorm(gamma, vmin=0, vmax=weights.max())(weights))[..., :3]
    image = pinned(axis, colours, extent)
    axis.set_axis_on()
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0)
    return image


@charts.style()
def formation_figure(chemical_potentials: np.ndarray, energies: core.Scalar,
                     density: core.Scalar, hopping: float) -> Figure:
    """Excitation energies beside spatial mode weights across a chemical-potential sweep."""
    values = energies.to_array() / hopping
    weights = density.to_array()
    ratios = chemical_potentials / hopping
    # The spectrum beside the mode weights, a pixel a site and a sample, tall enough for their samples.
    figure, (spectrum, modes) = charts.subplots(560, 280, ncols=2, width_ratios=[2.4, 1], layout="constrained")
    # Each excitation has two real quadratures with the same frequency.
    spectrum.plot(ratios, values[:, ::2], color=ink_colour)
    spectrum.axvline(2, color="0.65", linestyle=":", linewidth=charts.FRAME_WIDTH)
    # Read off: the threshold at twice the hopping, and zero energy.
    spectrum.set(xlim=(ratios[0], ratios[-1]), ylim=(0, None), xticks=[2], yticks=[0],
                 xlabel="chemical potential inside, μ/t", ylabel="excitation energy, E/t")
    plain_axes(spectrum)
    # One row per chemical potential, the outside value on top, half a sample beyond the first and last.
    half_row = (ratios[0] - ratios[-1]) / (len(ratios) - 1) / 2
    weight_image(modes, weights, (-0.5, weights.shape[-1] - 0.5, ratios[-1] - half_row, ratios[0] + half_row), 0.55)
    modes.axhline(2, color="white", linestyle=":", linewidth=charts.FRAME_WIDTH, alpha=0.6)
    modes.set(xticks=[], yticks=[2], xlabel="site", ylabel="μ/t")
    return figure


def transport_animation(
    progress: np.ndarray, potentials: core.Scalar,
    target: core.Majorana, hopping: float, states: Iterable[tuple[core.Majorana, core.Scalar]],
) -> Iterator[np.ndarray]:
    """Yields frames comparing slow and fast Majorana transport against the instantaneous boundary mode and bulk leakage."""
    expected = target.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    sites = np.arange(expected.shape[-1])
    regime = topological_depth(chemical, hopping)
    leakages: list[np.ndarray] = []

    with charts.style():
        # Three strips stacked, each with its two labels: the two speeds' modes along the wire, and the bulk weight.
        figure, axes = charts.subplots(360, 400, nrows=3, height_ratios=[1, 1, 0.7], layout="constrained")
        lines, references, gates, fills = [], [], [], []
        for case, (axis, colour) in enumerate(zip(axes[:2], (slow_colour, fast_colour))):
            # A two-row pcolormesh shaded between its vertices fills the panel with the gate profile.
            gates.append(axis.pcolormesh(sites, [0, weight_limit], np.broadcast_to(regime[0], (2, len(sites))),
                                         shading="gouraud", cmap=gate_cmap, vmin=0, vmax=1, zorder=0))
            line, = axis.plot(sites, expected[0], color=colour)
            # The adiabatic target thin, so that the mode's own curve shows under it.
            reference, = axis.plot(sites, expected[0], color=ink_colour, linestyle="--", linewidth=charts.FRAME_WIDTH)
            fills.append(axis.fill_between(sites, expected[0], color=colour, alpha=0.25, linewidth=0))
            lines.append(line)
            references.append(reference)
            axis.set(xlim=(sites[0], sites[-1]), ylim=(0, weight_limit), xticks=[], yticks=[], xlabel="site",
                     ylabel="mode weight")
            plain_axes(axis)
        traces = [axes[2].plot([], [], color=colour)[0] for colour in (slow_colour, fast_colour)]
        cursor = axes[2].axvline(0, color="0.7", linewidth=charts.FRAME_WIDTH)
        # The bulk weight from none to all of the mode, along the gate's motion.
        axes[2].set(xlim=(0, 1), ylim=(0, 1.02), xticks=[], yticks=[0, 1], xlabel="fraction of gate motion",
                    ylabel="bulk weight")
        plain_axes(axes[2])
        # Lay the figure out once; only the curves, the gate, and the cursor change between frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")

        for frame, (fraction, (state, share)) in enumerate(zip(progress, states)):
            weights = state.scalar_norm_squared().to_array()
            leakages.append(1 - share.to_array())
            leakage_history = np.asarray(leakages)
            for case, colour in enumerate((slow_colour, fast_colour)):
                lines[case].set_ydata(weights[case])
                references[case].set_ydata(expected[frame])
                gates[case].set_array(np.broadcast_to(regime[frame], (2, len(sites))))
                fills[case].remove()
                fills[case] = axes[case].fill_between(sites, weights[case], color=colour, alpha=0.25,
                                                      linewidth=0)
                traces[case].set_data(progress[:frame + 1], leakage_history[:, case])
            cursor.set_xdata([fraction, fraction])
            yield capture(figure)
        plt.close(figure)


@charts.style()
def transport_figure(progress: np.ndarray, history: core.Majorana, potentials: core.Scalar,
                     hopping: float) -> Figure:
    """Space-time mode weights along the wire for slow and fast gate motion, with the topological threshold dashed."""
    weights = history.scalar_norm_squared().to_array()
    chemical = potentials.to_array()
    sites = np.arange(weights.shape[-1])
    # Two images side by side, a pixel a site and a frame, and room for their labels.
    figure, axes = charts.subplots(260, 290, ncols=weights.shape[1], layout="constrained")
    # One row per frame, the start at the bottom, half a frame beyond the first and last.
    half_row = (progress[1] - progress[0]) / 2
    extent = (sites[0] - 0.5, sites[-1] + 0.5, progress[0] - half_row, progress[-1] + half_row)
    for case, axis in enumerate(axes):
        weight_image(axis, weights[::-1, case], extent, 0.5)
        axis.contour(sites, progress, chemical, levels=[2 * hopping], colors="white",
                     linewidths=charts.FRAME_WIDTH, linestyles="--", alpha=0.7)
        axis.set(xticks=[], yticks=[], xlabel="site")
    axes[0].set_ylabel("fraction of gate motion")
    return figure


def overlap_animation(separations: np.ndarray, splitting: core.Scalar, potentials: core.Scalar,
                      modes: core.Majorana, hopping: float) -> Iterator[np.ndarray]:
    """Yields frames showing the two end modes at each gate separation beside the lowest excitation energy."""
    energies = splitting.to_array() / hopping
    weights = modes.scalar_norm_squared().to_array()
    regime = topological_depth(potentials.to_array(), hopping)
    sites = np.arange(weights.shape[-1])
    top = weights.max() * 1.1
    with charts.style():
        # Three quarters of the column: the energy's decades beside the two modes.
        figure, (spectrum, profile) = charts.subplots(600, 270, ncols=2, width_ratios=[1, 1.1], layout="constrained")
        spectrum.semilogy(separations, energies, color=ink_colour)
        # The separation of the moment, a dot twice the usual size on a white rim, to stand out on its curve.
        marker = spectrum.scatter([], [], color=ink_colour, edgecolors="white", s=charts.points(2 * charts.MARKER_PX) ** 2,
                                  linewidths=charts.FRAME_WIDTH, zorder=3)
        # The ends of the separations swept; the energy's decades show the exponential.
        spectrum.set(xlabel="gate separation (sites)", ylabel="lowest excitation, E/t",
                     xticks=[separations.min(), separations.max()])
        plain_axes(spectrum)
        # A two-row pcolormesh shaded between its vertices fills the panel with the gate profile.
        shading = profile.pcolormesh(sites, [0, top], np.broadcast_to(regime[0], (2, len(sites))),
                                     shading="gouraud", cmap=gate_cmap, vmin=0, vmax=1, zorder=0)
        lines = [profile.plot(sites, density, color=colour)[0]
                 for density, colour in zip(weights[0], (slow_colour, fast_colour))]
        fills = [profile.fill_between(sites, density, color=colour, alpha=0.35, linewidth=0)
                 for density, colour in zip(weights[0], (slow_colour, fast_colour))]
        profile.set(xlim=(sites[-1] - separations.max() - 20, sites[-1]), ylim=(0, top),
                    xlabel="site", ylabel="mode weight", xticks=[], yticks=[])
        plain_axes(profile)
        # Lay the figure out once; only the marker, the gate, and the modes change between frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")

        for frame, separation in enumerate(separations):
            marker.set_offsets([[separation, energies[frame]]])
            shading.set_array(np.broadcast_to(regime[frame], (2, len(sites))))
            for end, colour in enumerate((slow_colour, fast_colour)):
                lines[end].set_ydata(weights[frame, end])
                fills[end].remove()
                fills[end] = profile.fill_between(sites, weights[frame, end], color=colour, alpha=0.35, linewidth=0)
            yield capture(figure)
        plt.close(figure)
