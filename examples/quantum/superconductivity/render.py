"""Draws GPU-shaded pseudospin coherence arrows alongside matplotlib charts of the equilibrium band and collective gap."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import stack
from examples.util.animation import capture
from examples.quantum.superconductivity import core
from examples.util import charts
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import Scene, meshes, simplices, solids
from examples.util.panels import fitted, labels, show

sm = solids.mv
# Intrinsic patches of the planes xy and yz, paired with a field to read its components.
xy_patch = core.mv.xy.dual()                          # Antibivector-
yz_patch = core.mv.yz.dual()                          # Antibivector-

ink_colour = "#263b51"
pair_colour = "#198b87"
fixed_colour = "#d17c24"
n_display_levels = 33
display_cutoff = 3.0
axis_colour = np.full(3, 0.55)
circle_colour = np.full(3, 0.75)
circle_opacity = 0.4
path_opacity = 0.3
circle_radius = 0.5
circle_stride = 8

energy_scale = 0.4
energy_start_value = -3.4
energy_end_value = 3.7
coherence_reach = 0.68
# Azimuth zero in solids.view looks along +y, so +90 degrees measures the camera angle from +x.
elevation = np.radians(24.0)
azimuth = np.radians(-58.0 + 90.0)
margin = 0.08
n_pixels = 470

energy_start = solids.origin + sm.x.dual() * (energy_scale * energy_start_value)  # [] Point-
reaches = np.array([energy_scale * (energy_end_value - energy_start_value), coherence_reach, coherence_reach])

n_circle_points = 100
axis_width, circle_width, path_width, stem_width = 0.8, 0.64, 0.8, 1.43
chevron_length = 4.0
quench_px = (640, 270)


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes, across: str, up: str) -> None:
    """Formats `axis` with unnumbered spines through zero and places the horizontal label below the frame."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.spines["bottom"].set_position(("data", 0))
    axis.set(xticks=[], yticks=[], xlabel=across, ylabel=up)
    axis.xaxis.set_label_coords(0.5, -0.03)
    axis.xaxis.label.set_va("top")


def coherences(spins: core.Spin, planes: tuple[core.Field, ...]) -> np.ndarray:
    """Reads the pseudospins' components by pairing against `planes` (`[..., n_levels, n_planes]`)."""
    return (stack(planes) & spins.batch()[..., None]).to_array()


def readings(fields: core.Field, patch: core.Spin) -> np.ndarray:
    """Reads a field's component across the intrinsic `patch` of one plane."""
    return (fields & patch).to_array()


@charts.style()
def equilibrium_figure(dispersion: core.Field, spins: core.Spin) -> Figure:
    """Plots ground-state electron occupation and pairing coherence across the energy band."""
    energies = readings(dispersion, xy_patch)
    coherence = coherences(spins, (core.mv.yz, core.mv.zx, core.mv.xy))
    figure, axes = charts.subplots(640, 250, ncols=2, layout="constrained")
    axes[0].plot(energies, coherence[:, 2] + 0.5, color=ink_colour)
    axes[0].set(ylim=(0, 1.04))
    axes[1].plot(energies, coherence[:, 0], color=pair_colour)
    axes[1].set(ylim=(0, 0.54))
    for axis, up in zip(axes, ("occupation per electron state", r"pair coherence in yz, $s_{yz}$")):
        axis.set(xlim=(energies[0], energies[-1]))
        axis.axvline(0, color="0.85", linewidth=charts.FRAME_WIDTH, zorder=0)
        plain_axes(axis, r"energy from chemical potential, $\xi$", up)
    return figure


def chart_scene(energies: np.ndarray, axis: plt.Axes) -> tuple[solids.View, Draw, solids.Point]:
    """Builds the fitted 3D view, coordinate-axis arrows, and reference coherence circles (`[..., n_circle_points] Point-`)."""
    ends = energy_start + solids.axes.dual() * reaches                                      # [3] Point-
    circle_energies = energies[::circle_stride]
    n_circles = len(circle_energies)
    levels = solids.origin + sm.x.dual() * (energy_scale * circle_energies)                 # [n_circles] Point-
    circles = meshes.circles(levels + sm.z.dual() * circle_radius, sm.yz, n_circle_points)  # [n_circles, n_circle_points] Point-
    view = fitted(solids.view([circles, energy_start, ends], 0.0, elevation, azimuth, margin, n_pixels), axis)
    starts = energy_start[None].broadcast_to(ends.shape)                                    # [3] Point-
    axes = simplices.tubes(
        stack((starts, ends), axis=-1).reshape(-1), Topology.chains([2] * 3), axis_colour,
        axis_width, view, solids.gloss, chevrons=np.ones(3), chevron=chevron_length * axis_width,
    )
    return view, axes, circles


def quench_animation(
    dispersion: core.Field, times: np.ndarray, states: Iterable[tuple[core.Spin, core.Field]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the sampled pseudospin coherence arrows alongside the evolving gap amplitude."""
    energies = readings(dispersion, xy_patch)
    visible = np.flatnonzero(np.abs(energies) < display_cutoff)
    indices = visible[np.linspace(0, len(visible) - 1, n_display_levels, dtype=int)]
    sampled_energies = energies[indices]
    origins = solids.origin + sm.x.dual() * (energy_scale * sampled_energies)               # [n_display_levels] Point-
    colours = plt.colormaps["coolwarm"](np.linspace(0.08, 0.92, n_display_levels))[:, :3]

    with charts.style():
        figure = plt.figure(figsize=(quench_px[0] / charts.DPI, quench_px[1] / charts.DPI))
        try:
            axis = figure.add_axes((0.01, 0.08, 0.57, 0.85))
            curve = figure.add_axes((0.68, 0.22, 0.29, 0.64))
            view, axes, circles = chart_scene(sampled_energies, axis)
            rings = (
                simplices.tubes(
                    circles.reshape(-1), Topology.loops([circles.shape[1]] * len(circles)),
                    circle_colour, circle_width, view, solids.gloss,
                ),
                circle_opacity,
            )
            scene = Scene(view, np.ones(3), [], [rings, (axes, 1.0)], inks=[], translucent=[])

            def arrow_picture(coherence: np.ndarray) -> np.ndarray:
                """Draws each sampled level's coherence arrow from the energy axis and connects the tips into a curve."""
                tips = origins + sm.y.dual() * coherence[:, 0] + sm.z.dual() * coherence[:, 1]  # [n_display_levels] Point-
                stems = simplices.tubes(
                    stack((origins, tips), axis=-1).reshape(-1), Topology.chains([2] * n_display_levels),
                    colours, stem_width, view, solids.gloss, chevrons=np.ones(n_display_levels),
                    chevron=chevron_length * stem_width,
                )
                path = simplices.tubes(
                    tips, Topology.chains([n_display_levels]), np.array(to_rgb(ink_colour)),
                    path_width, view, solids.gloss,
                )
                return scene.draw([], inks=[], translucent=[], meshes=[(stems, 1.0), (path, path_opacity)])

            state_stream = iter(states)
            first_spins, first_gap = next(state_stream)
            first_coherence = coherences(first_spins, (core.mv.yz, core.mv.zx))[indices]
            first_amplitude = float(first_gap.norm().to_array())
            amplitudes = [first_amplitude]

            image = show(axis, arrow_picture(first_coherence), view)
            label_points = energy_start + solids.axes.dual() * (reaches + [0.12, 0.1, 0.1])  # [3] Point-
            labels(axis, view, label_points, [r"$\xi$", r"$s_{yz}$", r"$s_{zx}$"], ha="center", va="center", color=axis_colour)
            gap_line, = curve.plot(times[:1], amplitudes, color=pair_colour)
            gap_tip, = curve.plot([times[0]], [first_amplitude], "o", color=pair_colour)
            # Frame the vertical axis slightly above the initial post-quench gap before it drops.
            curve.set(xlim=(times[0], times[-1]), ylim=(0, first_amplitude * 1.12))
            plain_axes(curve, "time", r"$|\Delta|$")
            yield capture(figure)

            for index, (spins, gap) in enumerate(state_stream, start=1):
                coherence = coherences(spins, (core.mv.yz, core.mv.zx))[indices]
                amplitude = float(gap.norm().to_array())
                amplitudes.append(amplitude)
                image.set_data(arrow_picture(coherence))
                gap_line.set_data(times[:index + 1], amplitudes)
                gap_tip.set_data([times[index]], [amplitude])
                yield capture(figure)
        finally:
            plt.close(figure)


@charts.style()
def gap_figure(times: np.ndarray, gaps: core.Field, equilibrium_gap: core.Field) -> Figure:
    """Plots the post-quench gap amplitude ringing down above the final ground-state gap."""
    amplitudes = gaps.norm().to_array()
    equilibrium = equilibrium_gap.norm().to_array()
    figure, axis = charts.subplots(540, 240, layout="constrained")
    axis.plot(times, amplitudes, color=pair_colour, label="isolated condensate")
    axis.axhline(equilibrium, color=ink_colour, linestyle="--", label="final ground-state gap")
    axis.set(xlim=(times[0], times[-1]), ylim=(0, amplitudes.max() * 1.05))
    axis.legend(frameon=False, ncol=2, loc="lower center")
    plain_axes(axis, "time", r"$|\Delta|$")
    return figure


@charts.style()
def response_figure(
    times: np.ndarray, exact: core.Field, predicted: core.Field, names: tuple[str, str],
) -> Figure:
    """Compares the nonlinear gap oscillation after a weak quench with its collective and fixed-field linearizations."""
    exact_amplitude = readings(exact, yz_patch)
    predicted_amplitude = readings(predicted, yz_patch)
    figure, axis = charts.subplots(600, 260, layout="constrained")
    # Draw the nonlinear trace three curves wide at half opacity so the dashed predictions stay legible over it.
    axis.plot(
        times, exact_amplitude, color=ink_colour, linewidth=charts.points(3 * charts.LINE_PX), alpha=0.5,
        label="nonlinear evolution",
    )
    for values, name, colour in zip(predicted_amplitude.T, names, (pair_colour, fixed_colour)):
        axis.plot(times, values, color=colour, linestyle="--", label=name)
    axis.set(xlim=(times[0], times[-1]))
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    plain_axes(axis, "time", r"$\delta\Delta_{yz}$")
    return figure
