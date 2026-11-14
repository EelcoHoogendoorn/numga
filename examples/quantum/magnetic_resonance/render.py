"""Drawing for magnetic resonance renders Bloch-ball trajectories on the GPU and absorption lines,
dispersion lines, and spin-echo signals in matplotlib."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.gatype import Orientation
from numga.render import drawing, simplices, solids
from numga.topology import Topology
from examples.quantum.magnetic_resonance import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import labels, offset_panel, overlay, show

colours = ("#c0392b", "#7d3c98", "#2e86c1")
sphere_colour, sphere_opacity = np.array([0.70, 0.76, 0.86]), 0.09
axis_colour = np.array([0.45, 0.47, 0.52])
circle_colour, circle_opacity = np.full(3, 0.75), 0.55
# Camera elevation and azimuth in radians, half-width of the square view, and pixel width.
elevation, azimuth = np.radians(18.0), np.radians(40.0)
reach, n_pixels = 1.3, 300
# Stroke widths and marker radii in pixels, axis label font size in points, and axis extents in ball radii.
circle_width, path_width, arrow_width, tip_radius, dot_radius = 1.0, 1.4, 1.0, 1.6, 2.6
label_size = 8
axis_length, label_distance = 1.18, 1.26
# Quantity plot pixel dimensions: side-by-side absorption/dispersion lines, single signal trace, and pinned echo GIF.
lines_px, signal_px = (640, 280), (440, 250)
echo_px = (603, 281)
# Echo trace duration in microseconds from the first pulse, equator ink, and rim sample count.
echo_span = 8.0
equator_colour, n_rim_samples = np.full(3, 0.8), 119


def bloch_components(states: core.State) -> np.ndarray:
    """Pairs the spin part of `states` with the turning planes `yz`, `zx`, and `xy` and doubles the
    readings into `[..., 3]` Bloch coordinates across both spacetime and spatial representations."""
    mv = states.context.multivector
    spin_planes = stack((mv.yz, mv.zx, mv.xy))                           # [3] Bivector+
    return 2 * (spin_planes & states.select.antibivector()[..., None]).to_array()


def bloch_points(states: core.State) -> solids.Point:
    """Places each state's spin inside the GPU Bloch ball (`[...] Point-`).

    In the spatial twin the Bloch vector is an extrinsic vector whose dual direction embeds directly
    around the ball's origin; in spacetime the intrinsic spin patch is read against `yz`, `zx`, and
    `xy` and laid out along the ball's axes.
    """
    mv = states.context.multivector
    spin = 2 * states.select.antibivector()
    if spin.gatype.orientation is Orientation.extrinsic:
        return solids.origin + solids.context.include_from_algebra(spin.dual())
    readings = stack((mv.yz, mv.zx, mv.xy)) & spin[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)


def nutation_figure(states: Iterable[core.State], steady_states: core.State) -> plt.Figure:
    """Draws each spin trajectory (`states`) nutating inside a translucent Bloch ball and spiralling
    into its stationary dot (`steady_states`)."""
    paths = bloch_points(stack(list(states), axis=-1))                   # [..., ...] Point-
    n_paths, n_times = paths.shape
    path_colours = np.array([to_rgb(colour) for colour in colours[:n_paths]])
    view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, reach), n_pixels)
    # Each spin trajectory forms an open tube chain coloured by its detuning.
    tracks = simplices.tubes(paths.reshape(-1), Topology.chains([n_times] * n_paths),
                             np.repeat(path_colours, n_times - 1, axis=0), path_width, view, gloss=solids.gloss)
    axis_tips = solids.origin + solids.axes.dual() * axis_length         # [3] Point-
    shafts = solids.rods(solids.origin, axis_tips, arrow_width / 2 * view.pixel)
    heads = solids.ball(axis_tips, tip_radius * view.pixel)[:, None]
    parts = [(shafts, axis_colour), (heads, axis_colour),
             (solids.ball(bloch_points(steady_states), dot_radius * view.pixel)[:, None], path_colours)]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    great_circles = solids.Ink(len(parts), solids.half_space(solids.axes), solids.everywhere[None].broadcast_to((3,)),
                               circle_colour, circle_width, circle_opacity)
    figure = plt.figure(figsize=(n_pixels / 100, n_pixels / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    show(ax, solids.draw(parts, view, np.ones(3), [great_circles], translucent=ball, meshes=[(tracks, 1.0)]), view)
    labels(ax, view, solids.origin + solids.axes.dual() * label_distance, list("xyz"), ha="center", va="center",
           fontsize=label_size, color=axis_colour)
    return figure


@charts.style()
def lines_figure(detunings: np.ndarray, drives: np.ndarray, steady_states: core.State) -> plt.Figure:
    """Plots steady-state absorption and dispersion per unit drive against `detunings` for each drive
    strength in `drives`."""
    readings = bloch_components(steady_states)                           # [..., ..., 3]
    figure, (absorbing, dispersing) = charts.subplots(*lines_px, ncols=2, layout="constrained")
    for index, (drive, colour) in enumerate(zip(drives, colours)):
        absorbing.plot(detunings, -readings[:, index, 1] / drive, color=colour)
        dispersing.plot(detunings, readings[:, index, 0] / drive, color=colour)
    absorbing.set_ylabel("absorption, per unit drive")
    dispersing.set_ylabel("dispersion, per unit drive")
    absorbing.set_ylim(bottom=0)
    for ax in (absorbing, dispersing):
        ax.axhline(0, color="black", linewidth=charts.FRAME_WIDTH, zorder=0)
        ax.axvline(0, color="0.85", linewidth=charts.FRAME_WIDTH, zorder=0)
        ax.set(xticks=[], yticks=[], xlabel="detuning")
        ax.spines[["top", "right"]].set_visible(False)
    return figure


def transverse_length(states: core.State) -> np.ndarray:
    """Reads the transverse spin magnitude of `states` (`[...]`) by subtracting a half turn in `xy`."""
    mv = states.context.multivector
    spins = states.select.antibivector()
    transverse = spins - (mv.xy >> spins)
    return transverse.antinorm().to_array()


def coil_signal(ensemble: core.State) -> np.ndarray:
    """Reads the mean transverse spin magnitude of `ensemble` across its trailing spin axis."""
    return transverse_length(ensemble.mean(axis=-1))


@charts.style()
def echo_figure(states: Iterable[core.State], dt: float, delay: float, t2: float) -> plt.Figure:
    """Plots the ensemble's mean transverse spin signal sampled every `dt` alongside the `t2` envelope."""
    ensemble = stack(list(states))                                       # [..., ...] State-
    times = np.arange(ensemble.shape[0]) * dt
    figure, ax = charts.subplots(*signal_px)
    ax.plot(times, coil_signal(ensemble), color="#2e86c1")
    ax.plot(times, np.exp(-times / t2), "--", color="0.5")
    for when, name in ((delay, "π pulse"), (2 * delay, "echo")):
        ax.axvline(when, color="0.7", linewidth=charts.FRAME_WIDTH)
        ax.annotate(name, (when, 0.95), textcoords="offset points", xytext=(4, 0))
    ax.set(xlabel="time", ylabel="mean transverse spin", xlim=(times[0], times[-1]), ylim=(0, 1.05), xticks=[],
           yticks=[])
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


@charts.style()
def decays_figure(times: np.ndarray, echo_states: core.State, decay_states: core.State, t2: float) -> plt.Figure:
    """Plots the refocused `echo_states` and unrefocused `decay_states` amplitudes against `times` on a
    logarithmic time axis alongside the `t2` envelope."""
    fine_times = np.geomspace(times[0], times[-1], 200)
    figure, ax = charts.subplots(*signal_px)
    ax.semilogx(times, transverse_length(echo_states), "o-", color="#2e86c1")
    ax.semilogx(times, transverse_length(decay_states), "o-", color="#c0392b")
    ax.semilogx(fine_times, np.exp(-fine_times / t2), "--", color="0.5")
    ax.set(xlabel="time after the first pulse (µs)", ylabel="transverse spin", ylim=(0, 1.05), yticks=[])
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure


def echo_frames(detunings: np.ndarray, dt: float, delay: float, t2: float, frame_stride: int,
                states: Iterable[core.State]) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames of the spins fanning out and refocusing from above the
    magnet axis alongside the mean transverse signal trace every `frame_stride` recorded steps."""
    span = np.linspace(0.0, echo_span)
    shades = plt.get_cmap("coolwarm")(plt.Normalize()(detunings))[:, :3]   # [..., 3]
    states = iter(states)
    first = next(states)
    mv = first.context.multivector
    # Pairing with twice `yz` and `zx` reads transverse `x` and `y` coordinates directly in both algebras.
    across, up = 2 * mv.yz, 2 * mv.zx
    unit = across.dual() / (across & across.dual())
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False) / 2)).exp()
    equator = turns >> unit                                                # [n_rim_samples]
    with charts.style():
        figure, (top, trace) = charts.subplots(*echo_px, ncols=2, width_ratios=(1, 1.4))
        top.set_xlim(-1.1, 1.1)
        top.set_ylim(-1.1, 1.1)
        top.set_aspect("equal")
        top.set_axis_off()
        so_far, = trace.plot([], [], color="#2e86c1")
        trace.plot(span, np.exp(-span / t2), "--", color="0.8")
        trace.axvline(delay, color="0.85", linewidth=charts.FRAME_WIDTH)
        trace.annotate("π pulse", (delay, 0.95), textcoords="offset points", xytext=(4, 0))
        trace.set_xlim(0.0, echo_span)
        trace.set(ylim=(0, 1.05), xticks=[], yticks=[], xlabel="time", ylabel="mean transverse spin")
        trace.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        camera = offset_panel(top, across, up)
        rim = simplices.strokes(equator, Topology.loops([n_rim_samples]), equator_colour, charts.LINE_PX,
                                camera)
        spins = overlay(top, [rim])
        times, strength = [], []
        for index, rho in enumerate(chain((first,), states)):
            times.append(index * dt)
            strength.append(coil_signal(rho))
            if index % frame_stride == 0:
                spins.set_data(drawing.composited([
                    rim, simplices.markers(rho.select.antibivector(), shades, charts.MARKER_PX / 2, camera)]))
                so_far.set_data(times, strength)
                yield capture(figure)
        plt.close(figure)
